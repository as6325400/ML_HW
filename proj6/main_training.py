import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import cv2
import seaborn as sns
from PIL import Image
import glob
from collections import Counter

# 訓練參數 - 調整以減少記憶體使用
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs_per_fold = 50
base_lr = 2e-5
max_lr = 2e-4
batch_size = 256  # 減小批量大小以避免 OOM 錯誤
num_folds = 3
num_classes = 12  # 假設有 12 個植物種子類別
use_mixed_precision = True  # 使用混合精度訓練以加速
img_size = 224  # 減小圖像尺寸以節省記憶體
model_name = 'vit_base_patch16_224'  # 使用較小的模型

# 路徑設定
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "plant-seedlings-classification", "train")
weight_dir = os.path.join(base_path, "weights5")
result_dir = os.path.join(base_path, "result5")
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 背景移除函數 (簡單 HSV 顏色閾值)
def remove_background(image, **kwargs):
    """背景移除函數，接受任意額外參數以兼容 Albumentations Lambda"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        result = image.copy()
        result[mask == 0] = 0  # 背景設為黑色
        return result
    except Exception as e:
        print(f"背景移除失敗: {str(e)}")
        return image  # 發生錯誤時返回原始圖像

# 資料增強管道 (使用 Albumentations) - 增強數據多樣性
train_transform = A.Compose([
    A.Resize(img_size, img_size),  # 減小圖像尺寸以節省記憶體
    A.Lambda(image=remove_background, p=1.0),  # 背景移除
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),  # 增加旋轉範圍和概率，模擬不同角度
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),  # 增加亮度和對比度變化
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.7),  # 增加顏色抖動，幫助區分類別
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.CoarseDropout(p=0.5),  # 使用默認參數
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Lambda(image=remove_background, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 自定義數據集類 - 添加錯誤處理
class PlantSeedlingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 檢查文件是否存在
            if not os.path.exists(img_path):
                print(f"警告: 找不到圖像 {img_path}")
                # 創建空白圖像作為替代
                image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"警告: 無法讀取圖像 {img_path}")
                    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            if self.transform:
                try:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                except Exception as e:
                    print(f"轉換時出錯: {str(e)}")
                    # 如果轉換失敗，使用基本轉換
                    image = torch.from_numpy(
                        cv2.resize(image, (img_size, img_size))
                    ).permute(2, 0, 1).float() / 255.0
            
            return image, label
        except Exception as e:
            print(f"在 __getitem__ 中發生錯誤: {str(e)}, idx={idx}")
            # 返回零張量作為替代
            return torch.zeros((3, img_size, img_size)), label

# 獲取數據集 - 包含類別分佈統計和過採樣
def load_data(train_data_path):
    try:
        class_dirs = sorted(os.listdir(train_data_path))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        image_paths = []
        labels = []
        class_counts = {cls_name: 0 for cls_name in class_dirs}
        
        for cls_name in class_dirs:
            cls_path = os.path.join(train_data_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
                
            imgs = glob.glob(os.path.join(cls_path, "*.png"))
            if not imgs:
                imgs = glob.glob(os.path.join(cls_path, "*.jpg"))
            
            class_counts[cls_name] = len(imgs)
            for img_name in imgs:
                image_paths.append(img_name)
                labels.append(class_to_idx[cls_name])
                
        print(f"原始類別分佈: {class_counts}")
        
        # 過採樣 Black-grass
        # max_count = max(class_counts.values())
        # black_grass_idx = class_to_idx['Black-grass']
        # black_grass_paths = [p for p, l in zip(image_paths, labels) if l == black_grass_idx]
        # black_grass_labels = [black_grass_idx] * len(black_grass_paths)
        # repeat_factor = max_count // len(black_grass_paths) + 1
        # image_paths.extend(black_grass_paths * (repeat_factor - 1))  # 過採樣至接近 max_count
        # labels.extend(black_grass_labels * (repeat_factor - 1))
        
         # 加入偽標籤數據
        pseudo_label_file = os.path.join(result_dir, "pseudo_labels.csv")
        if os.path.exists(pseudo_label_file):
            pseudo_df = pd.read_csv(pseudo_label_file)
            pseudo_paths = pseudo_df['image_path'].tolist()
            pseudo_labels = pseudo_df['pseudo_label'].tolist()
            image_paths.extend(pseudo_paths)
            labels.extend(pseudo_labels)
            print(f"加入了 {len(pseudo_paths)} 個偽標籤數據")
        else:
            print(f"警告: 找不到偽標籤檔案 {pseudo_label_file}")
        
        updated_counts = Counter(labels)
        print(f"過採樣後類別分佈: {updated_counts}")
        print(f"載入了 {len(image_paths)} 張圖像，共 {len(class_to_idx)} 個類別")
        return image_paths, labels, class_to_idx
    except Exception as e:
        print(f"載入數據集時出錯: {str(e)}")
        return [], [], {}

# Focal Loss 實現 - 包含類別加權
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', weight=self.class_weights)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 訓練函數 - 修正混合精度訓練 (保持不變)
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, fold, scaler=None):
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_loss_list, valid_loss_list = [], []
    train_acc_list, valid_acc_list = [], []
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs} - Fold {fold+1}')
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for data, target in tqdm(train_loader, desc="Training"):
            try:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                if use_mixed_precision and scaler:
                    try:
                        # 新版 PyTorch 的正確用法
                        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                            output = model(data)
                            loss = criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    except (AttributeError, TypeError):
                        # 舊版 PyTorch 的用法
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                train_loss += loss.item() * data.size(0)
                _, preds = torch.max(output, 1)
                train_correct += torch.sum(preds == target.data)
                
                if scheduler:
                    scheduler.step()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("警告: CUDA 記憶體不足，跳過此批次")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                
        train_loss /= len(train_loader.dataset)
        train_acc = float(train_correct) / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc="Validation"):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    if use_mixed_precision:
                        try:
                            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                                output = model(data)
                                loss = criterion(output, target)
                        except (AttributeError, TypeError):
                            with torch.cuda.amp.autocast():
                                output = model(data)
                                loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                        
                    valid_loss += loss.item() * data.size(0)
                    _, preds = torch.max(output, 1)
                    valid_correct += torch.sum(preds == target.data)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("警告: CUDA 記憶體不足，跳過此批次")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                        
        valid_loss /= len(valid_loader.dataset)
        valid_acc = float(valid_correct) / len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(weight_dir, f'best_model_fold_{fold}.pth'))
    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, best_model_wts

# 主訓練循環 (部分代碼，重點在損失函數調整)
def main():
    try:
        # 設置混合精度訓練
        if use_mixed_precision:
            try:
                # 新版 PyTorch
                scaler = torch.amp.GradScaler(device_type='cuda' if torch.cuda.is_available() else 'cpu')
            except (TypeError, ValueError, AttributeError):
                # 舊版 PyTorch
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
            
        # 載入數據
        image_paths, labels, class_to_idx = load_data(train_data_path)
        if not image_paths:
            print("錯誤: 未能載入任何圖像，請檢查數據路徑")
            return
            
        labels = np.array(labels)
        # 交叉驗證
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        all_train_losses, all_valid_losses = [], []
        all_train_accs, all_valid_accs = [], []
        for fold, (train_idx, valid_idx) in enumerate(skf.split(image_paths, labels)):
            print(f'\nTraining Fold {fold+1}/{num_folds}')
            
            # 準備數據集
            train_paths = [image_paths[i] for i in train_idx]
            train_labels = labels[train_idx]
            valid_paths = [image_paths[i] for i in valid_idx]
            valid_labels = labels[valid_idx]
            train_dataset = PlantSeedlingDataset(train_paths, train_labels, transform=train_transform)
            valid_dataset = PlantSeedlingDataset(valid_paths, valid_labels, transform=valid_transform)
            # 減少 num_workers 以避免記憶體問題
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            # 初始化模型 (使用較小的模型來減輕記憶體負擔)
            print(f"創建模型: {model_name}")
            model = create_model(model_name, pretrained=True, num_classes=num_classes)
            model = model.to(device)
            if fold == 0:
                # 只在第一個 fold 打印模型結構
                print(model)
                
            # 設置類別權重，增加 Black-grass 的權重
            class_weights = torch.ones(num_classes)
            black_grass_idx = class_to_idx['Black-grass']
            class_weights[black_grass_idx] = 2.0  # 增加 Black-grass 的權重
            class_weights = class_weights.to(device)
            
            # 設置優化器和損失函數
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
            criterion = FocalLoss(alpha=1, gamma=2, class_weights=class_weights)
            # ... 其他訓練代碼保持不變 ...
            # 第一階段訓練 (僅頭部)
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=max_lr, 
                epochs=epochs_per_fold//2,
                steps_per_epoch=len(train_loader)
            )
            train_loss1, valid_loss1, train_acc1, valid_acc1, _ = train_model(
                model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                epochs=epochs_per_fold//2, fold=fold, scaler=scaler
            )
            # 解凍主幹進行微調
            for param in model.parameters():
                param.requires_grad = True
                
            optimizer = optim.AdamW(model.parameters(), lr=base_lr/10, weight_decay=1e-4)
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=max_lr/10, 
                epochs=epochs_per_fold,
                steps_per_epoch=len(train_loader)
            )
            train_loss2, valid_loss2, train_acc2, valid_acc2, _ = train_model(
                model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                epochs=epochs_per_fold//2, fold=fold, scaler=scaler
            )
            # 合併結果
            all_train_losses.extend(train_loss1 + train_loss2)
            all_valid_losses.extend(valid_loss1 + valid_loss2)
            all_train_accs.extend(train_acc1 + train_acc2)
            all_valid_accs.extend(valid_acc1 + valid_acc2)
            # 清理記憶體
            del model, optimizer, criterion, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # 繪製訓練曲線
        pd.DataFrame({"train-loss": all_train_losses, "valid-loss": all_valid_losses}).plot()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(result_dir, "Loss_curve.png"))
        plt.close()
        pd.DataFrame({"train-accuracy": all_train_accs, "valid-accuracy": all_valid_accs}).plot()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(result_dir, "Accuracy_curve.png"))
        plt.close()
        
    except Exception as e:
        print(f"訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()