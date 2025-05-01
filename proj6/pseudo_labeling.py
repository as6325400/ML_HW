import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from timm import create_model
import pandas as pd
import cv2
from PIL import Image
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 參數設定 (與 main_training.py 一致)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 8
num_classes = 12
num_folds = 5
img_size = 224
model_name = 'efficientnet_b0'
confidence_threshold = 0.99  # 偽標籤置信度閾值
base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "plant-seedlings-classification", "test")
weight_dir = os.path.join(base_path, "weights")  # 與訓練時一致
result_dir = os.path.join(base_path, "result1")
pseudo_label_path = os.path.join(result_dir, "pseudo_labels.csv")
os.makedirs(result_dir, exist_ok=True)

# 類別名稱 (與訓練時一致)
class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 
               'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 
               'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

# 背景移除函數 (與訓練時相同)
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

# 測試集轉換 (與 valid_transform 一致)
test_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Lambda(image=remove_background, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 測試數據集
class PlantSeedlingTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            if not os.path.exists(img_path):
                print(f"警告: 找不到圖像 {img_path}")
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
                    image = torch.from_numpy(
                        cv2.resize(image, (img_size, img_size))
                    ).permute(2, 0, 1).float() / 255.0
            
            return image, img_path
        except Exception as e:
            print(f"在 __getitem__ 中發生錯誤: {str(e)}, idx={idx}")
            return torch.zeros((3, img_size, img_size)), img_path

# 載入測試數據
def load_test_data(test_data_path):
    try:
        image_paths = glob.glob(os.path.join(test_data_path, "*.png"))
        if not image_paths:
            image_paths = glob.glob(os.path.join(test_data_path, "*.jpg"))
        print(f"載入了 {len(image_paths)} 張測試圖像")
        return sorted(image_paths)  # 確保順序一致
    except Exception as e:
        print(f"載入測試數據集時出錯: {str(e)}")
        return []

# 偽標籤生成
def generate_pseudo_labels(model, test_loader, confidence_threshold):
    model.eval()
    all_preds = np.zeros((len(test_loader.dataset), num_classes))
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader, desc="Predicting for Pseudo-Labels")):
            try:
                data = data.to(device)
                preds = model(data)
                start_idx = i * batch_size
                end_idx = start_idx + data.size(0)
                all_preds[start_idx:end_idx] = preds.cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("警告: CUDA 記憶體不足，跳過此批次")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    softmax_preds = torch.softmax(torch.tensor(all_preds), dim=1).numpy()
    max_probs = np.max(softmax_preds, axis=1)
    pseudo_labels = np.argmax(softmax_preds, axis=1)
    high_conf_idx = np.where(max_probs > confidence_threshold)[0]
    return high_conf_idx, pseudo_labels[high_conf_idx], test_loader.dataset.image_paths

# 主偽標籤流程
def main_pseudo_labeling():
    try:
        test_image_paths = load_test_data(test_data_path)
        if not test_image_paths:
            print("錯誤: 未能載入任何測試圖像，請檢查數據路徑")
            return

        test_dataset = PlantSeedlingTestDataset(test_image_paths, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 使用最佳模型生成偽標籤 (集成多折模型)
        all_pseudo_paths, all_pseudo_labels = [], []
        for fold in range(num_folds):
            weight_path = os.path.join(weight_dir, f'best_model_fold_{fold}.pth')
            if os.path.exists(weight_path):
                print(f"Loading weights from: {weight_path}")
                model = create_model(model_name, pretrained=False, num_classes=num_classes)
                model.load_state_dict(torch.load(weight_path), strict=False)
                model = model.to(device)
                high_conf_idx, pseudo_labels, image_paths = generate_pseudo_labels(
                    model, test_loader, confidence_threshold)
                all_pseudo_paths.extend([image_paths[i] for i in high_conf_idx])
                all_pseudo_labels.extend(pseudo_labels.tolist())
                print(f"Fold {fold}: Generated {len(high_conf_idx)} high-confidence pseudo-labels")
                # 清理記憶體
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 保存偽標籤結果
        pseudo_df = pd.DataFrame({
            'image_path': all_pseudo_paths,
            'pseudo_label': all_pseudo_labels,
            'class_name': [class_names[l] for l in all_pseudo_labels]
        })
        pseudo_df.to_csv(pseudo_label_path, index=False)
        print(f"Pseudo-labels saved with {len(all_pseudo_paths)} high-confidence samples at {pseudo_label_path}")

        # 提示用戶如何使用 pseudo_labels.csv 進行再訓練
        print("To retrain with pseudo-labels, modify main_training.py to include pseudo-labeled data.")
    except Exception as e:
        print(f"偽標籤生成過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_pseudo_labeling()