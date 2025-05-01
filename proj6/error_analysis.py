import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from timm import create_model
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

# 參數設定 (與 main_training.py 一致)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 8
num_classes = 12
img_size = 224
model_name = 'efficientnet_b0'
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "plant-seedlings-classification", "train")
weight_dir = os.path.join(base_path, "weights1")
result_dir = os.path.join(base_path, "result1")
os.makedirs(result_dir, exist_ok=True)

# 類別名稱
class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 
               'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 
               'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

# 背景移除函數 (與 main_training.py 一致)
def remove_background(image, **kwargs):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        result = image.copy()
        result[mask == 0] = 0
        return result
    except Exception as e:
        print(f"背景移除失敗: {str(e)}")
        return image

# 驗證集轉換 (與 main_training.py 一致)
valid_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Lambda(image=remove_background, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 自定義數據集類 (與 main_training.py 一致)
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
            
            return image, label
        except Exception as e:
            print(f"在 __getitem__ 中發生錯誤: {str(e)}, idx={idx}")
            return torch.zeros((3, img_size, img_size)), label

# 獲取數據集 (與 main_training.py 一致)
def load_data(train_data_path):
    try:
        class_dirs = sorted(os.listdir(train_data_path))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        image_paths = []
        labels = []
        for cls_name in class_dirs:
            cls_path = os.path.join(train_data_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            imgs = glob.glob(os.path.join(cls_path, "*.png"))
            if not imgs:
                imgs = glob.glob(os.path.join(cls_path, "*.jpg"))
            for img_name in imgs:
                image_paths.append(img_name)
                labels.append(class_to_idx[cls_name])
        print(f"載入了 {len(image_paths)} 張圖像，共 {len(class_to_idx)} 個類別")
        return image_paths, labels, class_to_idx
    except Exception as e:
        print(f"載入數據集時出錯: {str(e)}")
        return [], [], {}

# Grad-CAM 實現 (修正版)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[:, target_class].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach()  # 使用 detach() 分離梯度計算

# 錯誤分析主函數
def main_error_analysis():
    try:
        # 載入驗證數據 (使用 fold 0 的驗證集)
        image_paths, labels, class_to_idx = load_data(train_data_path)
        if not image_paths:
            print("錯誤: 未能載入任何圖像，請檢查數據路徑")
            return
        labels = np.array(labels)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, valid_idx = list(skf.split(image_paths, labels))[0]
        valid_paths = [image_paths[i] for i in valid_idx]
        valid_labels = labels[valid_idx]
        valid_dataset = PlantSeedlingDataset(valid_paths, valid_labels, transform=valid_transform)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 載入模型 (使用 fold 0 的權重)
        model = create_model(model_name, pretrained=False, num_classes=num_classes)
        weight_path = os.path.join(weight_dir, 'best_model_fold_0.pth')
        if not os.path.exists(weight_path):
            print(f"錯誤: 找不到模型權重 {weight_path}")
            return
        model.load_state_dict(torch.load(weight_path), strict=False)
        model = model.to(device)

        # 獲取目標層 (EfficientNet-B0 的最後卷積層)
        # 注意：根據模型結構調整，可能是 blocks[-1][-1].conv_pwl 或其他層
        target_layer = model.blocks[-1][-1].conv_pwl  # 根據模型結構調整
        grad_cam = GradCAM(model, target_layer)

        # 預測並收集結果
        model.eval()
        all_preds, all_targets = [], []
        misclassified_images = []
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(valid_loader, desc="Validation Prediction")):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                # 收集誤分類的圖像 (最多保存 10 個以節省時間)
                for j in range(data.size(0)):
                    if preds[j] != target[j] and len(misclassified_images) < 10:
                        misclassified_images.append((data[j].cpu(), target[j].item(), preds[j].item(), i * batch_size + j))

        # 混淆矩陣
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
        plt.close()

        # Grad-CAM 可視化誤分類案例
        for img, true_label, pred_label, idx in misclassified_images:
            # 復原原圖
            img_pil = TF.to_pil_image(
                img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) +
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            )

            # 儲存原圖
            img_pil.save(os.path.join(result_dir, f"original_{idx}.png"))

            cam = grad_cam(img.unsqueeze(0).to(device))
            img_pil = TF.to_pil_image(img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
            cam = cam[0, 0].cpu().numpy()  # 已經使用 detach()，可以直接轉為 numpy
            cam = cv2.resize(cam, (img_size, img_size))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            result = cv2.addWeighted(np.array(img_pil), 0.5, heatmap, 0.5, 0)
            plt.figure(figsize=(5, 5))
            plt.imshow(result)
            plt.title(f"Pred: {class_names[pred_label]}, True: {class_names[true_label]}")
            plt.savefig(os.path.join(result_dir, f"gradcam_error_{idx}.png"))
            plt.close()

        print("Error analysis completed. Check result directory for visualizations.")
    except Exception as e:
        print(f"錯誤分析過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_error_analysis()