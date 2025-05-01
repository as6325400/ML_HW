import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import cv2
from PIL import Image
import glob
from timm import create_model

# 參數設定 (與 main_training.py 保持一致)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 256  # 與訓練時相同
num_classes = 12
num_folds = 3
img_size = 224  # 與訓練時相同
model_name = 'vit_base_patch16_224'  # 與訓練時相同
base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "plant-seedlings-classification", "test")
weight_dir = os.path.join(base_path, "weights5")
result_dir = os.path.join(base_path, "result5")
submission_path = os.path.join(result_dir, "predictions.csv")
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

# TTA 增強 (基於訓練時的 transform 設置)
tta_transforms = [
    A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=remove_background, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=remove_background, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=remove_background, p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=remove_background, p=1.0),
        A.Rotate(limit=10, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
]

# 測試數據集 (與訓練時的 PlantSeedlingDataset 一致)
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

# 預測函數 (支援 TTA，修正遍歷方式)
def predict_with_tta(model, test_loader, tta_transforms):
    model.eval()
    all_preds = np.zeros((len(test_loader.dataset), num_classes))
    with torch.no_grad():
        # 處理原始增強
        for i, (data, _) in enumerate(tqdm(test_loader, desc="Predicting with Original")):
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
        
        # 處理其他 TTA 增強
        for tta_idx, tta_transform in enumerate(tta_transforms[1:], 1):  # 跳過第一個（原始）
            tta_dataset = PlantSeedlingTestDataset(test_loader.dataset.image_paths, transform=tta_transform)
            tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            tta_preds = np.zeros((len(test_loader.dataset), num_classes))
            for i, (data, _) in enumerate(tqdm(tta_loader, desc=f"Predicting with TTA {tta_idx}")):
                try:
                    data = data.to(device)
                    preds = model(data)
                    start_idx = i * batch_size
                    end_idx = start_idx + data.size(0)
                    tta_preds[start_idx:end_idx] = preds.cpu().numpy()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("警告: CUDA 記憶體不足，跳過此批次")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            all_preds += tta_preds
        
        all_preds /= len(tta_transforms)  # 平均 TTA 結果
    return all_preds

# 主預測和集成函數
def predict_test_data():
    try:
        test_image_paths = load_test_data(test_data_path)
        if not test_image_paths:
            print("錯誤: 未能載入任何測試圖像，請檢查數據路徑")
            return

        test_dataset = PlantSeedlingTestDataset(test_image_paths, transform=tta_transforms[0])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 集成多折模型
        all_ensemble_preds = np.zeros((len(test_image_paths), num_classes))
        total_models = 0

        for fold in range(num_folds):
            weight_path = os.path.join(weight_dir, f'best_model_fold_{fold}.pth')
            if os.path.exists(weight_path):
                print(f"Loading weights from: {weight_path}")
                model = create_model(model_name, pretrained=False, num_classes=num_classes)
                # 使用 strict=False 忽略不匹配的鍵
                model.load_state_dict(torch.load(weight_path), strict=False)
                model = model.to(device)
                fold_preds = predict_with_tta(model, test_loader, tta_transforms)
                all_ensemble_preds += fold_preds
                total_models += 1
                print(f"Loaded and predicted with fold {fold}")
                # 清理記憶體
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if total_models > 0:
            all_ensemble_preds /= total_models  # 平均所有模型的預測
        else:
            raise ValueError("No model weights found. Please ensure models are trained and weights are saved.")

        # 生成最終預測
        final_preds = np.argmax(all_ensemble_preds, axis=1)
        predictions = [class_names[p] for p in final_preds]

        # 保存結果到 CSV
        df_dict = {
            'file': [os.path.basename(path) for path in test_image_paths],
            'species': predictions
        }
        df = pd.DataFrame(df_dict)
        df.to_csv(submission_path, index=False)
        print(f"Predictions saved to {submission_path}")
    except Exception as e:
        print(f"預測過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_test_data()