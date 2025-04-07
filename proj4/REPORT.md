
# 貓狗分類訓練實驗報告

本次實驗旨在透過調整訓練參數（epoch、batch size、learning rate）與更換不同損失函數，觀察其對模型訓練效果（Loss 與 Accuracy）的影響。

---

## 🧪 調整 Epoch 數量

| Epoch 數 | 圖片連結 |
|----------|-----------|
| 20       | ![](result/Training_accuracy_e20_bs32_lr0.01.png) |
| 40       | ![](result/Training_accuracy_e40_bs32_lr0.01.png) |
| 80       | ![](result/Training_accuracy_e80_bs32_lr0.01.png) |

---

## 🧪 調整 Batch Size

| Batch Size | 圖片連結 |
|------------|-----------|
| 8          | ![](result/Training_accuracy_e20_bs8_lr0.01.png) |
| 16         | ![](result/Training_accuracy_e20_bs16_lr0.01.png) |
| 32         | ![](result/Training_accuracy_e20_bs32_lr0.01.png) |

---

## 🧪 調整 Learning Rate

| Learning Rate | 圖片連結 |
|----------------|-----------|
| 0.1            | ![](result/Training_accuracy_e20_bs32_lr0.1.png) |
| 0.01           | ![](result/Training_accuracy_e20_bs32_lr0.01.png) |
| 0.001          | ![](result/Training_accuracy_e20_bs32_lr0.001.png) |

---

## 🧪 比較不同損失函數（Loss Function）

| Loss Function      | Loss 曲線 | Accuracy 曲線 |
|--------------------|-----------|----------------|
| CrossEntropyLoss   | ![](result/Loss_curve_loss_ce.png) | ![](result/Training_accuracy_loss_ce.png) |
| BCEWithLogitsLoss  | ![](result/Loss_curve_loss_bce.png) | ![](result/Training_accuracy_loss_bce.png) |
| FocalLoss          | ![](result/Loss_curve_loss_focal.png) | ![](result/Training_accuracy_loss_focal.png) |

---

## ✅ 小提醒

- 所有圖檔預設路徑為 `result/`，若你圖片放在不同資料夾，請自行調整路徑。
- 若欲轉為 PDF，可使用 Typora 或 VS Code + Markdown PDF 套件匯出。

