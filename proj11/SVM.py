import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 讀取資料集
df = pd.read_csv("train.csv")

# 移除 Id 與 Cover_Type 欄位（Cover_Type 不參與訓練）
X = df.drop(columns=["Id", "Cover_Type"]).values

# ---------- 資料標準化處理 ----------
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# ---------- 訓練已標準化的 One-Class SVM ----------
model = OneClassSVM(kernel='rbf', gamma=0.05, nu=0.05)
# 訓練
model.fit(X)
# 評估模型的性能
pred = model.predict(X)
# 計算資料異常點
anomaly_count = np.sum(pred == -1)

# ---------- 輸出結果 ----------
print("【異常偵測結果】")
print(f"資料的異常點數量：{anomaly_count}")
