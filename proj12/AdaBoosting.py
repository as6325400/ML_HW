import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 讀取 train 與 test 資料
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 分離特徵與標籤
X_train = train_df.drop(columns=["Id", "Cover_Type"])
y_train = train_df["Cover_Type"]

X_test = test_df.drop(columns=["Id", "Cover_Type"])
y_test = test_df["Cover_Type"]

# 模型參數
n_estimators = 1000

# 建立與訓練 AdaBoost 模型
base_learner = DecisionTreeClassifier(max_depth=7)
model = AdaBoostClassifier(
    estimator=base_learner,
    n_estimators=n_estimators,
    random_state=42
)

model.fit(X_train, y_train)

# 預測與評估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(train_acc, test_acc, n_estimators)
