import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# https://www.kaggle.com/competitions/aia-cm111-ev111-dl-kaggle/overview

class_name = ['Krummholz', 'Aspen', 'Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine',
              'Cottonwood/Willow', 'Douglas-fir']
feature = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
           "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
           "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
           "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1",
           "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8",
           "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15",
           "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22",
           "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
           "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36",
           "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
# 讀train.csv
df_train = pd.read_csv("train.csv", index_col=0)
x_train = df_train[feature]  # feature
y_train = df_train['Cover_Type']  # class

# 讀test.csv
df_test = pd.read_csv("test.csv", index_col=0)
x_test = df_test[feature]  # feature
y_test = df_test['Cover_Type']  # class

# 初始化並訓練決策樹分類模型

####################
# 題目: 更改 tree 的深度
model = DecisionTreeClassifier(random_state=42, max_depth=7)
####################

# 訓練
model.fit(x_train, y_train)

# 評估模型的性能
y_pred = model.predict(x_test)

# 準確率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 顯示一些預測結果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(10))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(constrained_layout=True)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_name,
            yticklabels=class_name)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('cofusion_matrix.jpg')

feature_importances = model.feature_importances_

# 將特徵重要性與特徵名結合，並排序
feature_importance_df = pd.DataFrame({
    'Feature': feature,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 取重要性大於0.0000001的feature
important_features = feature_importance_df[feature_importance_df['Importance'] > 0.0000001]
print(important_features.to_string(index=False))
