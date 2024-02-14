import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import _plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

data = pd.read_csv("数据13.1.csv")
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# 按7:3划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 数据归一化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 模型实现(线性核函数)
model = SVC(kernel="linear", random_state=10)
model.fit(X_train_s, y_train)
print(model.score(X_test_s, y_test))

# 性能评价
np.set_printoptions(suppress=True) # 直接以数字显示，而非科学计数法
pred = model.predict(X_test_s)
cm = confusion_matrix(y_test, pred)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(cohen_kappa_score(y_test, pred))

# 使用seaborn绘制混淆矩阵的热度图
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",annot_kws={"size": 16})
plt.ylabel('True labels', fontsize=16)
plt.title('Confusion Matrix', fontsize=12)
plt.xticks(rotation=0, fontsize=12)  # 确保x轴标签水平显示

# 将x轴标签移到顶部
ax = plt.gca()  # 获取当前轴对象
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

plt.xlabel('Predicted labels', fontsize=12)  # 仅设置标签文本
plt.show()
