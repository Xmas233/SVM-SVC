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

data = pd.read_csv("数据15.1.csv")
X = data.iloc[:, 1:]  # 除第一列都为特征
Y = data.iloc[:, 0]  # 第一列为预测值

# 按7:3划分训练集和验证集，设置随机数种子，保证随机抽样结果可重复
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30, random_state=10)

# 数据归一化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 模型实现
model = SVR(kernel="linear")  # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
model.fit(X_train_s, Y_train)
print(model.score(X_test_s, Y_test))
pred = model.predict(X_test_s)

df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': pred,
    'Index': np.arange(len(Y_test))
})

# 使用seaborn绘制实际值
sns.lineplot(data=df, x='Index', y='Actual', label='Actual')

# 使用seaborn绘制预测值
sns.lineplot(data=df, x='Index', y='Predicted', label='Predicted')

plt.grid()
plt.show()