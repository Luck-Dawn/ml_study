import numpy as np
import os
import matplotlib
import warnings
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')
np.random.seed(42)

print("模块1：画出sigmoid函数图形")
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.title('Figure 4-21. Logistic function')
# plt.show()

print("\n模块2：使用sklearn工具包进行逻辑回归")
plt.close()

iris = datasets.load_iris()
# 取一个维度数据
X = iris['data'][:, 3:]
# 将数据的Y做成 0,1
y = (iris['target'] == 2).astype(np.int)

log_res = LogisticRegression()
log_res.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# predict_proba 返回预测结果概率值。y_proba1.shape：(1000, 2)  predict返回结果值。y_proba：(1000,)
y_proba = log_res.predict_proba(X_new)
print("预测结果概率集合：\n", y_proba)
# y_proba1 = log_res.predict(X_new)
# print(y_proba1.shape)


print("\n模块3：随着输入特征数值的变化，结果概率值也会随之变化")
plt.close()

plt.figure(figsize=(12, 5))
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]
plt.plot([decision_boundary, decision_boundary], [-1, 2], 'k:', linewidth=2)
plt.plot(X_new, y_proba[:, 1], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Not Iris-Virginica')
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.text(decision_boundary + 0.02, 0.15, 'Decision Boundary', fontsize=16, color='k', ha='center')
plt.xlabel('Peta width(cm)', fontsize=16)
plt.ylabel('y_proba', fontsize=16)
plt.axis([0, 3, -0.02, 1.02])
plt.legend(loc='center left', fontsize=16)
plt.show()

print("\n模块4：决策边界的绘制")
# 决策边界的绘制：
# step1：构建坐标数据，合理的范围当中，根据实际训练时输入数据来决定
# step2：整合坐标点，得到所有测试输入数据坐标点
# step3：预测，得到所有点的概率值
# step4：绘制等高线，完成决策边界
plt.close()

X = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.int)

# step1：构建坐标数据，合理的范围当中，根据实际训练时输入数据来决定
x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
x0, x1 = np.meshgrid(np.linspace(x0_min, x0_max, 500).reshape(-1, 1), np.linspace(x1_min, x1_max, 200).reshape(-1, 1))
# 测试数据，用来做预测结果，step2：整合坐标点，得到所有测试输入数据坐标点
X_new = np.c_[x0.ravel(), x1.ravel()]

# 模型训练
log_res = LogisticRegression(C=10000)
log_res.fit(X, y)

# 获取预测概率值 step3：预测，得到所有点的概率值
y_proba = log_res.predict_proba(X_new)

plt.figure(figsize=(10, 4))
# 背景原始数据图
plt.plot(X[y == 0, 0], X[y == 0, 1], 'bs')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'g^')

# 画出决策线 step4：绘制等高线，完成决策边界
zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.axis([2.9, 7, 0.8, 2.7])
plt.text(3.5, 1.5, 'NOT Vir', fontsize=16, color='b')
plt.text(6.5, 2.3, 'Vir', fontsize=16, color='g')
plt.show()

print("\n模块5：多分类 softmax")
plt.close()
X = iris['data'][:, (2, 3)]
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
softmax_reg.fit(X, y)

print("数据点[5,2] 的预测结果为：", softmax_reg.predict([[5, 2]]))
print("数据点[5,2] 的多分类预测各个类别的概率值为：", softmax_reg.predict_proba([[5, 2]]))

x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
