import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# 设置画图字体
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
# X.shape (70000, 784)
# y.shape (70000,)
# 将前6w条数据作为训练集，剩余的做测试集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 洗牌操作
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 做2分类任务，结果弄成设置为是5 还是非5数据集
# y_train_5[:10] array([False, False, False, False, False, False, False, False, False,True])
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# y_train_5 = np.hstack((np.ones((len(y_train_5), 1)), y_train_5))
print(X_train.shape)
print(y_train_5.shape)
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(1122)
#@x
# sgd_clf.predict([X[35000]]) array([ True])
# dawnfds
# 交叉验证并获取每次交叉测试的得分
all_cross_score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(all_cross_score)
# print("交叉验证的得分：" + all_cross_score)
