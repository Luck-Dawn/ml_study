from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

print("模块1：调用机器学习包来做线性回归")
# 读取数据集 Mnist数据是图像数据：(28,28,1)的灰度图
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

# 洗牌操作,随机选6w条数据作为训练数据。
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# print(y_train) ---> ['6' '9' '7' ... '5' '8' '6']
# bug1:这里数据集都是数字字符串,y_train == 5 这个条件判断，出来的结果都是false，然后训练的时候报错。说什么
# ValueError: The number of classes has to be greater than one; got 1 class。意思就是结果数据集就一个类别，怎么给你做训练？
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(max_iter=500, random_state=42)
sgd_clf.fit(X_train, y_train_5)
# print("索引在35000的预测数据集中的数据x：" + str([X[35000]]))
print("索引在35000的预测数据集中的数据y：" + y[35000] + " ,索引在35000的是否是5的预测结果：" + str(sgd_clf.predict([X[35000]])))

# print("\n模块2：调用机器学习包来做交叉验证")
# '''
# cv 将传入的训练数据集做几份
# scoring：用什么指标来评估，有召回率、准确率accuracy来评估
# '''
# score_list3 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# print("所有交叉验证得分为：" + str(score_list3))
#
# print("\n模块3：自己手动做交叉验证。将训练的数据集拆成3分。对每份都计算他的准确率")
# skfolds = StratifiedKFold(n_splits=3, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     # split方法返回切分数据的索引值
#     # 克隆一个新的模型句柄
#     clone_clf = clone(sgd_clf)
#
#     # 获取切分部分的训练集
#     x_train_folds = X_train[train_index]
#     y5_train_folds = y_train_5[train_index]
#
#     # 获取切分部分的测试集
#     x_test_folds = X_train[test_index]
#     y5_test_folds = y_train_5[test_index]
#
#     # 对每个切分数据进行训练
#     clone_clf.fit(x_train_folds, y5_train_folds)
#     # 获取预测结果
#     y5_predict = clone_clf.predict(x_test_folds)
#
#     # 将预测结果 和 实际结果进行对比。获取结果相同的个数
#     n_correct = sum(y5_predict == y5_test_folds)
#     print("本次交叉验证得分为：" + str(n_correct / len(y5_predict)))

print("\n模块4：混淆矩阵")
y_train_5_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# 返回结果 [[tn, fp],[ fn, tp]]
# true negatives: 53,272个数据被正确的分为非5类别
# false positives：1307张被错误的分为5类别
# false negatives：1077张错误的分为非5类别
# true positives： 4344张被正确的分为5类别
confusion_matrix = confusion_matrix(y_train_5, y_train_5_predict)
print("TN:" + str(confusion_matrix[0][0]))
print("FP:" + str(confusion_matrix[0][1]))
print("FN:" + str(confusion_matrix[1][0]))
print("TP:" + str(confusion_matrix[1][1]))

print("\n模块5：常用的一些模型评估指标")
# https://blog.csdn.net/Mr_Suda/article/details/122025282 就这里有介绍4中评价指标

precision_score = precision_score(y_train_5, y_train_5_predict)
accuracy_score = accuracy_score(y_train_5, y_train_5_predict)
recall_score = recall_score(y_train_5, y_train_5_predict)
f1_score = f1_score(y_train_5, y_train_5_predict)
print("精确率：" + str(precision_score))
print("召回率：" + str(recall_score))
print("准确率：" + str(accuracy_score))
print("F1值：" + str(f1_score))
