from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier

# 读取数据集
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

print(X_train)
print(y_train_5)

sgd_clf = SGDClassifier(max_iter=500, random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([X[35000]]))
