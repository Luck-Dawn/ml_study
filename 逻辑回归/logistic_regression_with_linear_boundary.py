import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

data = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

x_axis = 'petal_length'
y_axis = 'petal_width'

# 画出原始数据集分布
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type],
                data[y_axis][data['class'] == iris_type],
                label=iris_type
                )

plt.show()

num_examples = data.shape[0]
# 获取训练的测试数据X，提取2个特征维度来分析
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
# 获取训练的测试数据Y，整个结果类别这一列取出
y_train = data['class'].values.reshape((num_examples, 1))

max_iterations = 1000
polynomial_degree = 0
sinusoid_degree = 0

# 模型训练
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
thetas, cost_histories = logistic_regression.train(max_iterations)

# 每次二分类的损失函数变化图
labels = logistic_regression.unique_labels
plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], label=labels[2])
plt.show()

# 调用预测方法。并计算出准确度
y_train_predictions = logistic_regression.predict(x_train)
precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
print('precision:', precision)

# 绘制决策边界
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
samples = 150
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

# 3个类别的决策边界都画出来
Z_SETOSA = np.zeros((samples, samples))
Z_VERSICOLOR = np.zeros((samples, samples))
Z_VIRGINICA = np.zeros((samples, samples))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        # 每个值都拿去做一次预测
        prediction = logistic_regression.predict(data)[0][0]
        if prediction == 'SETOSA':
            Z_SETOSA[x_index][y_index] = 1
        elif prediction == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] = 1
        elif prediction == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] = 1

# 把原始数据分布图画上去，作为一个背景图，然后再这上面画出决策边界
for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(), 0],
        x_train[(y_train == iris_type).flatten(), 1],
        label=iris_type
    )

# 这个决策边界的图没有看懂。只知道数据 X、Y是自己根据原始数据2个特征值的最大最小构造的，
# 然后Z是对应每个X、Y位置上的预测值。然后画出来的一个等高线。每个类别都进行一次该操作。就绘制出来一个决策边界。反正不是很懂这个决策边界。跳过。跳过
plt.contour(X, Y, Z_SETOSA)
plt.contour(X, Y, Z_VERSICOLOR)
plt.contour(X, Y, Z_VIRGINICA)
plt.show()
