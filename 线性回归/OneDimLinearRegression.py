import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray

from linear_regression import LinearRegression

data = pd.read_csv('data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train: ndarray = train_data[[input_param_name]].values
y_train: ndarray = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

# 画出梯度下降过程中，迭代次数和损失函数的关系
plt.plot(range(num_iterations), cost_history)
plt.xlabel('iter_num')
plt.ylabel('cost')
plt.title('GD')
plt.show()

# 根据上面的计算过程，得出theta的值，所以线性关系函数得出。根据该行数。自己来造数据。然后该数据经过得出的线性关系函数。得到预测值
predictions_num = 100
x_predict = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
y_predict = linear_regression.predict(x_predict)

# 画出 测试、训练、
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predict, y_predict, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
