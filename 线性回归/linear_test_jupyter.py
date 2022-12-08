import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

print("模块1：直接求解回归方程")
# 随机构造函数
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# 画出数据分布图
plt.plot(X, y, 'b.')
plt.xlabel('X_1')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
# plt.show()

# 数据拼接1列 全为1的。方便偏置项计算
# X:100个X的数据集, X_b：在X的基础上填加了一列全为1，X_new：输入2个数据集的X ，X_new_b：在X_new的基础上填加了1列
X_b = np.c_[np.ones((100, 1)), X]
# 根据回归算法，直接求解theta
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("直接求解得出的theta为：\n{}".format(str(theta_best)))
# 传入2个x的值 带入到计算好的回归方程函数 x1=0 x2=2
X_new = np.array([[0], [2]])
# np.ones((2,1)) 把2行1列的数据插入到矩阵中去，
X_new_b = np.c_[np.ones((2, 1)), X_new]
print("x的输入为：\n{}".format(str(X_new_b)))
# 计算出y的值
y_predict = X_new_b.dot(theta_best)
print("y的输出为：\n{}".format(str(y_predict)))

plt.plot(X_new, y_predict, 'r--')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
# plt.show()


# print("\n模块2：调用机器学习包实现线性回归")
# lin_reg = LinearRegression()
# # 训练参数
# lin_reg.fit(X, y)
# print("theta为：" + str(lin_reg.coef_))
# print("截距为:" + str(lin_reg.intercept_))

print("\n模块3：批量梯度下降")
plt.close()
eta = 0.1
n_iterations = 1000
m = X_b.shape[0]
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
print("通过批量梯度下降方法计算出来的theta为：\n{}".format(str(theta)))
# 顺便看看不同学习率对优化的结果
theta_path_bgd = []


def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, 'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new, y_predict, 'b-')
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('X_1')
    plt.axis([0, 2, 0, 15])
    plt.title('eta = {}'.format(eta))


theta = np.random.randn(2, 1)
plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)
print(len(theta_path_bgd))
# plt.show()


print("\n模块4：随机梯度下降")
plt.close()
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)
n_epochs = 50

t0 = 5
t1 = 50


def learning_schedule(t):
    return t0 / (t1 + t)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch < 10 and i < 10:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, 'r-')
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        #         学习率
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
# plt.show()

print("\n模块5：MiniBatch梯度下降")
plt.close()
theta_path_mgd = []
n_epochs = 50
minibatch = 16
theta = np.random.randn(2, 1)
t0, t1 = 200, 1000


def learning_schedule(t):
    return t0 / (t + t1)


np.random.seed(42)
t = 0
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch):
        t += 1
        xi = X_b_shuffled[i:i + minibatch]
        yi = y_shuffled[i:i + minibatch]
        gradients = 2 / minibatch * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
print("计算出来的theta为：\n{}".format(str(theta)))
print("3种策略的对比实验")
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(12, 6))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], 'r-s', linewidth=1, label='SGD')
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'g-+', linewidth=2, label='MINIGD')
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], 'b-o', linewidth=3, label='BGD')
plt.legend(loc='upper left')
plt.axis([3.5, 4.5, 2.0, 4.0])
# plt.show()

print("\n模块5：多项式回归")
plt.close()
# 自己定义一个曲线方程
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + np.random.randn(m, 1)

plt.plot(X, y, 'b.')
plt.xlabel('X_1')
plt.ylabel('y')
plt.axis([-3, 3, -5, 10])
# plt.show()

# PolynomialFeatures个人理解：就是将数据做丰富一点。比如原来我就1个维度。然后我做个特征变化。我可以把原来那个维度进行转换操作
# 原来数据加上2次方操作，形成一个新的维度,现在有（a，b）两个特征，使用degree=2的二次多项式则为（1，a, a^2, ab, b ,b^2)
# degree：度数，决定多项式的次数
# interaction_only： 默认为False，字面意思就是只能交叉相乘，不能有a^2这种.
# include_bias: 默认为True, 这个bias指的是多项式会自动包含1，设为False就没这个1了.
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print("原始数据X[0]：{},经过特征变化X_poly[0]: {}".format(str(X[0]), str(X_poly[0])))
print("对X[0] ** 2 操作结果为： {}".format(str(X[0] ** 2)))

# 使用特征变化后的数据进行线性回归
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print("theta为：" + str(lin_reg.coef_))
print("截距为:" + str(lin_reg.intercept_))

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, 'r--', label='prediction')
plt.axis([-3, 3, -5, 10])
plt.legend()
# plt.show()

# 测试：不同维度的特征变化。对结果的影响
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(12, 6))
for style, width, degree in (('g-', 1, 100), ('b--', 1, 2), ('r-+', 1, 1)):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_reg = Pipeline([('step1_poly_features', poly_features),
                               ('step2_StandardScaler', std),
                               ('step3_lin_reg', lin_reg)])
    polynomial_reg.fit(X, y)
    y_new_2 = polynomial_reg.predict(X_new)
    plt.plot(X_new, y_new_2, style, label='degree   ' + str(degree), linewidth=width)
plt.plot(X, y, 'b.')
plt.axis([-3, 3, -5, 10])
plt.legend()
# plt.show()

print("\n模块6：数据样本数量对结果的影响")
plt.close()
# 画出随着数据量的变化，对应的训练结果、测试结果的均方误差变化图
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict[:m]))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train_error')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val_error')
    plt.xlabel('Trainsing set size')
    plt.ylabel('RMSE')
    plt.legend()


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3.3])
# plt.show()

print("\n模块7：多项式回归的过拟合风险")
plt.close()

polynomial_reg = Pipeline([('poly_features', PolynomialFeatures(degree=25, include_bias=False)),
                           ('lin_reg', LinearRegression())])
plot_learning_curves(polynomial_reg, X, y)
plt.axis([0, 80, 0, 5])
plt.show()
