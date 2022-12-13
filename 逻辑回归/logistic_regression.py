import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=False)

        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 特征个数。2个数据本身特征+1个数据预处理加的1列求theta0
        num_features = self.data.shape[1]
        # 总的分类个数，
        num_unique_labels = np.unique(labels).shape[0]
        # 构造theta：3个类别，2个特征加上1列求theta就是3个特征
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        """
        逻辑回归模型训练方法
        :param max_iterations:最大迭代次数
        :return:返回不同类别下求解的theta 和 损失函数
        """
        cost_histories = []
        num_features = self.data.shape[1]
        # 逻辑回归本质上是二分类任务。因此如果有n个类别进行分类。就需要进行几次逻辑回归处理
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            # 训练集的的Y。不过全都是转换成 0/1这种数据，来做训练数据集的Y
            current_labels = (self.labels == unique_label).astype(float)
            (current_theta, cost_history) = LogisticRegression.gradient_descent(self.data, current_labels,
                                                                                current_initial_theta, max_iterations)

            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)

        return self.theta, cost_histories

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iterations):
        """
        流水线式的执行梯度下降方法
        :param data:
        :param labels:
        :param current_initial_theta:
        :param max_iterations:
        :return: 计算后的theta、cost集合
        """
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
            # 要优化的目标：
            lambda current_theta: LogisticRegression.cost_function(data, labels,
                                                                   current_theta.reshape(num_features, 1)),
            # 初始化的权重参数
            current_initial_theta,
            # 选择优化策略
            method='CG',
            # 梯度下降迭代计算公式
            # jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_initial_theta.reshape(num_features,1)),
            jac=lambda current_theta: LogisticRegression.gradient_step(data, labels,
                                                                       current_theta.reshape(num_features, 1)),
            # 记录结果
            callback=lambda current_theta: cost_history.append(
                LogisticRegression.cost_function(data, labels, current_theta.reshape((num_features, 1)))),
            # 迭代次数
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Can not minimize cost function' + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    @staticmethod
    def cost_function(data, labels, theta):
        """
        逻辑回归的损失函数,参照pdf文档中的公式，使用交叉熵来的生成的损失函数
        :param data: 输入的X
        :param labels: 输入的Y（0 / 1）
        :param theta: 经过梯度下降计算出的theta
        :return: 返回本次计算的损失值
        """
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        """
        矩阵运算，并将矩阵运算后的结果又参与sigmoid运算
        :param data:
        :param theta:
        :return: 经过sigmoid运算的结果
        """
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    @staticmethod
    def gradient_step(data, labels, theta):
        """
        逻辑回归对损失函数求导进行梯度下降法求解最小值，该方法只返回计算出来的梯度值
        :param data:
        :param labels:
        :param theta:
        :return: 梯度值
        """
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1 / num_examples) * np.dot(data.T, label_diff)

        return gradients.T.flatten()

    def predict(self, data):
        """
        预测方法
        :param data:
        :return:返回预测值
        """
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[
            0]

        # prob.shape:(150, 3)
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)

        # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1
        #  1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2
        #  2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        #  2 2]
        # 找出每行最大值的索引位置，有150个结果数，如果axis为0，则代表找出每列最大值索引位置有3个结果数
        max_prob_index = np.argmax(prob, axis=1)

        # 根据max_prob_index中的索引和标签字典表中的索引，找对应关系的标签名字
        result_predict_label = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            # index: 0 label: SETOSA
            # index: 1 label: VERSICOLOR
            # index: 2 label: VIRGINICA

            # 这一步的结果就是，把属于本标签的值赋给结果值，赋值3次，就将真个结果集填满了
            # max_prob_index == index，这个操作返回的是一个全为True、False的集合
            # [True  True  True  True  True  True  True  True  True  True  True  True
            #   ....
            #  False False False False False False False False False False False False]
            result_predict_label[max_prob_index == index] = label
        return result_predict_label.reshape((num_examples, 1))
