import numpy as np


class KMeans:
    def __init__(self, data, num_cluster):
        self.data = data
        self.num_cluster = num_cluster

    def train(self, max_iter):
        # 1.先随机选择K个中心点
        center_ids = KMeans.centroids_init(self.data, self.num_cluster)
        # 2.开始训练
        num_example = self.data.shape[0]
        closest_type_list = np.empty((num_example, 1))
        for _ in range(max_iter):
            # 3.得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_type_list = KMeans.data_tag_type(self.data, center_ids)
            # 4.进行中心点位置更新
            center_ids = KMeans.update_center_ids(self.data, closest_type_list, self.num_cluster)
        return center_ids, closest_type_list

    @staticmethod
    def centroids_init(data, num_cluster):
        # 初始化中心点 sfa
        num_examples = data.shape[0]
        rand_ids = np.random.permutation(num_examples)
        center_ids = data[rand_ids[:num_cluster], :]
        return center_ids

    @staticmethod
    #     给每个元素进行中心点定位。给每个点打上类别标签。判断依据：该点离那个中心点近就是那个类
    def data_tag_type(data, center_ids):
        num_examples = data.shape[0]
        num_center = center_ids.shape[0]

        # 用集合保存每个位置的类别，
        closest_type_list = np.zeros((num_examples, 1))
        # 外层遍历每个元素
        for example_index in range(num_examples):
            distance_list = np.zeros((num_center, 1))
            # 内存遍历每个类别
            for center_index in range(num_center):
                distance_diff = data[example_index, :] - center_ids[center_index, :]
                distance_list[center_index] = np.sum(distance_diff ** 2)

            # 将最小的类别值放入到集合中
            closest_type_list[example_index] = np.argmin(distance_list)

        return closest_type_list

    @staticmethod
    def update_center_ids(data, closest_centroids_ids, num_cluster):
        # 将每类下的所有数据求平均值进行更新
        num_features = data.shape[1]
        center_ids = np.zeros((num_cluster, num_features))
        for center_id in range(num_cluster):
            chose_ids = closest_centroids_ids == center_id
            center_ids[center_id] = np.mean(data[chose_ids.flatten(), :], axis=0)
        return center_ids
