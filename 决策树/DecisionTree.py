# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
from plot_util import createPlot


def create_dataset():
    """
    手动构造数据集、标签集合
    :return:
    """
    data_set = [[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return data_set, labels


def create_tree(dataset, labels, feat_labels):
    # 获取最后一列yes、no数据
    class_list = [example[-1] for example in dataset]
    # 停止条件1：某个叶子节点的熵为0，也就是分的很纯，全都是yes或者no
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 停止条件2：数据已经切分的只有1列标签值yes、no的时候，标签值那个占比多就返回那个
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    # 找出最好的特征索引编号，去作为决策节点，第一次找的话就是找根节点
    best_feat_index = choose_best_feature_index_to_split(dataset)
    # 通过索引去标签labels字典表中获取值对应的标签，并将结果放入到决策树中去
    best_feat_label = labels[best_feat_index]
    feat_labels.append(best_feat_label)
    my_tree = {best_feat_label: {}}

    # 删除已加入决策树的标签

    del labels[best_feat_index]

    # 从数据集中取出最好特征这一列的所有类别的属性值
    feat_value = [example[best_feat_index] for example in dataset]
    # 去重，遍历所有类别的属性值
    unique_feat_value = set(feat_value)
    for value in unique_feat_value:
        # 拆分数据集，只取属于同类别的数据集，数据集去掉 索引 = bestFeat这一列，已加入到决策树的维度就不参与后续的
        sub_dataset = split_dataset(dataset, best_feat_index, value)
        # 使用新的标签、新的数据集来递归构建决策树，
        my_tree[best_feat_label][value] = create_tree(sub_dataset, labels, feat_labels)
    return my_tree


def choose_best_feature_index_to_split(dataset):
    num_feature = len(dataset[0]) - 1
    base_entropy = cal_entropy_by_dataset(dataset)
    best_info_gain = 0
    best_feature_index = -1

    for feature_index in range(num_feature):
        feat_list = [example[feature_index] for example in dataset]
        unique_feat_list = set(feat_list)
        each_class_entropy = 0
        # 计算该特征下的不同类别的熵值，并求和
        for feat_value in unique_feat_list:
            # dataset:数据集， feature_index：特征位置索引值，feat_value：某一列特征下的某一类的值
            sub_dataset = split_dataset(dataset, feature_index, feat_value)
            prob = len(sub_dataset) / float(len(dataset))
            each_class_entropy += prob * cal_entropy_by_dataset(sub_dataset)

        # 获取最大的信息增益
        cur_info_gain = base_entropy - each_class_entropy
        if cur_info_gain > best_info_gain:
            best_info_gain = cur_info_gain
            best_feature_index = feature_index
    return best_feature_index


def split_dataset(dataset, feature_index, feat_value):
    result_dataset = []
    # 该方法有2个功能。第一，移除集合中的第axis列，并且只保留axis这一列的元素等于val这个类别的数据。有点绕，图例如下：
    # dataset=[[1,2,'yes'],[1,0,'no'],[0,1,'yes']], axis=0，val=1, 经过该方法后，返回的结果为：retDataSet [[2,'yes'],[0,'no']]
    for each_dataset in dataset:
        if each_dataset[feature_index] == feat_value:
            # 选取第feature_index列之前的所有元素
            elem = each_dataset[:feature_index]
            # 跳过第axis列，取后面剩余的其他元素，例如 featVec[feature_index] = [1, 0, 0, 1, 'no'],feature_index=1，
            # 经过这2步操作，就得到reducedFeatVec=[1, 0, 1, 'no']
            elem.extend(each_dataset[feature_index + 1:])
            result_dataset.append(elem)
    return result_dataset


def cal_entropy_by_dataset(dataset):
    num_examples = len(dataset)
    label_count_dic = {}

    # 计算每个类别标签的个数
    for each_dataset in dataset:
        cur_label = each_dataset[-1]
        if cur_label not in label_count_dic.keys():
            label_count_dic[cur_label] = 0
        label_count_dic[cur_label] += 1

    entropy = 0
    for key in label_count_dic:
        prop = float(label_count_dic[key]) / num_examples
        entropy -= prop * log(prop, 2)

    return entropy


def majority_cnt(class_list):
    class_count = {}
    for each_class in class_list:
        if each_class not in class_count.keys():
            class_count[each_class] = 0
        class_count[each_class] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    dataset, labels = create_dataset()
    featLabels = []
    myTree = create_tree(dataset, labels, featLabels)
    print("决策树结构为：\n", myTree)
    createPlot(myTree)
