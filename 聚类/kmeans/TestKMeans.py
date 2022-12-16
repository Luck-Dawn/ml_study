import pandas as pd
import matplotlib.pyplot as plt
from Kmeans import KMeans
data = pd.read_csv('../data/iris.csv')
# 所有花的类别
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# 根据花瓣的长宽维度来聚类
x_axis = 'petal_length'
y_axis = 'petal_width'

# 子图1：画出原始数据分布图 不带类别提示
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    # data[x_axis][data['class'] == iris_type] ，第一个中括号取某一列，第二个中括号取那些行，里面有个过滤条件：data['class'] == iris_type取本类的
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

# 子图2：画出原始数据分布图 带类别提示
plt.subplot(1, 2, 2)
plt.scatter(data[x_axis][:], data[y_axis][:])
plt.title('label unknown')
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)

# 指定好训练所需的参数
num_clusters = 3
max_iteritions = 50

k_means = KMeans(x_train, num_clusters)
centroids, closest_centroids_ids = k_means.train(max_iteritions)

# 对比结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

plt.subplot(1, 2, 2)
for centroid_id, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=centroid_id)

for centroid_id, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')
plt.legend()
plt.title('label kmeans')
plt.show()