# K近邻算法

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris      # 导入鸢尾花数据
from sklearn.model_selection import train_test_split    # 导入切分数据集的方法
from sklearn.metrics import accuracy_score   # 导入评价模型标准--准确率


# 1.数据加载和预处理

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target
df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1]})
print(df.head(10))
print(df.describe())     # 数据报告


x = iris.data
y = iris.target.reshape(-1, 1)
print('原始数据的维度：')
print(x.shape, y.shape)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)
print('划分后训练数据的维度：')
print(x_train.shape, x_test.shape)
print('划分后测试数据的维度：')
print(y_train.shape, y_test.shape)

a = np.array([[3,2,4,2],
             [2,1,4,23],
             [12,3,2,3],
             [2,3,15,23],
             [1,3,2,3],
             [13,3,2,2],
             [213,16,3,63],
             [23,62,23,23],
             [23,16,23,43]])
b = np.array([[1,1,1,1]])
np.sum(np.abs(a - b), axis=1)
dist = np.sqrt(np.sum((a-b) ** 2, axis=1))


# 2. 核心算法实现
# 距离函数定义
def L1_distance(a, b):
    return np.sum(np.abs(a-b), axis=1)


def L2_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))


# 分类器实现
class KNN(object):
    def __init__(self, n_neighbors=1, dist_func=L1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 训练模型
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # 模型预测方法
    def predict(self, x):
        # 初始化预测分类数组
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)

        # 遍历输入的x数据点，取出每一数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有寻来你数据计算距离
            distances = self.dist_func(self.x_train, x_test)

            # 得到的距离按照由近到远排序，取出索引值
            nn_index = np.argsort(distances)

            # 选取最近的K个点，保存他们对应的分类类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel()

            # 统计类别中出现频率最高的那个，赋值给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))

        return y_pred

    nn_index = np.argsort(dist)
    nn_y = y_train[nn_index[:9]].ravel()


# 3.测试
# 定义一个KNN实例
knn = KNN(n_neighbors=3)
# 训练模型
knn.fit(x_train, y_train)
# 传入测试数据，做预测
y_pred = knn.predict(x_test)

# 求出预测准确率
accuracy = accuracy_score(y_test, y_pred)
print('预测准确率是：', accuracy)


# 定义一个knn实例
knn = KNN()
# 训练模型
knn.fit(x_train, y_train)

# 保存结果list
result_list = []

# 针对不同的参数选取，做预测
for p in [1, 2]:
    knn.dist_func = L1_distance if p == 1 else L2_distance

    # 考虑不同的k取值，步长为2
    for k in range(1, 10, 2):
        knn.n_neighbors = k
        # 传入测试数据，做预测
        y_pred = knn.predict(x_test)
        # 求出预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])
df = pd.DataFrame(result_list, columns=['k', '距离函数', '预测准确率'])
print(df)