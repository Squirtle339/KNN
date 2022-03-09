"""KNN"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 计算距离

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def loadData(self, path):
        data = pd.read_csv(path, header=None)
        # 特征类别及label
        # data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
        # 取前四列特征
        X = data.iloc[:, 0:4].values
        # 取最后一列label
        y = data.iloc[:, 4].values
        # 三种花分别由0 1 2表示
        y[y == 'Iris-setosa'] = 0
        y[y == 'Iris-versicolor'] = 1
        y[y == 'Iris-virginica'] = 2
        # 从数据集的对应位置取出三种花的对应数据
        self.X_setosa, self.y_setosa = X[0:50], y[0:50]
        self.X_versicolor, self.y_versicolor = X[50:100], y[50:100]
        self.X_virginica, self.y_virginica = X[100:150], y[100:150]
        # 训练集，占3/5
        self.X_train = np.vstack([self.X_setosa[:30, :], self.X_versicolor[:30, :], self.X_virginica[:30, :]])
        self.y_train = np.hstack([self.y_setosa[:30], self.y_versicolor[:30], self.y_virginica[:30]])
        # 测试集，占2/5
        self.X_test = np.vstack([self.X_setosa[30:50, :], self.X_versicolor[30:50, :], self.X_virginica[30:50, :]])
        self.y_test = np.hstack([self.y_setosa[30:50], self.y_versicolor[30:50], self.y_virginica[30:50]])

    def predict(self, X, k, method='M'):
        num_test = X.shape[0]
        if method == 'E':
            # 计算欧氏距离
            # (X - X_train)^2 = -2X*X_train + X_train^2+X^2
            dist = -2 * np.dot(X, self.X_train.T) + np.sum(np.square(X), axis=1, keepdims=True) + np.sum(
                np.square(self.X_train), axis=1)
            distance = np.square(dist)
        else:
            # 计算曼哈顿距离
            distance = []
            for i in range(num_test):
                distance.append(np.sum(np.abs(X[i, :] - self.X_train), axis=1))
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 按距离排序并选择最近的k个点的索引
            dist_k_min = np.argsort(distance[i])[:k]
            # 取出最近的k个点的label
            y_kclose = self.y_train[dist_k_min]
            # 找出k个标签中从属类别最多的作为预测类别
            y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))
        return y_pred


if __name__ == "__main__":
    path = "Iris.data"
    knn = KNearestNeighbor()
    knn.loadData(path)
    accuracy_E = []
    accuracy_M = []
    for k in range(1, 15):
        y_pred = knn.predict(X=knn.X_test, k=k, method='E')
        accuracy_E.append(np.mean(y_pred == knn.y_test))
        y_pred = knn.predict(X=knn.X_test, k=k)
        accuracy_M.append(np.mean(y_pred == knn.y_test))
    # 不同k值准确率的折线图
    plt.title('The accuracy of Euler distance and Manhattan distance in different K')  # 折线图标题
    plt.plot(range(1, 15), accuracy_E,range(1, 15), accuracy_M)
    plt.legend(['Euler distance','Manhattan distance'])
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()


