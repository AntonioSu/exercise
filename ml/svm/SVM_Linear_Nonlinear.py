# Sklearn SVM支持向量机
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
    def loadDataSet(self):
        x = [[1, 8], [3, 20], [1, 15], [3, 35], [5, 35], [4, 40], [7, 80], [6, 20]]
        y = [1, 1, -1, -1, -1, -1, -1, 1]
        return x, y
    def plot_point(self,dataArr, labelArr, Support_vector_index, W, b,clf):
        for i in range(np.shape(dataArr)[0]):
            if labelArr[i] == 1:
                plt.scatter(dataArr[i][0], dataArr[i][1], c='r', s=20)
            else:
                plt.scatter(dataArr[i][0], dataArr[i][1], c='g', s=20)
        #画出支持向量的点
        for j in Support_vector_index:
            plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')
        #绘制分类曲线
        x = np.arange(0, 10, 0.01)
        y = (W[0][0] * x + b) / (-1 * W[0][1])
        plt.scatter(x, y, s=5, marker='h')
        plt.show()
    def Linea(self):
        # 读取数据,针对二维线性可分数据
        dataArr, labelArr = self.loadDataSet()
        # 定义SVM分类器
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        # fit训练数据
        clf.fit(dataArr, labelArr)
        # 给定数据 预测

        # 获取模型返回值
        n_Support_vector = clf.n_support_  # 支持向量个数
        Support_vector_index = clf.support_  # 支持向量索引
        W = clf.coef_  # 方向向量W
        b = clf.intercept_  # 截距项b
        rdm_arr = []
        for rand in range(15):
            y = np.random.randint(1, 80)
            x = np.random.randint(1, 10)
            rdm_arr.append([x, y])
        for i in rdm_arr:
            res = clf.predict(np.array(i).reshape(1, -1))
            if res > 0:
                plt.scatter(i[0], i[1], c='r', marker='.')
            else:
                plt.scatter(i[0], i[1], c='g', marker='.')

        # 绘制分类超平面
        self.plot_point(dataArr, labelArr, Support_vector_index, W, b, clf)

class NonLinear(nn.Module):
    def loadDataSet(self,fileName):
        dataMat = []
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat, labelMat

    def plot_point(self,dataArr, labelArr, Support_vector_index):
        for i in range(np.shape(dataArr)[0]):
            if labelArr[i] == 1:
                plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
            else:
                plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)

        for j in Support_vector_index:
            plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')
        plt.show()

    def Linea(self):
        # 读取数据,针对二维线性不可分数据
        dataArr, labelArr = self.loadDataSet('data.txt')

        # 交叉验证划分数据集，train：test = 0.8 : 0.2
        X_train, X_test, y_train, y_test = train_test_split(dataArr, labelArr, test_size=.2, random_state=0)

        """
        C：目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
        C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
        kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
        gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则gamma=1/n_features
        coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
        probability ：是否采用概率估计。默认为False。要采用的话必须先于调用fit,这个过程会增加用时。
        shrinking ：是否采用shrinking heuristic方法，默认为true
        tol ：停止训练的误差值大小，默认为1e-3
        cache_size ：核函数cache缓存大小，默认为200
        class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
        verbose ：允许冗余输出。跟多线程有关系。默认为False。
        max_iter ：最大迭代次数。-1为无限制。
        decision_function_shape ：是否返回模型中每一个类别的样本的ovr决策函数，或者ovo决策函数。 默认为None
        random_state ：数据洗牌时的种子值，int值
        主要调节的参数有：C、kernel、degree、gamma、coef0。
        """
        # 初始化模型参数
        clf = SVC(cache_size=200, class_weight=None, coef0=0.0, C=1.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        clf.fit(X_train, y_train)
        # 预测X_test
        predict_list = clf.predict(X_test)
        # 预测精度
        precition = clf.score(X_test, y_test)
        print('precition is : ', precition * 100, "%")
        # 获取模型返回值
        n_Support_vector = clf.n_support_  # 支持向量个数
        print("支持向量个数为： ", n_Support_vector)
        Support_vector_index = clf.support_  # 支持向量索引
        self.plot_point(dataArr, labelArr, Support_vector_index)

if __name__ == "__main__":
    linear=NonLinear()
    linear.Linea()
