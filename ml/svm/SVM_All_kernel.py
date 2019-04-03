import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x = []  # 存储样本数据
y = []  # 存储类标号
for i in range(0, 10):  # 构造10个点作为训练数据
    if (i <= 3):  # if(i<=3 or i>=8):
        x.append([i, i])
        y.append(0)
    else:
        x.append([i, i])
        y.append(1)

train_x = np.array(x)  # 转换为数组
train_y = np.array(y)

'''
创建svm分类器的格式：
svm.SVC(kernel=某个核函数).fit(训练样本，类标签)
'''
# linear
linear_svc = svm.SVC(kernel="linear").fit(train_x, train_y)

# poly 要定义维度，degree决定了多项式的最高次幂.关于SVC参数的意义请参见文章后头的内容。
poly_svc = svm.SVC(kernel="poly", degree=4).fit(train_x, train_y)

# 径向基核函数(这时SVC默认的核函数)
rbf_svc = svm.SVC().fit(train_x, y)

# Sigmoid
sigmoid_svc = svm.SVC(kernel="sigmoid").fit(train_x, train_y)

# 下面就可以进行预测了
x1, x2 = np.meshgrid(np.arange(train_x[:, 0].min(), train_x[:, 0].max(), 1),
                      np.arange(train_x[:, 1].min(), train_x[:, 1].max(), 1))

# 先生成各个点。定义最小值和最大值后，定义隔多少值建立一个点。
# np.arange(train_x[:,1].min(),train_x[:,1].max(),0.01))返回的是900个元素的数组
# meshgrid函数用来产生矩阵。上面的语句也是就是numpy.meshgrid(numpy.arange(0,9,0.01),numpy.arange(0,9,0.01))

'''
x1是矩阵 
      [[ 0.  ,  0.01,  0.02, ...,  8.97,  8.98,  8.99],
       [ 0.  ,  0.01,  0.02, ...,  8.97,  8.98,  8.99],
       [ 0.  ,  0.01,  0.02, ...,  8.97,  8.98,  8.99],
       ..., 
       [ 0.  ,  0.01,  0.02, ...,  8.97,  8.98,  8.99],
       [ 0.  ,  0.01,  0.02, ...,  8.97,  8.98,  8.99],
       [ 0.  ,  0.01,  0.02, ...,  8.97,  8.98,  8.99]]

x2是矩阵
      [[ 0.  ,  0.  ,  0.  , ...,  0.  ,  0.  ,  0.  ],
       [ 0.01,  0.01,  0.01, ...,  0.01,  0.01,  0.01],
       [ 0.02,  0.02,  0.02, ...,  0.02,  0.02,  0.02],
       ..., 
       [ 8.97,  8.97,  8.97, ...,  8.97,  8.97,  8.97],
       [ 8.98,  8.98,  8.98, ...,  8.98,  8.98,  8.98],
       [ 8.99,  8.99,  8.99, ...,  8.99,  8.99,  8.99]]

'''
splocation = 1
for i in [linear_svc, poly_svc, rbf_svc, sigmoid_svc]:  # 遍历各个模型以便绘图，以看哪个核函数的准确率更高
    rst = i.predict(np.c_[
                        x1.ravel(), x2.ravel()])  # 横坐标和纵坐标的组合。x1.ravel()和x2.ravel()都是长度为810000的数组。c_[]用来将前后两个数组串联成一个810000行、2列的矩阵。
    su,wen=x1.ravel(), x2.ravel()
    yuan=np.c_[
        x1.ravel(), x2.ravel()]
    # 因为上面用到了四种分类模型，那么一个2×2的图就能够显示完全了。
    plt.subplot(2, 2, splocation)  # 第一个参数代表的是横向要划分的子图个数，第二个参数代表的是纵向要划分的子图的个数，第三个参数表示当前的定位
    plt.contourf(x1, x2, rst.reshape(x1.shape))  # contourf用来填充颜色。（当前横坐标，当前纵坐标，预测的分类结果（转为x1的规模维数））

    # 训练数据的点也绘制出来
    for j in range(0, len(y)):
        t = train_x[j:j + 1, 0]
        x = train_x[j:j + 1]
        if (int(y[j]) == 0):
            plt.plot(train_x[j:j + 1, 0], train_x[j:j + 1], "yo")  # y代表黄色，o代表散点图
        else:
            plt.plot(train_x[j:j + 1, 0], train_x[j:j + 1], "ko")  # 类别为1填充为黑色。k代表黑色
    splocation += 1
plt.show()