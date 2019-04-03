#coding=utf-8
from numpy import *
import time
from sklearn.linear_model import LogisticRegression

def createTrainDataSet():
    trainDataMat = [[1, 1, 4],
                    [1, 2, 3],
                    [1, -2, 3],
                    [1, -2, 2],
                    [1, 0, 1],
                    [1, 1, 2]]
    trainShares = [1, 1, 1, 0, 0,  0]
    return trainDataMat, trainShares

def createTestDataSet():
    testDataMat = [[1, 1, 1],
                   [1, 2, 0],
                   [1, 2, 4],
                   [1, 1, 3]]
    return testDataMat

def main():
    lr = LogisticRegression()
    trainDataSet, trainShares = createTrainDataSet()
    testDataSet = createTestDataSet()
    #trainDataSet, testDataSet = autoNorm(vstack((mat(trainDataSet), mat(testDataSet))))
    # 调用LogisticRegression中的fit函数/模块用来训练模型参数。
    lr.fit(trainDataSet, trainShares)
    # 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
    lr_y_predict = lr.predict(testDataSet)
    print(lr_y_predict)

if __name__ == "__main__":
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))