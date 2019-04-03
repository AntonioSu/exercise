# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

class Clear(object):
    def __init__(self):
        super(object, self).__init__()
        #导数据
        root_path = '.'
        self.train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
        self.test = pd.read_csv('%s/%s' % (root_path, 'test.csv'))
    def distPlot(self,train):
        # Check the train distribution
        sns.distplot(train, fit=norm)

        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(train)
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

        # Now plot the distribution
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        # Get also the QQ-plot
        fig = plt.figure()
        res = stats.probplot(train, plot=plt)
        plt.show()

    def Graph(self,train):
        # box plot overallqual/saleprice,Relationship with categorical features
        sns.boxplot(x='OverallQual', y="SalePrice", data=train)
        #绘制散点图
        sns.lmplot(x='OverallQual', y='SalePrice', data=train, fit_reg=False, scatter=True)
        #组合起来的散点图
        sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF'], y_vars=['SalePrice'], data=train,
                     dropna=True)
        plt.show()

    def statsShow(self,train):
        stats.probplot(train, plot=plt)
        plt.show()

    def lmPlot(self,train,x,y):
        #sns.lmplot(x='GrLivArea', y='SalePrice', data=train, fit_reg=False, scatter=True)
        sns.lmplot(x=x, y=y, data=train, fit_reg=False, scatter=True)
        plt.show()

    def Clear_train_Obeject(self,train):
        #查看train的数据分布
        colums = train.columns
        args = train['SalePrice'].describe()
        print(colums)
        print(args)

        # distPlot(train['SalePrice'])#展示其分布
        # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column，通过函数来改变原来数据的分布
        train['SalePrice'] = np.log(train['SalePrice'])

        # 通过图表删除GrLivArea当中数据离群点
        #lmPlot(train, 'GrLivArea', 'SalePrice')
        train = train[-((train.SalePrice < 12.5) & (train.GrLivArea > 4000))]
        return train

    def Clear_other(self,train,test):
        # 设置sns图文字大小
        sns.set(font_scale=1.5)
        fig = plt.figure()
        train['GrLivArea'] = np.log(train['GrLivArea'])
        test['GrLivArea'] = np.log(test['GrLivArea'])
        #statsShow(test['GrLivArea'])

        #改变'TotalBsmtSF'中为0的值为1
        train.loc[train['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = 1
        test.loc[test['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = 1
        #distPlot(train['TotalBsmtSF'])

        #对于BsmtQual这个特征，取值有 Ex，Gd，TA，Fa，Po. 从数据的说明中可以看出，这依次是优秀，好，次好，一般，差几个等级，这具有明显的可比较性
        train= train.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
        test= test.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})

        return train,test

    def Select_fearture(self,train):
        # 选特征
        train_corr = train.drop(labels='Id', axis=1).corr()
        k = 15  # number of variables for heatmap
        cols = train_corr.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(train[cols].values.T)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                    xticklabels=cols.values)
        plt.show()
        return cols

    def ClearAll(self):
        train=self.Clear_train_Obeject(self.train)
        train,test=self.Clear_other(train,self.test)
        #cols=self.Select_fearture(train)
        #根据热力图选取和房价相关的特征，但是在选择的过程中，有两个很相关的特征，可以不用选，只选用其一就好，比如GrLivArea和GrLivCars只用选其一就好
        x_train = train[['OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF','1stFlrSF', 'BsmtQual','FullBath', 'YearBuilt']]
        y_train = train[["SalePrice"]].values.ravel()

        test=test[['OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF','1stFlrSF', 'BsmtQual','FullBath', 'YearBuilt']]
        #如下按照train:test=2:1选择数据
        x_test = x_train[:500]
        x_train = x_train[500:]
        y_test = y_train[:500]
        y_train = y_train[500:]
        return x_train,y_train,x_test,y_test,test


# if __name__ == '__main__':
#     train,test=read_data()
#     main(train,test)



