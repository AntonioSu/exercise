#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
"""
将文本变为one-hot编码，有多少个特征就有多少维度，有多少维度就添加多少特征，其中的1表示出现了某个文本特征
"""
class TextVec(object):
    def __init__(self):
        root_path = '.'
        self.train = pd.read_csv('%s/%s' % (root_path, 'data/train.csv'))
        self.test = pd.read_csv('%s/%s' % (root_path, 'data/test.csv'))
        self.ChangeColumn=['Sex','Embarked']#,'Cabin'

    def getdummies(self,train, ls):
        def encode(encode_df):
            encode_df = np.array(encode_df)
            enc = OneHotEncoder()
            le = LabelEncoder()
            le.fit(encode_df)
            res1 = le.transform(encode_df).reshape(-1, 1)
            enc.fit(res1)
            s=enc.transform(res1)
            return pd.DataFrame(s.toarray()), le, enc
        decoder = []
        outres = pd.DataFrame({'A' : []})

        for l in ls:
            cat, le, enc = encode(train[l])
            cat.columns = [l+str(x) for x in cat.columns]
            #outres.reset_index(drop=True, inplace=True)
            outres = pd.concat([outres, cat], axis = 1)
            decoder.append([le,enc])
        return (outres, decoder)

    def Trans(self):
        res = self.getdummies(self.train,self.ChangeColumn)
        df = res[0]
        decoder = res[1]
        floatAndordinal=list(set(self.train.columns.values).difference(set(self.ChangeColumn)))
        df=df.drop('A',axis=1)
        print(df.shape)

        df = pd.concat([df,self.train[floatAndordinal]],axis=1)
        #pca = PCA(n_components=1)
        #df = pd.DataFrame(pca.fit_transform(df))
        print(df)
        #df.drop(['SalePrice'],axis=1,inplace=True)

if __name__=='__main__':
    text=TextVec()
    text.Trans()
    print('done')