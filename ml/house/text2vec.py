#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
"""
将文本变为one-hot编码，有多少个特征就有多少维度，有多少维度就添加多少特征，其中的1表示出现了某个文本特征
"""
root_path = '.'
train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))

qualitative = [f for f in train.columns if train.dtypes[f] == 'object'or train.dtypes[f] == 'str']
oridnals=['BsmtFinType1','MasVnrType','Foundation','HouseStyle','Functional','BsmtExposure','GarageFinish','PavedDrive','Street',
         'ExterQual', 'PavedDrive','ExterQua','ExterCond','KitchenQual','HeatingQC','BsmtQual','FireplaceQu','GarageQual','PoolQC']
qualitative=list(set(qualitative).difference(set(oridnals)))
def getdummies(res, ls):
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

    #for l in ls:
    cat, le, enc = encode(res[ls])
    cat.columns = [ls+str(x) for x in cat.columns]
    #outres.reset_index(drop=True, inplace=True)
    outres = pd.concat([outres, cat], axis = 1)
    decoder.append([le,enc])
    return (outres, decoder)
catpredlist=qualitative
res = getdummies(train[catpredlist],'Neighborhood')
df = res[0]
decoder = res[1]
floatAndordinal=list(set(train.columns.values).difference(set(qualitative)))
df=df.drop('A',axis=1)
print(df.shape)

#df = pd.concat([df,train[floatAndordinal]],axis=1)
pca = PCA(n_components=1)
df = pd.DataFrame(pca.fit_transform(df))
print(df)
#df.drop(['SalePrice'],axis=1,inplace=True)