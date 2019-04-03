from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
list=['green','red','zlue','red','zlue','zlue','zlue','green']
enc = OneHotEncoder()
le = LabelEncoder()
s=le.fit(list) #按照字典顺序排序，定标签

res1 = le.transform(list).reshape(-1, 1)

enc.fit(res1)
su=enc.transform(res1)#记录矩阵中为1的位置
s=pd.DataFrame(enc.transform(res1).toarray())#横向是特征的维度，纵向是样本的个数
print(s)

enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # 注意：第1、2、3列分别有2、3、4个可能的取值
s=enc.transform([[0, 1, 3]]).toarray() #要对[0,1,3]进行编码
print(s)# [1,0]对应数值0，[0,1,0]对应数值1，[0,0,0,1]对应数值3