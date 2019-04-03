# coding=utf-8
from sklearn.datasets import load_digits
# -------------
from sklearn.model_selection import train_test_split
# -------------
# load data standardize model
from sklearn.preprocessing import StandardScaler
# load SVM:LinearSVC which is based on Linear hypothesis
from sklearn.svm import LinearSVC
# -------------
from sklearn.metrics import classification_report
import torch
import cv2
import matplotlib.pyplot as plt
# -------------  store handwrite num datas on digits
digits = load_digits()
print('Total dataset shape', digits.data.shape)

# -------------  data prepare
# 75% training set,25% testing set
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

#查看具体的数据，以及显示图片
su=X_train[0]
su=su.reshape(8,-1)
plt.imshow(su,cmap='gray')
plt.show()
cv2.imwrite('su'+'.jpg', img=su)
print(su)
# -------------  training，转化为以0为均值，1为方差的数据，这段的作用是加速训练
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# initialize LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
# use trained model to predict testing dataset,and store the result on y_predict
y_predict = lsvc.predict(X_test)

# -------------  performance measure
print('The Accuracy is', lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
