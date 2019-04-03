# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold

from DealMiss import FeatureEngineering as FE
from WriteExcel import Write
import matplotlib.pyplot as plt

def modelAll(model,name,*tests):
    (x_train, y_train, x_test, y_test, test)=tests
    model.fit(x_train, y_train)
    print(name)
    print('train score is:{} '.format(model.score(x_train, y_train)))
    print('test accury:{}'.format(model.score(x_test, y_test)))
    result = model.predict(test)
    return result
def drawAuc(test_y,y_score):
    FPR, TPR, _ = roc_curve(test_y, y_score)
    ROC_AUC = auc(FPR, TPR)
    print('ROC_AUC:', ROC_AUC)

    plt.figure(figsize=[11, 9])
    plt.plot(FPR, TPR, label='ROC curve(area = %0.2f)' % ROC_AUC, linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC for Titanic survivors', fontsize=18)
    plt.show()

def drawPR(test_y,y_score):
    precision, recall, _ = precision_recall_curve(test_y, y_score)
    PR_AUC = auc(recall, precision)
    print('PR_auc:',PR_AUC)
    plt.figure(figsize=[11, 9])
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
    plt.legend(loc="lower right")
    plt.show()
if __name__=='__main__':
    wri = Write()
    faeture=FE()
    tests=faeture.calls()
    (train_x, train_y,test_x, test_y,test)=tests

    model= LinearRegression()
    SalePrice=modelAll(model,'LinearR',*tests)

    model=LogisticRegression()
    SalePrice=modelAll(model,'LogisticR',*tests)

    # param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    model = SVC(cache_size=200, class_weight=None, coef0=0.0, C=1,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    SalePrice = modelAll(model, 'SVC', *tests)
    wri.write(SalePrice)


    max_depth = range(1, 30)
    max_feature = [21, 22, 23, 24, 25, 26, 28, 29, 30, 'auto']
    criterion = ["entropy", "gini"]
    param = {'max_depth': max_depth,
             'max_features': max_feature,
             'criterion': criterion}
    model = GridSearchCV(DecisionTreeClassifier(),
                                     param_grid=param,
                                     verbose=False,
                                     cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                     n_jobs=-1)
    model.fit(train_x, train_y)
    model=model.best_estimator_
    SalePrice = modelAll(model, 'DTC', *tests)

    model = GradientBoostingClassifier()
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    SalePrice = modelAll(model, 'GBC', *tests)

    y_score = model.predict(test_x)
    drawAuc(test_y,y_score)
    drawPR(test_y,y_score)