# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import mean_squared_error
#from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from Clear_data import Clear
from WriteExcel import Write

def modelAll(model,name,*tests):
    (x_train, y_train, x_test, y_test, test)=tests
    model.fit(x_train, y_train)
    print(name)
    print('train score is:{} '.format(model.score(x_train, y_train)))
    print('test accury:{}'.format(model.score(x_test, y_test)))
    result = model.predict(test)
    result = np.exp(result)
    return result

if __name__=='__main__':
    clear = Clear()
    tests = clear.ClearAll()
    wri=Write()
    model = RandomForestRegressor(n_estimators = 500, n_jobs = -1,random_state =50,
                                    max_features = 6, min_samples_leaf = 2)
    SalePrice=modelAll(model,'RFR',*tests)
    wri.write(SalePrice)
    model=GradientBoostingRegressor(alpha=0.7, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=200, presort='auto', random_state=0,
                 subsample=1.0, verbose=0, warm_start=False)
    SalePrice=modelAll(model,'GBR',*tests)

    model = tree.DecisionTreeRegressor(max_depth=9)
    SalePrice=modelAll(model,'DTR',*tests)




