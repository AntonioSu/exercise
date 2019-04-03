#coding=utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
将文本变为one-hot编码，有多少个特征就有多少维度，有多少维度就添加多少特征，其中的1表示出现了某个文本特征
"""
class DealMiss(object):
    def __init__(self):
        root_path = '.'
        self.train = pd.read_csv('%s/%s' % (root_path, 'data/train.csv'))
        self.test = pd.read_csv('%s/%s' % (root_path, 'data/test.csv'))
        self.ChangeColumn=['Sex','Embarked']#,'Cabin'
    def showMiss(self,train):
        total = train.isnull().sum().sort_values(ascending=False)
        percent = round(train.isnull().sum().sort_values(ascending=False) / len(train) * 100, 2)
        calc=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(calc)

    def dealEmbarked(self, train,test,elment1,elment2):
        print(train[train[elment1].isnull()])
        percent = pd.DataFrame(round(train[elment1].value_counts(dropna=False, normalize=True) * 100, 2))
        total = pd.DataFrame(train[elment1].value_counts(dropna=False))
        total.columns = ["Total"]
        percent.columns = ['Percent']
        cal=pd.concat([total, percent], axis=1)
        print(cal)

        # fig, ax = plt.subplots(figsize=(10, 6), ncols=2)
        # ax1 = sns.boxplot(x=elment1, y=elment2, hue="Pclass", data=train, ax=ax[0])
        # ax2 = sns.boxplot(x=elment1, y=elment2, hue="Pclass", data=test, ax=ax[1])
        # ax1.set_title("Training Set", fontsize=18)
        # ax2.set_title('Test Set', fontsize=18)
        #fig.show()#
        #plt.show()#线程阻止
        train[elment1].fillna("C", inplace=True)
        return train,test

    """
    在原始特征中，cabin这个特征缺失的数量比较多，故而先将这些缺失的新定义一类(N)，
    而后再通过与目标值最相关的特征（fare），检测fare在cabin的分布，而后再更改(N),
    将其更改为按照fare的分布的效果
    """
    def dealCabin(self, train, test, elment1):
        print("Train Cabin missing: " + str(train[elment1].isnull().sum() / len(train[elment1])))
        print("Test Cabin missing: " + str(test[elment1].isnull().sum() / len(test[elment1])))

        ## Concat train and test into a variable "all_data"
        survivers = train.Survived
        train.drop(["Survived"], axis=1, inplace=True)
        all_data = pd.concat([train, test], ignore_index=False)

        ## Assign all the null values to N
        all_data[elment1].fillna("N", inplace=True)
        all_data[elment1] = [i[0] for i in all_data[elment1]]
        with_N = all_data[all_data[elment1] == "N"]
        without_N = all_data[all_data[elment1] != "N"]

        #通过fare来把那些cabin中是N的值改变成其他的，因为fare和结果的相关性很大，故而用这个特征很可靠
        all_data.groupby("Cabin")['Fare'].mean().sort_values()
        def cabin_estimator(i):
            a = 0
            if i < 16:
                a = "G"
            elif i >= 16 and i < 27:
                a = "F"
            elif i >= 27 and i < 38:
                a = "T"
            elif i >= 38 and i < 47:
                a = "A"
            elif i >= 47 and i < 53:
                a = "E"
            elif i >= 53 and i < 54:
                a = "D"
            elif i >= 54 and i < 116:
                a = 'C'
            else:
                a = "B"
            return a
        ##applying cabin estimator function.
        with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))
        ## getting back train.
        all_data = pd.concat([with_N, without_N], axis=0)

        ## PassengerId helps us separate train and test.
        all_data.sort_values(by='PassengerId', inplace=True)
        ## Separating train and test from all_data.
        train = all_data[:891]
        test = all_data[891:]
        # adding saved target variable with train.
        train['Survived'] = survivers

        return train,test

    #因为这是一个很关键的特征，所以补充为缺失数据其他比较重要的特征的平均值
    def dealFare(self,test):
        print(test[test.Fare.isnull()])
        missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
        ## replace the test.fare null values with test.fare mean
        test.Fare.fillna(missing_value, inplace=True)
        return test

    def dealSex(self,train,test):
        train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
        test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)
        return train,test

    def Correlation(self,train):
        corr=pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending=False))
        #corr=train.corr().sort(ascending=False)
        return corr

    def calls(self):
        #self.showMiss(self.train)
        """
                      Total  Percent
        Cabin          687    77.10
        Age            177    19.87
        Embarked         2     0.22
        Fare             0     0.00
        Ticket           0     0.00
        Parch            0     0.00
        SibSp            0     0.00
        Sex              0     0.00
        Name             0     0.00
        Pclass           0     0.00
        Survived         0     0.00
        PassengerId      0     0.00
        """
        train,test=self.dealEmbarked(self.train,self.test,'Embarked','Fare')
        train, test =self.dealCabin(train,test,'Cabin')
        test=self.dealFare(test)
        train, test=self.dealSex(train, test)
        corr=self.Correlation(train)

        print('done')
        return train,test

class FeatureEngineering(object):
    def __init__(self):
        super()
        miss=DealMiss()
        self.train, self.test=miss.calls()
        self.sc = StandardScaler()

    def Name(self,train,test):
        # 增加姓名的长度
        train['name_length'] = [len(i) for i in train.Name]
        test['name_length'] = [len(i) for i in test.Name]

        def name_length_group(size):
            a = ''
            if (size <= 20):
                a = 'short'
            elif (size <= 35):
                a = 'medium'
            elif (size <= 45):
                a = 'good'
            else:
                a = 'long'
            return a
        #再增加一列姓名长度的区间
        train['nLength_group'] = train['name_length'].map(name_length_group)
        test['nLength_group'] = test['name_length'].map(name_length_group)
        return train,test

    def Title(self,train,test):
        ## get the title from the name
        train["title"] = [i.split('.')[0] for i in train.Name]
        train["title"] = [i.split(',')[1] for i in train.title]
        test["title"] = [i.split('.')[0] for i in test.Name]
        test["title"] = [i.split(',')[1] for i in test.title]
        # rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
        # train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
        ## train Data
        train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
        train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
        train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
        train["title"] = [i.replace('Dr', 'rare') for i in train.title]
        train["title"] = [i.replace('Col', 'rare') for i in train.title]
        train["title"] = [i.replace('Major', 'rare') for i in train.title]
        train["title"] = [i.replace('Don', 'rare') for i in train.title]
        train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
        train["title"] = [i.replace('Sir', 'rare') for i in train.title]
        train["title"] = [i.replace('Lady', 'rare') for i in train.title]
        train["title"] = [i.replace('Capt', 'rare') for i in train.title]
        train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
        train["title"] = [i.replace('Rev', 'rare') for i in train.title]

        # rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
        # train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
        ## test data
        test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
        test['title'] = [i.replace('Dr', 'rare') for i in test.title]
        test['title'] = [i.replace('Col', 'rare') for i in test.title]
        test['title'] = [i.replace('Dona', 'rare') for i in test.title]
        test['title'] = [i.replace('Rev', 'rare') for i in test.title]
        return train,test

    def FamilySize(self,train,test):
        ## Family_size seems like a good feature to create
        train['family_size'] = train.SibSp + train.Parch + 1
        test['family_size'] = test.SibSp + test.Parch + 1

        def family_group(size):
            a = ''
            if (size <= 1):
                a = 'loner'
            elif (size <= 4):
                a = 'small'
            else:
                a = 'large'
            return a

        train['family_group'] = train['family_size'].map(family_group)
        test['family_group'] = test['family_size'].map(family_group)

        train['is_alone'] = [1 if i < 2 else 0 for i in train.family_size]
        test['is_alone'] = [1 if i < 2 else 0 for i in test.family_size]
        return train,test

    def Fare(self,train,test):
        def fare_group(fare):
            a = ''
            if fare <= 4:
                a = 'Very_low'
            elif fare <= 10:
                a = 'low'
            elif fare <= 20:
                a = 'mid'
            elif fare <= 45:
                a = 'high'
            else:
                a = "very_high"
            return a
        #calculated_fare
        train['fare_group'] = train['Fare'].map(fare_group)
        test['fare_group'] = test['Fare'].map(fare_group)
        return train,test

    def Age(self,train,test):
        print("number age missing: " + str(train.Age.isnull().sum()))
        print("number total train: " + str(len(train.Age)))
        print("age missing rate: " + str(train.Age.isnull().sum() / len(train)))
        train = pd.concat([train[["Survived", "Age", "Sex", "SibSp", "Parch"]], train.loc[:, "is_alone":]], axis=1)
        test = pd.concat([test[["Age", "Sex"]], test.loc[:, "SibSp":]], axis=1)

        ## create bins for age
        def age_group_fun(age):
            a = ''
            if age <= 1:
                a = 'infant'
            elif age <= 4:
                a = 'toddler'
            elif age <= 13:
                a = 'child'
            elif age <= 18:
                a = 'teenager'
            elif age <= 35:
                a = 'Young_Adult'
            elif age <= 45:
                a = 'adult'
            elif age <= 55:
                a = 'middle_aged'
            elif age <= 65:
                a = 'senior_citizen'
            else:
                a = 'old'
            return a

        ## Applying "age_group_fun" function to the "Age" column.
        train['age_group'] = train['Age'].map(age_group_fun)
        test['age_group'] = test['Age'].map(age_group_fun)

        ## Creating dummies for "age_group" feature.
        train = pd.get_dummies(train, columns=['age_group'], drop_first=True)
        test = pd.get_dummies(test, columns=['age_group'], drop_first=True)

        train.drop('Age', axis=1, inplace=True)
        test.drop('Age', axis=1, inplace=True)
        return train,test

    def calls(self):
        train, test = self.Name(self.train, self.test)
        train, test = self.Title(train, test)
        train, test = self.FamilySize(train, test)
        train, test = self.Fare(train, test)

        train.drop(['Ticket'], axis=1, inplace=True)
        test.drop(['Ticket'], axis=1, inplace=True)
        train.drop(['PassengerId'], axis=1, inplace=True)
        test.drop(['PassengerId'], axis=1, inplace=True)

        train = pd.get_dummies(train, columns=['title', "Pclass", 'Cabin', 'Embarked', 'nLength_group', 'family_group',
                                               'fare_group'], drop_first=False)
        test = pd.get_dummies(test, columns=['title', "Pclass", 'Cabin', 'Embarked', 'nLength_group', 'family_group',
                                             'fare_group'], drop_first=False)
        train.drop(['family_size', 'Name', 'Fare', 'name_length'], axis=1, inplace=True)
        test.drop(['Name', 'family_size', "Fare", 'name_length'], axis=1, inplace=True)

        train, test = self.Age(train, test)
        y = train["Survived"]
        X = train.drop(['Survived'], axis=1)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.33, random_state=0)

        ## transforming "train_x"
        train_x = self.sc.fit_transform(train_x)
        ## transforming "train_x"
        test_x = self.sc.transform(test_x)

        ## transforming "The testset"
        test = self.sc.transform(test)
        return train_x, train_y,test_x, test_y,test


# if __name__=='__main__':
#     text=DealMiss()
#     FE=FeatureEngineering()
#     train,test=text.calls()
#     train,test,x,y,test=FE.calls()
