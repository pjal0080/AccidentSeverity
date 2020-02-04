import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost



df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")
y = df1.iloc[:,0].values
aid = df2['Accident_ID']
df1['Severity'].describe().count()

df1.describe()
df1.info()
df2.info()

mp = {0 : 'Highly_Fatal_And_Damaging', 1 : 'Minor_Damage_And_Injuries', 2 : 'Significant_Damage_And_Fatalities', 3 : 'Significant_Damage_And_Serious_Injuries'}

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
y = lbl.fit_transform(y)

df1 = df1.drop(['Severity'],axis = 1)

y1 = pd.DataFrame(y,columns = ['Severity'])

df1 = pd.concat([df1,y1], axis = 1)

corrm = df1.corr()
sns.heatmap(corrm,vmax = .8,annot = True,square = True)



df1['Safety_Score'].describe()
df1['Accident_Type_Code'].describe()

sns.boxplot(x = df1['Safety_Score'])
sns.boxplot(x = df2['Safety_Score'])

q1=df1["Safety_Score"].quantile(0.25)
q3=df1["Safety_Score"].quantile(0.75)
iqr=q3-q1
low=q1-1.5*iqr
high=q3+1.5*iqr
df1=df1.loc[(df1["Safety_Score"]>low) & (df1["Safety_Score"]<high)]
sns.boxplot(x = df1["Safety_Score"])

df1['Safety_Score'].skew()

sns.distplot(df1["Safety_Score"], color="m", label="Skewness : %.2f"%(df1['Safety_Score'].skew()))
sns.distplot(df1["Accident_Type_Code"], color="m", label="Skewness : %.2f"%(df1['Accident_Type_Code'].skew()))


df1 = df1.drop(['Accident_ID'],axis = 1)
df2 = df2.drop(['Accident_ID'],axis = 1)


df1['Safety_Score'].describe()
sns.distplot(df1['Safety_Score'])
sns.distplot(df1['Total_Safety_Complaints'])
sns.distplot(df1['Control_Metric'])


X_train = df1.iloc[:,0:3].values
y_train = df1.iloc[:,4].values
X_test = df2.iloc[:,0:3].values


'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score

etc = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsExtC = GridSearchCV(etc,param_grid = ex_param_grid, cv=10, scoring="accuracy", n_jobs= -1)

gsExtC.fit(X_train,y_train)

ExtC_best = gsExtC.best_estimator_

gsExtC.best_params_
gsExtC.best_score_


RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=10, scoring="accuracy", n_jobs= -1)
gsRFC.fit(X_train,y_train)

RFC_best = gsRFC.best_estimator_
gsRFC.best_params_
gsRFC.best_score_


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= -1)

gsGBC.fit(X_train,y_train)

GBC_best = gsGBC.best_estimator_
gsGBC.best_score_



DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state = 0)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=10, scoring="accuracy", n_jobs= -1)

gsadaDTC.fit(X_train,y_train)

gsadaDTC.best_score_


ada_best = gsadaDTC.best_estimator_


'''
from sklearn.svm import SVC

SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=10, scoring="accuracy", n_jobs= -1)

gsSVMC.fit(X_train,y_train)

SVMC_best = gsSVMC.best_estimator_

'''

vc = VotingClassifier(estimators = [('rfc',RFC_best ),('dtc',ada_best),('etc',ExtC_best)], voting = 'soft',n_jobs = -1)
vc.fit(X_train,y_train)
vc.score(X_train,y_train)
y_predvc = vc.predict(X_test)








y_predvc = pd.DataFrame(y_predvc,columns = ['Severity'])
y_predvc['Severity'] = y_predvc['Severity'].map(mp)
y_predvc.describe()

ans = pd.concat([aid,y_predvc],axis = 1)
ans.to_csv("sub.csv",index = False)




'''
y_pred1 = pd.DataFrame(y_pred1,columns = ['Severity'])
y_pred1['Severity'] = y_pred1['Severity'].map(mp)
y_pred1.describe()

ans1 = pd.concat([aid,y_pred],axis = 1)
ans1.to_csv("subm.csv",index = False)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train,y_pred)

'''