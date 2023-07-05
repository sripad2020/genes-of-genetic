import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
data=pd.read_csv('test.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
#for i in data.select_dtypes(include='number').columns.values:
#    sn.boxplot(data[i])
#    plt.show()
#blood cell count
data['z-scores']=(data['Blood cell count (mcL)']-data['Blood cell count (mcL)'].mean())/data['Blood cell count (mcL)'].std()
df=data[(data['z-scores']>-3)&(data['z-scores']<3)]
qu1=df['Blood cell count (mcL)'].quantile(0.25)
qu3=df['Blood cell count (mcL)'].quantile(0.75)
iqr=qu3-qu1
upp=qu3+1.5*iqr
low=qu1-1.5*iqr
df=df[(df['Blood cell count (mcL)']>low)&(df['Blood cell count (mcL)']<upp)]
print(data.shape)
thres=3
for i in data.select_dtypes(include="number").columns.values:
    mean=df[i].mean()
    std=df[i].std()
    up=mean+thres*std
    lo=mean-thres*std
    df=df[(df[i]>lo)&(df[i]<up)]
print('-----------')
print(df.shape)
'''for i in df.select_dtypes(include="number").columns.values:
    sn.boxplot(df[i])
    plt.show()'''
#for i in df.select_dtypes(include='number').columns.values:
#    sn.boxplot(df[i])
#    plt.show()
#for i in df.select_dtypes(include='object').columns.values:
#    if len(df[i].value_counts())< 5:
#        sn.countplot(df[i])
#        plt.show()
''''print(df.select_dtypes(include='number').columns.values)
print(df.select_dtypes(include='object').columns.values)
plt.figure(figsize=(17, 6))
corr = df.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()'''

print(df.select_dtypes(include='object').columns.values)
print('----------------------------------------------------')
print(df.select_dtypes(include='number').columns.values)


lab=LabelEncoder()
df["blood test result"]=lab.fit_transform(df['Blood test result'])
df["genes mother"]=lab.fit_transform(df["Genes in mother's side"])
df["father"]=lab.fit_transform(df["Inherited from father"])
df["maternal gene"]=lab.fit_transform(df["Maternal gene"])
df['paternal gene']=lab.fit_transform(df["Paternal gene"])
df['status']=lab.fit_transform(df['Status'])
df['respiratory Rate (breaths/min)']=lab.fit_transform(df['Respiratory Rate (breaths/min)'])
df['heart rate']=lab.fit_transform(df['Heart Rate (rates/min'])
df['follow']=lab.fit_transform(df['Follow-up'])
df['gender']=lab.fit_transform(df['Gender'])
df['parental-consent']=lab.fit_transform(df['Parental consent'])
df['respiratory']=lab.fit_transform(df['Respiratory Rate (breaths/min)'])
df['pob']=lab.fit_transform(df['Place of birth'])
df['folic acid details (peri-conceptional)']=lab.fit_transform(df['Folic acid details (peri-conceptional)'])
df['h/O serious maternal illness']=lab.fit_transform(df['H/O serious maternal illness'])
df['history of anomalies in previous pregnancies']=lab.fit_transform(df['History of anomalies in previous pregnancies'])
df['birth defects']=lab.fit_transform(df['Birth defects'])



x=df[["paternal gene","status",
      "gender","parental-consent","respiratory","pob","folic acid details (peri-conceptional)",
      "h/O serious maternal illness","history of anomalies in previous pregnancies","birth defects",
      'Patient Age', 'Blood cell count (mcL)','Test 1', 'Test 2', 'Test 3', 'Test 4',
      'Test 5','No. of previous abortion',
      'White Blood cell count (thousand per microliter)','blood test result']]

'''x=df[['Test 1','Test 2','Test 3','Test 4','Test 5',
      'White Blood cell count (thousand per microliter)'
      ,"gender","parental-consent","respiratory","folic acid details (peri-conceptional)"
      ,"paternal gene","status",'No. of previous abortion','blood test result'
      ,'Patient Age', 'Blood cell count (mcL)']]'''

x_train,x_test,y_train,y_test=train_test_split(x,df['heart rate'])
lr=LogisticRegression(max_iter=200)
lr.fit(x_train,y_train)
print('The logistic regression: ',lr.score(x_test,y_test))

xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print("the Xgb : ",xgb.score(x_test,y_test))

lgb=LGBMClassifier()
lgb.fit(x_train,y_train)
print('The LGB',lgb.score(x_test,y_test))

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
print('Dtree ',tree.score(x_test,y_test))

rforest=RandomForestClassifier()
rforest.fit(x_train,y_train)
print('The random forest: ',rforest.score(x_test,y_test))

adb=AdaBoostClassifier()
adb.fit(x_train,y_train)
print('the adb ',adb.score(x_test,y_test))

grb=GradientBoostingClassifier()
grb.fit(x_train,y_train)
print('Gradient boosting ',grb.score(x_test,y_test))

bag=BaggingClassifier()
bag.fit(x_train,y_train)
print('Bagging',bag.score(x_test,y_test))

sn.pairplot(df)
plt.show()