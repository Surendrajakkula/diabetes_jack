from ast import Return
from pyexpat import model
import warnings
from sklearn import model_selection
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
def diabetes_detection():
    df = pd.read_csv('diabetes_data_upload.csv')
    df.isna().sum()
#df.info()
    df['class'] = df['class'].apply(lambda x: 0 if x=='Negative' else 1)
    X= df.drop(['class'],axis=1)
    y=df['class']
    objList = X.select_dtypes(include = "object").columns
    le = LabelEncoder()
    for feat in objList:
        X[feat] = le.fit_transform(X[feat].astype(str))
#print (X.info())
    X.corrwith(y)
    X_fs = X[['Polyuria', 'Polydipsia','Age', 'Gender','partial paresis','sudden weight loss','Irritability', 'delayed healing','Alopecia','Itching']]
    X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size = 0.2,stratify=y, random_state = 1234)
    minmax = MinMaxScaler()
    X_train[['Age']] = minmax.fit_transform(X_train[['Age']])
    X_test[['Age']] = minmax.transform(X_test[['Age']])
    rf = RandomForestClassifier(criterion='gini',n_estimators=100)
    rf.fit(X_train,y_train)
    kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=7)
    scoring = 'accuracy'

    acc_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = kfold,scoring=scoring)
    acc_rf.mean()
    kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=7)
    scoring = 'accuracy'

    acc_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = kfold,scoring=scoring)
    acc_rf.mean()
    #print(acc_rf.mean())
    import pickle

    file= open('rf_final.pkl','wb')
    pickle.dump(rf,file)

