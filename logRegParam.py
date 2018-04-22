#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:15:42 2018

@author: Taranpreet Singh
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
 
#import collections

def get_class_weights(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}






df= pd.read_csv('creditcard.csv',dtype={'Time': np.int32, 'Class': np.int8})

df.Time.describe()

df.Time = (df.Time - df.Time.mean())/df.Time.std()
#df.Time = (df.Time-df.Time.min())/(df.Time.max()-df.Time.min())

df.Amount = (df.Amount - df.Amount.mean())/df.Amount.std()



#separating target and predictor varables
X = df.loc[:,:'Amount']
y= df.loc[:,'Class']

#train_df.head()
#train_df.info()

# Setup the hyperparameter grid
#logspace base 10 args are start stop numbers 
#c is inverse of regularization high val means overfit
c_space = np.logspace(-5, 5, 50)
param_grid = {'C': c_space, 'penalty': ['l1','l2']}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression(class_weight = 'balanced')





#stratified folding

folds=3
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

cw=get_class_weights(y)

# grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3 )
# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid , scoring='f1',cv=skf.split(X,y),verbose=3)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))
#
#Tuned Logistic Regression Parameters: {'C': 0.68664884500429979, 'penalty': 'l2'}
#Best score is 0.720258229373866

#Tuned Logistic Regression Parameters: {'C': 0.00026826957952797272, 'penalty': 'l1'}
#Best score is 0.14376434432682675


#C=0.000104811313415, penalty=l1, score=0.1452513966480447
#C=0.000268269579528, penalty=l1, score=0.15294742432288902,
y_pred1 = logreg_cv.predict(X)

# confusion matrix and classification report
print(confusion_matrix(y, y_pred1))
print(classification_report(y, y_pred1))
# Recall 0.90

# without balanced

knn = KNeighborsClassifier()

gridKnn = {'n_neighbors': [6], 'p': [1,1.5,2],'weights':['uniform','distance']}

knn_cv = GridSearchCV(knn,gridKnn , scoring='f1',cv=skf.split(X,y),verbose=3,n_jobs= -1)
knn_cv.fit(X,y)

print("Tuned knn Parameters: {}".format(knn_cv.best_params_)) 
print("Best score is {}".format(knn_cv.best_score_))
#Tuned Logistic Regression Parameters: {'n_neighbors': 6, 'p': 2, 'weights': 'distance'}
#Best score is 0.8493006146539605





