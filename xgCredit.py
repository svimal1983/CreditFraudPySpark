#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:11:44 2018

@author: Taranpreet
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


df= pd.read_csv('creditcard.csv',dtype={'Time': np.int32, 'Class': np.int8})
df[['Time','Class']] = df[['Time','Class']].applymap(np.int64)



df.Time = (df.Time - df.Time.mean())/df.Time.std()
#df.Time = (df.Time-df.Time.min())/(df.Time.max()-df.Time.min())

df.Amount = (df.Amount - df.Amount.mean())/df.Amount.std()

X = df.loc[:,:'Amount']
y= df.loc[:,'Class']

#train test split

# Create the DMatrix: churn_dmatrix
dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"binary:logistic", "max_depth":5}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=3, num_boost_round=70, metrics="error", 
                    as_pandas=True, seed=42)

# Print cv_results
print(cv_results)



y_pred1 = cv_results.predict(X)

# confusion matrix and classification report
print(confusion_matrix(y, y_pred1))
print(classification_report(y, y_pred1))



