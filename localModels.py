#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:09:18 2018

@author: Taranpreet Singh
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import f1_score

df= pd.read_csv('creditcard.csv')


df[['Time','Class']] = df[['Time','Class']].applymap(np.int64)

df = df.drop(['Time'], axis=1)

df.Amount = (df.Amount - df.Amount.mean())/df.Amount.std()


X = df.loc[:,:'Amount']
y= df.loc[:,'Class']


# fun to get results
def get_results(model):
    #lists get results
    scores=[]
    Times=[]
    for i in range(4,0,-1):
      
        size=0.2*i
        #split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size,
                                                    random_state=42, stratify=y)
        
        #time fit time
        fitStart = datetime.now()
        #fit
        model.fit(X_train,y_train)
        fitTime = datetime.now() - fitStart 
        
        #pred part
        predStart = datetime.now()
        y_pred = model.predict(X_test)
        predTime = datetime.now() - predStart
        
        Times.append((fitTime,predTime))
        #score
        scores.append(f1_score(y_test, y_pred))
        
    return(scores,Times)
        
      
      
#Models
lg = LogisticRegression(C =0.6866, penalty = 'l2')

knn = KNeighborsClassifier(n_neighbors=6,p=2,weights = 'distance')
#xgb = 
XgB = xgb.XGBClassifier(objective='binary:logistic', n_estimators=900, max_depth=6,learning_rate = 0.05,colsample_bytree=0.6)


mods = [lg,knn,XgB]

final = {}
for i,model in enumerate(mods):
    final[i] = get_results(model)

final


