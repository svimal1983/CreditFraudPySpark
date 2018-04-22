# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:51:20 2018

@author: Taranpreet
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from datetime import datetime
from sklearn.metrics import f1_score

df= pd.read_csv('creditcard.csv')

#df = pd.DataFrame(results)

#df.columns =  results[0].keys()

df.head()
#checking data types df.dtypes

df.info()

df[['Time','Class']] = df[['Time','Class']].applymap(np.int64)

df.describe()
#mean 0.17% fraud cases, imbalance in target
df.Class.describe()


# amount with class

# box plot 
df.loc[df['Class'] == 0].plot(y ='Amount', kind='box')
plt.yscale('log')
# box plot of the Amount prices for non fraud class
df.loc[df['Class'] == 1].plot( y='Amount', kind='box')
plt.yscale('log')


#scaling time and amount
df.Time.describe()

df.Time = (df.Time - df.Time.mean())/df.Time.std()
#df.Time = (df.Time-df.Time.min())/(df.Time.max()-df.Time.min())

df.Amount = (df.Amount - df.Amount.mean())/df.Amount.std()

#separating target and predictor varables
X = df.loc[:,:'Amount']
y= df.loc[:,'Class']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state=42, stratify=y)

y_test.describe()
y_train.describe()
# target is finding time it takes when we scale the models
# we can change the train set size to see performance

#Knn algorithm 
#choosing k

neighbors = np.arange(4, 6)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# we need to see if this can be parallelized


knn = KNeighborsClassifier(n_neighbors=4,p=1,verbose=3)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
f1_score(y_test, y_pred)



startTime = datetime.now()

for i, k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)
    #accuracy on the training set
    
    y_pred = knn.predict(X_test)
    
    train_accuracy[i] = knn.score(X_train, y_train)

    #accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

print(datetime.now() - startTime)


# changing order of test and pred
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))


#positive class is 1 as default
f1_score(y_test, y_pred)

# plot
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('No of neighbors')
plt.ylabel('Accuracy')
plt.show()


cw=get_class_weights(y_train)
logreg = LogisticRegression(solver='saga')

# fitting model
logreg.fit(X_train,y_train)

# y_pred
y_pred = logreg.predict(X_test)

# confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

f1_score(y_test, y_pred)


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))









