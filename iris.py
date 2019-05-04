# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:39:50 2019

@author: ADITHYA
"""

from sklearn import datasets
iris = datasets.load_iris()
X,y = X_iris[:,:2],y_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
print(X_train.shape,y_train.shape)

scaler= StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red','greenyellow','blue']
for i in range(len(colors)):
    xs=X_train[:,0][y_train==i]
    ys=X_train[:,1][y_train==i]
    plt.scatter(xs,ys,c=colors[i])
    
plt.legend(iris.target_names)
plt.xlabel('sepal length')
plt.ylabel('sepal width')

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train,y_train)
   
print(sgd.coef_) 
print(sgd.intercept_)
