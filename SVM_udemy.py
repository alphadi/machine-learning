# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:59:01 2019

@author: ADITHYA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer= load_breast_cancer()

cancer.keys()

print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

df_feat.head(n=5)

df_feat.info()

cancer['target']

from sklearn.model_selection import train_test_split

X = df_feat
y = cancer['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=101)

from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train,y_train)

predict = svm_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))

from sklearn.grid_search import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))