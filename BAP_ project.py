# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:37:54 2019

@author: ADITHYA
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

##loading the csv to a dataframe
df = pd.read_csv('ATM_Withdrawl_Prediction_Data (1).csv')

##checking for null values
df.isna().sum()

## commands to understand the data set

df.head()

df.describe()
df.info()

## Lets try appliying some basic EDA to understand the data better

#finding relation between ATM rating and how it looks
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.countplot(x='ATM RATING',hue='ATM looks',data=df)

## graph to find the number of ATM by location
plt.figure(figsize=(10,6))
sns.countplot(x='ATM TYPE',hue='ATM Placement',data=df)

##graph to find the ATM type by location
plt.figure(figsize=(10,6))
sns.countplot(x='ATM TYPE',hue='ATM Location TYPE',data=df)

## Distinguish ATM location type using amount withdrawn

plt.figure(figsize=(11,7))
sns.boxplot(x='ATM Location TYPE',y='AmountWithDrawn',data=df,palette="coolwarm")

## Frequency distribution of avg number of withdrawals per week 
plt.figure(figsize=(10,6))
sns.distplot(df['Avg No of Withdrawls Per Week'],kde=False,color='darkred',bins=30)

plt.figure(figsize=(10,6))
sns.countplot(x='Avg Withdrawls Per Hour',hue='ATM looks',data=df)

##plotting the correlation of columns using a  heatmap
sns.heatmap(df.corr(),cmap='viridis')

##some advanced plots -
plt.figure(figsize=(11,7))
sns.lmplot(x='Avg No of Withdrawls Per Week',y='Avg Withdrawls Per Hour',col='ATM Placement',data=df,palette='Set1')

## plot to distinguish ATMs based on number of houses and other ATMs in 1 km radius
plt.figure(figsize=(11,7))
sns.lmplot(x='Estimated Number of Houses in 1 KM Radius',y='No of Other ATMs in 1 KM radius',hue ='ATM looks',col = 'ATM Placement',data=df,palette='Set1')

##  facetgrid tp distinguish number of withdrawals by ATM Type

plt.figure(figsize=(12,6))
sns.set_style('darkgrid')
g=sns.FacetGrid(df,hue='ATM TYPE',sharey=True,size=6,aspect=5)
g.map(plt.hist,'Avg No of Withdrawls Per Week',bins=50,alpha=0.7)

## now lets try using the t test ( we will check of significance between ATM rating and average wait time)

import cufflinks as cf
cf.go_offline()

df['AmountWithDrawn'].iplot(kind='hist',bins=30,color='green')


from scipy import stats

a = df['ATM RATING']
b = df['Average Wait Time']

##choosing degree of freedom to be 1
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

## Calculate the Standard Deviation
#Calculate the variance to get the standard deviation
s = np.sqrt((var_a + var_b)/2)

## Calculate the t-statistics
t = (a.mean() - b.mean())/(s*np.sqrt(2/10))

## Compare with the critical t-value
#Degrees of freedom

df = 2*10 -2 

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(2*p))



##dropping the unnecessary columns
df.drop(['ID','ATM Zone','ATM Near','ATM Attached to','ATM Prox'],axis=1,inplace=True)

df.drop(['ATM TYPE','ATM Location TYPE','Holiday Sequence'],axis=1,inplace=True)

##Converting the categorical variables to numerical by the method of zero hot encoding

cols = pd.get_dummies(data=df,columns=['ATM Placement','ATM looks','Day Type'])

df = pd.concat([df,cols],axis=1)
df.info()
# Dropping the categorical columns
df.drop(['ATM Placement','ATM looks','Day Type'],axis=1,inplace=True)
df.info()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Splitting the dataset to train and test samples
X_train,X_test,y_train,y_test = train_test_split(df.drop('AmountWithDrawn',axis=1),df['AmountWithDrawn'],test_size=0.30,random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)
lm.score(X_train,y_train)
##printing the coefficients of the model
print('Coefficients: \n', lm.coef_)

linear_predictions = lm.predict( X_test)

#Scatter plot between test and predicted values
plt.scatter(y_test,linear_predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# calculating the regression metrics
from sklearn import metrics

print('Mean Average Error:', metrics.mean_absolute_error(y_test, linear_predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, linear_predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, linear_predictions)))

##Distribution plot of the residual values

sns.distplot((y_test-linear_predictions),bins=30,color='red')

## Ridge regression 
from sklearn.linear_model import Ridge

rm = Ridge(alpha=1.0)
rm.fit(X_train,y_train)
rm.score(X_train,y_train)

##printing the coefficients of the model
print('Coefficients: \n', rm.coef_)

ridge_predictions = rm.predict(X_test)

#Scatter plot between test and predicted values
plt.scatter(y_test,ridge_predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

print('Mean Average Error:', metrics.mean_absolute_error(y_test, ridge_predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, ridge_predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, ridge_predictions)))

## Ridge regression with CV
from sklearn.linear_model import RidgeCV

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X_train, y_train)
clf.score(X_train,y_train)

## printing the coefficients of the model
print('Coefficients: \n', clf.coef_)

from sklearn.linear_model import LassoCV

reg = LassoCV(cv=5).fit(X_train, y_train)

##min max scaling

y_new = pd.DataFrame(columns='amount withdrawn')
y_new['amount withdrawn'] = (y - y.min())/(y.max()-y.min)


##Logistic regression with CV
from sklearn.linear_model import LogisticRegressionCV

logmodel = LogisticRegressionCV(cv=5,random_state=0,multi_class='multinomial')

X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_new).reshape(-1,1)


logmodel.fit(X_train,y_train)

logpred = logmodel.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, logpred))
print('MSE:', metrics.mean_squared_error(y_test, logpred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, logpred)))

