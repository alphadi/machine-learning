# machine-learning
My first attempt at popular machine learning models ( K Neighbours Classifier)


import numpy as np    ## importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('Classified Data',index_col=1)     ## loading dataset to a dataframe


from sklearn.preprocessing import StandardScaler    ## preprocessing technique to scale the dataset
scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))    #applying scaler transform method
scaled_features

df.columns  #checking the columns
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])   #creating a new dataframe without the TARGET CLASS column

from sklearn.model_selection import train_test_split  

X = df_feat
y= df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)    ##splitting the train and test data

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)               ##initializing the K neighbors classifier
knn.fit(X_train,y_train)                                ## training the classifier


prediction = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print (confusion_matrix(y_test,prediction))               ## computing metrics with the test and predicted values
print (classification_report(y_test,prediction))

error_rate = []                                       ## calculating the error made by the model

for i in range(1,40):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
 plt.figure(figsize=(10,6))                               ## plotting error rate vs k value
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='*', markerfacecolor='red',markersize=10)
plt.title('Error rate vs K value')
plt.xlabel('k Value')
plt.ylabel('Error rate')


knn = KNeighborsClassifier(n_neighbors=15)          ##finding the optimal value K=15

knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('WITH K=15')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
