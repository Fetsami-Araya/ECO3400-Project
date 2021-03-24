# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:42:05 2021

@author: Matthew
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

#scale the data
from sklearn.preprocessing import StandardScaler

#remove the data column since this will get confused when we standardize the date
m= Mdataset.drop(['Date'], axis='columns')
m.head()

#standardize the data and keep headers
scaler = StandardScaler()
q = scaler.fit_transform(m)
q_df = pd.DataFrame(q, columns = m.columns)


#define variables
X = q_df.drop(['Real GDP'], axis='columns')
X.head()

y = q_df['Real GDP']
y.head()
             
#split into testing and training data but use 90% of beginning sequential rows to train and rest to test
X_train = X[:int(X.shape[0]*0.9)]
X_test = X[int(X.shape[0]*0.9):]
y_train = y[:int(X.shape[0]*0.9)]
y_test = y[int(X.shape[0]*0.9):]

len(X_train)
len(X_test)


#train and predict 
from sklearn.svm import SVR

Model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
svr= Model.fit(X_train, y_train)


y_pred = svr.predict(X_test)

#evaluate
from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

Model.score(X_test,y_test)



### Extras ###
# X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.30, random_state=42)

# scaler = StandardScaler()
# scaled_X_train = scaler.fit_transform(X_train)
# scaled_X_train

# scaled_y_train = scaler.fit_transform(y_train)
# scaled_y_train

# scaled_X_test = scaler.fit_transform(X_test)
# scaled_X_test