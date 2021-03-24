# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:43:33 2021

@author: Matthew
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import pandas as pd

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

#train the model
reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train)

y_pred = reg.predict(X_test)

#evaluate
from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse


      
      
      
      
      