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

#define variables
X = Mdataset.drop(['Real GDP','Date'], axis='columns')
X.head()

y = Mdataset['Real GDP']

             
#train the machine
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.30, random_state=42)

len(X_train)
len(X_test)

#predict 
from sklearn.svm import SVR

Model = SVR()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)


#evaluate
from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_test, y_pred)
mse

Model.score(X_test, y_test)

