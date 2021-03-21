"""
@Date: March 20, 2021
@Authors: Fetsami Araya
@Description: This file will estimate LASSO, Ridge, Elastic Net, Gradient Boosting Tree, SVM, and neural network
machine learning models to nowcast Canadian GDP from 1960 through to 2020.

In addition, the predictive performance of these models will be assessed and compared to the predicitive accuracy
of a benchmark AR(1) model.
"""

# Imports
import numpy as np
import pandas as pd
from data_clean import createMasterData

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

def readData(dataset_type='og'):
    master_data, master_data_some_na, master_data_no_na = createMasterData()
    master_data_dict = {'og':master_data, 'some_na':master_data_some_na, 'no_na':master_data_no_na}
    data = master_data_dict[dataset_type]
    return data

def split_data(df,start_date,end_date):
    return df.loc[start_date:end_date]


def ar1model(data,lag=1):
    ar1 = AutoReg(data,lags=lag)
    ar1_fit = ar1.fit()
    return ar1_fit

def elasticNetModel(X,y):
    elastic = ElasticNet(random_state=42)
    elastic_fit = elastic.fit(X,y)
    return elastic_fit

def gradientBoostingTrees(X,y):
    gb_tree = GradientBoostingRegressor(random_state=42)
    gb_tree_fit = gb_tree.fit(X,y)
    return gb_tree_fit


if __name__ == '__main__':
    data = readData()
    split = split_data(data,'1961-01-01','2000-01-01')
    ar = ar1model(split['GDP'])
    print(ar.predict())


