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
    ar1 = AutoReg(data,lags=lag,old_names=True)
    ar1_fit = ar1.fit()
    return ar1_fit

def elasticNetModel(X,y):
    elastic = ElasticNet(max_iter=20000,tol=1e-3)
    elastic_fit = elastic.fit(X,y)
    return elastic_fit

def gradientBoostingTrees(X,y):
    gb_tree = GradientBoostingRegressor()
    gb_tree_fit = gb_tree.fit(X,y)
    return gb_tree_fit


def makePredictionDF(start_predict='2010-01-01',end_predict='2017-10-01'):
    file_name = './data/realGDP.csv'
    GDP = pd.read_csv(file_name)
    GDP = GDP[GDP['Estimates']=="Gross domestic product at market prices"]
    GDP = GDP[['REF_DATE','VALUE']]
    GDP['REF_DATE'] = pd.to_datetime(GDP['REF_DATE'])
    GDP.columns = ['date','GDP']
    GDP = GDP.set_index('date')
    models = ['LASSO','Ridge','Elastic Net','Gradient Boosting','Neural Net','SVM','AR(1)']
    for model in models:
        GDP[model] = np.nan
    ar_gdp = ar1model(GDP['GDP'])
    GDP['AR(1)'] = ar_gdp.predict()
    return GDP.loc[start_predict:end_predict]

def rollingWindow(start_predict='2010-01-01',end_predict='2017-10-01'):
    _,_,master = createMasterData()
    X_train = master.copy().drop('GDP',axis=1)
    y_train = master.copy()['GDP']

    prediction_df = makePredictionDF(start_predict,end_predict)

    for date in prediction_df.index:
        print('Predictions for: ',date)
        X = X_train.loc[:date]
        y = y_train.loc[:date]

        elastic = elasticNetModel(X,y)
        gb_tree = gradientBoostingTrees(X,y)
        
        closest_date = X.index.get_loc(date,method='nearest')
        X_date = X.loc[X.index[closest_date],:].values.reshape(1, -1)
        
        prediction_df.loc[date,'Elastic Net'] = elastic.predict(X_date)
        prediction_df.loc[date,'Gradient Boosting'] = gb_tree.predict(X_date)

    return prediction_df

if __name__ == '__main__':
    print(rollingWindow())


