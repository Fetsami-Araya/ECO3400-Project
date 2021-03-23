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
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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
    elastic = ElasticNet(max_iter=30000,tol=1e-5)
    elastic_fit = elastic.fit(X,y)
    return elastic_fit

def gradientBoostingTrees(X,y):
    gb_tree = GradientBoostingRegressor()
    gb_tree_fit = gb_tree.fit(X,y)
    return gb_tree_fit

def LASSO(X, y):
    mod = Lasso(max_iter = 30000,tol=1e-5)
    lassofit = mod.fit(X,y)
    return lassofit

def RIDGE(X, y):
    mod = Ridge(max_iter = 30000, tol=1e-5,normalize = True) #Dealing with both large and small values, hence the normalization
    ridge_mod = mod.fit(X,y)
    return ridge_mod

def makePredictionDF(start_predict='2010-01-01',end_predict='2018-12-31'):
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
    return GDP.loc[start_predict:end_predict]

def rollingWindow(start_predict='2010-01-01',end_predict='2018-12-31'):
    _,_,master = createMasterData()
    master.index = pd.DatetimeIndex(master.index).to_period('D')
    X_train = master.copy().drop('GDP',axis=1)
    y_train = master.copy()['GDP']

    SS = StandardScaler()
    X_train = SS.fit_transform(X_train)
    
    X_train = pd.DataFrame(X_train, columns=master.columns.values[1:]).set_index(y_train.index)

    prediction_df = makePredictionDF(start_predict,end_predict)

    for date in prediction_df.index:
        print('Making predictions for: ',date)
        X = X_train.loc[:date]
        y = y_train.loc[:date]

        elastic = elasticNetModel(X,y)
        gb_tree = gradientBoostingTrees(X,y)
        lasso = LASSO(X,y)
        ridge = RIDGE(X,y)
        ar1 = ar1model(y)
        
        closest_date = X.index.get_loc(date,method='nearest')
        X_date = X.loc[X.index[closest_date],:].values.reshape(1, -1)
        
        prediction_df.loc[date,'AR(1)'] = ar1.predict(closest_date).values[0]
        prediction_df.loc[date,'Elastic Net'] = elastic.predict(X_date)
        prediction_df.loc[date,'Gradient Boosting'] = gb_tree.predict(X_date)
        prediction_df.loc[date,'LASSO'] = lasso.predict(X_date)
        prediction_df.loc[date,'Ridge'] = ridge.predict(X_date)

    return prediction_df

if __name__ == '__main__':
    df = rollingWindow().fillna(0).astype(int)
    predictions = df.drop('GDP',axis=1)
    actual = df['GDP']
    root_errors = {'LASSO':[],'Ridge':[],'Elastic Net':[],'Gradient Boosting':[],'Neural Net':[],'SVM':[],'AR(1)':[]}
    for col in predictions:
        rmse = np.sqrt(mean_squared_error(actual,predictions[col]))
        root_errors[col] = [rmse]
    rmse_df = pd.DataFrame(root_errors,index=['RMSE'])
    print(rmse_df)
    


