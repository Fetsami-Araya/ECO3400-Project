"""
@Date: March 20, 2021
@Authors: Fetsami Araya
@Description: This file will estimate LASSO, Ridge, Elastic Net, Gradient Boosting Tree, SVM, and neural network
machine learning models to nowcast Canadian GDP from 1992 through to 2020.

In addition, the predictive performance of these models will be assessed and compared to the predicitive accuracy
of a benchmark AR(1) model.
"""
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
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def readMasterData(dataset_type='Matthew'):
    if dataset_type != 'Matthew':
        _,_, data = createMasterData()
    else:
        data = pd.read_csv('./Master Datasets/M-ACTUAL.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    return data


def ar1model(data,lag=1):
    ar1 = AutoReg(data,lags=lag,old_names=True)
    ar1_fit = ar1.fit()
    return ar1_fit

@ ignore_warnings (category=ConvergenceWarning)
def elasticNetModel(X,y):
    parameters = {'alpha':np.logspace(-5, 5, 100, endpoint=True),
                'l1_ratio':np.arange(0, 1, 0.01),
                'fit_intercept':(True,False),
                'max_iter':[3000,5000,6000],
                'tol':[1e-5,1e-6]}
    elastic = ElasticNet()
    elastic_cv = RandomizedSearchCV(elastic,parameters,cv=TimeSeriesSplit(n_splits=5))
    elastic_cv_fit = elastic_cv.fit(X,y)
    best_params = elastic_cv_fit.best_params_
    elastic_model = ElasticNet(alpha=best_params['alpha'],l1_ratio=best_params['l1_ratio'],fit_intercept=best_params['fit_intercept'],max_iter=best_params['max_iter'],tol=best_params['tol'])
    return elastic_model.fit(X,y)

@ ignore_warnings (category=ConvergenceWarning)
def gradientBoostingTrees(X,y):
    parameters = {'loss':('ls','huber'),
                'learning_rate':np.linspace(0.1,1,10),
                'n_estimators':[100,300,500]}
    gb_tree = GradientBoostingRegressor()
    gb_tree_cv = RandomizedSearchCV(gb_tree, parameters,cv=TimeSeriesSplit(n_splits=5))
    gb_tree_cv_fit = gb_tree_cv.fit(X,y)
    best_params = gb_tree_cv_fit.best_params_
    gb_tree_fit = GradientBoostingRegressor(loss=best_params['loss'],n_estimators=best_params['n_estimators'],learning_rate=best_params['learning_rate']).fit(X,y)
    return gb_tree_fit

@ ignore_warnings (category=ConvergenceWarning)
def LASSO(X, y):
    parameters = {'alpha':np.arange(0.01, 1, 0.02),
                'fit_intercept':(True,False),
                'max_iter':[3000,5000,6000],
                'tol':[1e-5,1e-6]}
    lasso = Lasso()
    lasso_cv = RandomizedSearchCV(lasso,parameters,cv=TimeSeriesSplit(n_splits=5))
    lasso_cv_fit = lasso_cv.fit(X,y)
    best_params = lasso_cv_fit.best_params_
    lassofit = Lasso(alpha=best_params['alpha'], max_iter=best_params['max_iter'],fit_intercept=best_params['fit_intercept'],tol=best_params['tol']).fit(X,y)
    return lassofit

@ ignore_warnings (category=ConvergenceWarning)
def RIDGE(X, y):
    parameters = {'alpha':np.linspace(50,500,10),
                'fit_intercept':(True,False),
                'max_iter':[3000,5000,6000],
                'tol':[1e-5,1e-6]}
    ridge = Ridge()
    ridge_cv = RandomizedSearchCV(ridge,parameters,cv=TimeSeriesSplit(n_splits=5))
    ridge_cv_fit = ridge_cv.fit(X,y)
    best_params = ridge_cv_fit.best_params_
    ridgefit = Ridge(alpha=best_params['alpha'], max_iter=best_params['max_iter'],fit_intercept=best_params['fit_intercept'],tol=best_params['tol']).fit(X,y)
    return ridgefit

@ ignore_warnings (category=ConvergenceWarning)
def NeuralNet(X,y):
    param_grid = {'hidden_layer_sizes': [(25,25,25),(50,50,50,50,50), (50,50,50,50), (100,100,100,100,100), (200, 200, 200, 200)],
          'alpha': [0.001, 0.0001, 0.10],
          'learning_rate': ('constant','adaptive'),
          'activation': ['relu'],
          'solver': ['adam'],
          'max_iter': [200,350,500,600]}
    neural = MLPRegressor()
    neural_cv = RandomizedSearchCV(neural,param_grid,cv=TimeSeriesSplit(n_splits=5))
    neural_cv_fit = neural_cv.fit(X,y)
    best_params = neural_cv_fit.best_params_
    neural_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"], 
                        activation= best_params['activation'],
                        solver= best_params["solver"], alpha=best_params['alpha'],
                        learning_rate=best_params['learning_rate'],
                        max_iter= best_params['max_iter'])
    return neural_mlp.fit(X,y)

@ ignore_warnings (category=ConvergenceWarning)
def SVM_model(X,y):
    parameters = {'kernel':('rbf','linear','poly','sigmoid'),
                'degree' : [2,3,5,6,8],
                'gamma' : ('scale','auto'),
                'C': np.logspace(-0, 4, 8),
                'epsilon' : np.linspace(0.1,2,2),
                'shrinking': (True,False),
                'max_iter': [1,5,10,20]}
    svr = SVR()
    svr_cv = RandomizedSearchCV(svr, parameters,cv=TimeSeriesSplit(n_splits=5))
    svr_cv_fit = svr_cv.fit(X,y)
    best_params = svr_cv_fit.best_params_
    svr_fit = SVR(kernel=best_params['kernel'],degree=best_params['degree'],gamma=best_params['gamma'],C= best_params['C'],epsilon=best_params['epsilon'], 
                  shrinking= best_params['shrinking'], max_iter= best_params['max_iter']).fit(X,y)
    return svr_fit


def makePredictionDF(start_predict='2018-01-01',end_predict='2018-12-31'):
    file_name = './data/realGDP.csv'
    GDP = pd.read_csv(file_name)
    GDP = GDP[GDP['Estimates']=="Gross domestic product at market prices"]
    GDP = GDP[['REF_DATE','VALUE']]
    GDP['REF_DATE'] = pd.to_datetime(GDP['REF_DATE'])
    GDP.columns = ['date','GDP']
    GDP = GDP.set_index('date')
    models = ['LASSO','Ridge','Elastic Net','Gradient Boosting','Neural Net','SVM','AR(1)','Model Avg.']
    for model in models:
        GDP[model] = np.nan
    return GDP.loc[start_predict:end_predict]


def rollingWindow(start_predict='2020-01-01',end_predict='2020-12-31'):

    start = time.time()
    master = readMasterData()
    master.index = pd.DatetimeIndex(master.index).to_period('D')
       
    X_train = master.drop(['GDP'],axis='columns')
    y_train = master[['GDP']]

    X_train = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns,index=X_train.index)
    scalerGDP = StandardScaler().fit(y_train)
    y_train_scaled = scalerGDP.transform(y_train)
    y_train = pd.DataFrame(y_train_scaled,columns=y_train.columns,index=y_train.index)['GDP']

    prediction_df = makePredictionDF(start_predict,end_predict)

    for date in prediction_df.index:
        print('Making predictions for: ',date)
        X = X_train.loc[:date]
        y = y_train.loc[:date]
    
        elastic = elasticNetModel(X,y)
        gb_tree = gradientBoostingTrees(X,y)
        lasso = LASSO(X,y)
        ridge = RIDGE(X,y)
        neural = NeuralNet(X,y)
        svm = SVM_model(X,y)
        ar1 = ar1model(y)
        
        closest_date = X.index.get_loc(date,method='nearest')
        X_date = X.loc[X.index[closest_date],:].values.reshape(1, -1)
        
        ar1_predict = ar1.predict(closest_date).values[0]
        elastic_predict = elastic.predict(X_date)
        gb_tree_predict = gb_tree.predict(X_date)
        lasso_predict = lasso.predict(X_date)
        ridge_predict = ridge.predict(X_date)
        neural_predict = neural.predict(X_date)
        svm_predict = svm.predict(X_date)

        prediction_df.loc[date,'AR(1)'] = ar1_predict
        prediction_df.loc[date,'Elastic Net'] = elastic_predict
        prediction_df.loc[date,'Gradient Boosting'] = gb_tree_predict
        prediction_df.loc[date,'LASSO'] = lasso_predict
        prediction_df.loc[date,'Ridge'] = ridge_predict
        prediction_df.loc[date,'Neural Net'] = neural_predict
        prediction_df.loc[date,'SVM'] = svm_predict
        prediction_df.loc[date,'Model Avg.'] = (elastic_predict+gb_tree_predict+lasso_predict+ridge_predict+neural_predict+svm_predict)/6


    for model in ['AR(1)','Elastic Net','Gradient Boosting','LASSO','Ridge','Neural Net','SVM','Model Avg.']:
        prediction_df[model] = scalerGDP.inverse_transform(prediction_df[model])
    total_seconds = (time.time() - start)
    hours = int(total_seconds//(60*60))
    remaining_seconds = int(total_seconds-hours*60*60)
    minutes = remaining_seconds //60
    seconds = int(total_seconds-((hours*60*60)+(minutes*60)))
    print(f"Total Runtime: {hours} hours, {minutes} minutes, and {seconds} seconds")

    return prediction_df.astype(int)


def findRMSE(df):
    nrow = len(df)
    actual = df['GDP']
    predictions = df.drop('GDP',axis=1)
    root_errors = {'LASSO':[],'Ridge':[],'Elastic Net':[],'Gradient Boosting':[],'Neural Net':[],'SVM':[],'AR(1)':[],'Model Avg.':[]}
    for col in predictions:
        rmse = np.sqrt(np.mean(abs((1/nrow)*((actual - predictions[col])** 2))))
        mape = np.mean((np.abs(actual-predictions[col])/actual))*100
        root_errors[col] = [rmse,mape]
    rmse_df = pd.DataFrame(root_errors,index=['RMSE','MAPE']).T
    rmse_df = rmse_df.sort_values(by=['RMSE','MAPE'])

    return rmse_df

if __name__ == '__main__':
    df = rollingWindow().fillna(0)
    print(df)
    print(findRMSE(df))