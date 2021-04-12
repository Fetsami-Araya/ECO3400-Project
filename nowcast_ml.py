"""
@Date: March 20, 2021
@Authors: Fetsami Araya
@Description: This file will estimate LASSO, Ridge, Elastic Net, Gradient Boosting Tree, SVM, and neural network
machine learning models to nowcast Canadian GDP from 1992 through to 2020. The root mean squared error and 
mean absolute percenage error of prediction are calculate on the last two years od data (2019 and 2020)

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
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from dm_test import dm_test
import random

def readMasterData(dataset_type='Matthew'):
    """
    Returns the data frame including all macroeconomic predictors of GDP

    Arguments:
    @dataset_type: Determines which csv/dataframe to use, either the one generated by the code or the one
    compiled by Matthew Raykha

    Returns:
    data: Data frame of macroeconomic data
    """
    if dataset_type != 'Matthew':
        _,_, data = createMasterData()
    else:
        data = pd.read_csv('./Master Datasets/M-ACTUAL.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    return data


def ar1model(data,lag=1):
    """
    Returns an AR(1) model
    """
    ar1 = AutoReg(data,lags=lag,old_names=True)
    ar1_fit = ar1.fit()
    return ar1_fit

@ ignore_warnings (category=ConvergenceWarning)
def elasticNetModel(X,y):
    """
    Returns a fitted Elastic Net Model using the best parameters found through a randomized (time-series) cross-validation
    process
    """
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
    """
    Returns a fitted Gradient Boosting Tree Regressor Model using the best parameters found through a randomized (time-series) cross-validation
    process
    """
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
    """
    Returns a fitted Lasso regression model using the best parameters found through a randomized (time-series) cross-validation
    process
    """
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
    """
    Returns a fitted Ridge regression model using the best parameters found through a randomized (time-series) cross-validation
    process
    """
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
    """
    Returns a fitted feed-forward multi-layer perceptron neural network model using the best parameters found through a randomized (time-series) cross-validation
    process
    """
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
    """
    Returns a fitted Support Vector Machine regression model using the best parameters found through a randomized (time-series) cross-validation
    process
    """
    parameters = {'kernel':('rbf','linear','poly'),
                'degree' : [3,4,5],
                'gamma' : ('scale','auto'),
                'C': [1,2],
                'shrinking': (True,False)}
    svr = SVR(cache_size=7000)
    svr_cv = RandomizedSearchCV(svr, parameters,cv=TimeSeriesSplit(n_splits=5))
    svr_cv_fit = svr_cv.fit(X,y)
    best_params = svr_cv_fit.best_params_
    svr_fit = SVR(kernel=best_params['kernel'],degree=best_params['degree'],gamma=best_params['gamma'],C= best_params['C'], 
                  shrinking= best_params['shrinking']).fit(X,y)
    return svr_fit


@ ignore_warnings (category=ConvergenceWarning)
def SGD_model(X,y):
    """
    Returns a fitted Stochastic Gradient Descent regression model using the best parameters found through a randomized (time-series) cross-validation
    process.

    This model was introduced to avoid some computational problems we discovered with SVM.
    """
    parameters = {'loss':('squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'),
                'alpha':10.0**-np.arange(1,7),
                'penalty':('l2','l1','elasticnet'),
                'l1_ratio':np.linspace(0.01,1,20),
                'fit_intercept':(True,False)}
    sgd = SGDRegressor()
    sgd_cv = RandomizedSearchCV(sgd, parameters,cv=TimeSeriesSplit(n_splits=5))
    sgd_cv_fit = sgd_cv.fit(X,y)
    best_params = sgd_cv_fit.best_params_
    sgd_fit = SGDRegressor(loss=best_params['loss'],
                        alpha = best_params['alpha'],
                        penalty=best_params['penalty'],
                        l1_ratio=best_params['l1_ratio'],
                        fit_intercept = best_params['fit_intercept'],
                        max_iter = np.ceil(10**6/len(y))).fit(X,y)
    return sgd_fit

def makePredictionDF(start_predict='2018-01-01',end_predict='2018-12-31'):
    """
    Creates a data frame that will store all the predictions for each model and the actual values of GDP for each quarter

    Arguments:
    @start_predict: First quarter to get predictions for
    @end_predict: Last quarter to get predictions for

    Returns
    GDP: Dataframe of actual GDP and NaN values for machine learning models
    """
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


def rollingWindow(start_predict='2019-01-01',end_predict='2020-12-31'):
    """"
    Main function that performs nowcasting. This function uses a pseudo real-time estimation technique that expands the window of data available,
    training and testing each machine learning model on all the data that was available up to that quarter.

    Each model is trained and hyper-parameter tuned using a five-fold time series cross-validation split, and each model is than estimated
    with those hyper-parameters and is used to predict GDP for the next quarter.

    Arguments:
    @start_predict: First quarter to get predictions for
    @end_predict: Last quarter to get predictions for

    Returns
    @predictions_df: Dataframe containing actual GDP and the predictions generated by each machine learning model.
    
    """

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
        print('Generating predictions for: ',date)
        print('Training models')
        X = X_train.loc[:date]
        y = y_train.loc[:date]
        print('Elastic Net')
        elastic = elasticNetModel(X,y)
        print('Gradient Boosting Tree')
        gb_tree = gradientBoostingTrees(X,y)
        print('Lasso')
        lasso = LASSO(X,y)
        print('Ridge')
        ridge = RIDGE(X,y)
        print('Neural Net')
        neural = NeuralNet(X,y)
        print('SVM')
        svm = SVM_model(X,y)
        #svm = SGD_model(X,y)
        print('AR(1)')
        ar1 = ar1model(y)
        
        closest_date = X.index.get_loc(date,method='nearest')
        X_date = X.loc[X.index[closest_date],:].values.reshape(1, -1)
        
        print('Getting predictions')
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
        # Made modifications because SVM was not running. To modify code comment/uncomment the svm_predict code above and below, add it svm_predict to the model average
        # below on line 211, and change the denominator from 5 to 6
        #prediction_df.loc[date,'SVM'] = np.NaN
        prediction_df.loc[date,'SVM'] = svm_predict
        prediction_df.loc[date,'Model Avg.'] = (elastic_predict+gb_tree_predict+lasso_predict+ridge_predict+svm_predict+neural_predict)/6
        print(f'Predictions for {date} complete','\n')


    for model in ['AR(1)','Elastic Net','Gradient Boosting','LASSO','Ridge','Neural Net','SVM','Model Avg.']:
        prediction_df[model] = scalerGDP.inverse_transform(prediction_df[model])
    total_seconds = (time.time() - start)
    hours = int(total_seconds//(60*60))
    remaining_seconds = int(total_seconds-hours*60*60)
    minutes = remaining_seconds //60
    seconds = int(total_seconds-((hours*60*60)+(minutes*60)))
    print(f"Total Runtime: {hours} hours, {minutes} minutes, and {seconds} seconds")

    return prediction_df.fillna(0).astype(int)


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

@ ignore_warnings (category=RuntimeWarning)
def getDieboldMariano(df):
    actual = df['GDP']
    ar1 = df['AR(1)']
    predictions = df.drop('GDP',axis=1)
    diebold_mariano = {'LASSO':[],'Ridge':[],'Elastic Net':[],'Gradient Boosting':[],'Neural Net':[],'SVM':[],'Model Avg.':[]}
    for col in predictions:
        test_stat, p_value = dm_test(actual,ar1,predictions[col])
        diebold_mariano[col] = [p_value,test_stat]
    diebold_mariano_df = pd.DataFrame(diebold_mariano,index=['p-value','DM Test Stat']).T
    diebold_mariano_df = diebold_mariano_df.sort_values(by=['p-value','DM Test Stat'])
    return diebold_mariano_df.dropna()


if __name__ == '__main__':
    # Get predictions
    df = rollingWindow()
    # Display predictions
    print('\n','PREDICTIONS')
    print(df)
    df.to_csv('./Results/predictions_sam.csv')
    # Calculate root-mean squared error and mean absolute error of predictions
    print('\n','ROOT MEAT-SQUARED ERROR and MEAN ABSOLUTE PERCENTAGE ERROR')
    rmse = findRMSE(df)
    print(rmse)
    rmse.to_csv('./Results/rmse_sam.csv')
    # Perform the Diebold-Mariano Test to statistically identify forecast accuracy equivalence
    print('\n','DIEBOLD-MARIANO TEST RESULTS')
    diebold_mariano_results = getDieboldMariano(df)
    print(diebold_mariano_results)
    diebold_mariano_results.to_csv('./Results/diebold_mariano_sam.csv')