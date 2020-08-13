#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:20:49 2020

@author: raghu

"""
import os
from datetime import datetime

# Log building function
def log(logType, logData):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    dirPath = os.path.join(fileDir, 'logs')
    
    if logType == 'error':
        fileName = os.path.join(dirPath, 'ErrorLog.txt')
        nameHandle = open(fileName, 'a')
        nameHandle.write(str(datetime.now())+'\n'+logData+'\n\n')
        nameHandle.close()
    if logType == 'process':
        fileName = os.path.join(dirPath, 'ProcessLog.txt')
        nameHandle = open(fileName, 'a')
        nameHandle.write(str(datetime.now())+'\n'+logData+'\n\n')
        nameHandle.close()
    if logType == 'retrain':
        fileName = os.path.join(dirPath, 'RetrainLog.txt')
        nameHandle = open(fileName, 'a')
        nameHandle.write(str(datetime.now())+','+logData+'\n')
        nameHandle.close()
log('process', 'Implemented a function to log process, errors and retraining data')     

try:
    import pandas as pd
    import numpy as np
    import pymongo
    import csv
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    
    import time
    import pickle
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
    from sklearn.linear_model import LinearRegression, BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import LinearSVR
    from sklearn.ensemble import RandomForestRegressor
    
    import streamlit as st
    import sys
    log('process', 'Imported necessary packages')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Retrain check function       
def needsRetrain(data):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    dirPath = os.path.join(fileDir, 'logs')
    fileName = os.path.join(dirPath, 'RetrainLog.txt')
    status = os.path.isfile(fileName)
    if status:
        nameHandle = open(fileName, 'r')
        means = nameHandle.read().splitlines()[-1].split(',')[1:]
        means = list(map(float, means))
        newMeansStore = ''
        newMeans = []
        for column in data.columns.values: 
            currentMean = data[column].mean()
            newMeans.append(currentMean)
            newMeansStore += str(currentMean)+','
        newMeansStore = newMeansStore[:-1]
        if (means == newMeans):
            nameHandle.close()
            return False
        else:
            log('retrain', newMeansStore)
            nameHandle.close()
            return True
        
    else:
        means = ''
        for column in data.columns.values:        
            means += str(data[column].mean())+','
        means = means[:-1]
        log('retrain', means)
        return True
log('process', 'Implemented a function to check if retraining is needed')     

# Database ingestion
try:
    #we can just use MongoClient() without any details inside to connect to default IP and port
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017')

    #creating 'QSAR' database
    #if a database already exists with this name it connects to it else creates it
    mydb = client['QSAR']

    #creating a collection, it is synonymous to table in mysql
    information = mydb.fishinformation
    
    #dropping collection as a safety measure in case it already exists and has some data in it
    information.drop()  
    
    #reading CSV file content
    reader = csv.DictReader(open('qsar_fish_toxicity.csv'))

    #ingesting reader object content to MongoDB database
    for raw in reader:
        information.insert_one(raw)
    log('process', 'Database ingestion completed')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Loading data into DataFrame
try:
    #reading 'fishinformation' collection from 'QSAR' database and loading data into a dataframe
    client = pymongo.MongoClient()
    db = client.QSAR
    collection = db.fishinformation
    data = pd.DataFrame(list(collection.find()))
    log('process', 'Successfully loaded data from database to dataframe')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))


#dropping _id column 
try:
    data.drop(columns = ['_id'], inplace = True)
    log('process', 'Removed database id label from dataframe')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Typecasting data types for features
try:
    data['CIC0'] = data['CIC0'].astype(float)
    data['SM1_Dz(Z)'] = data['SM1_Dz(Z)'].astype(float)
    data['GATS1i'] = data['GATS1i'].astype(float)
    data['NdsCH'] = data['NdsCH'].astype(int)
    data['NdssC'] = data['NdssC'].astype(int)
    data['MLOGP'] = data['MLOGP'].astype(float)
    data['LC50'] = data['LC50'].astype(float)
    log('process', 'Typecasted data types to int and float from object')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Finding and handling duplicate rows
try:
    data.drop_duplicates(keep='first', inplace=True) 
    log('process', 'Dropping duplicate rows from dataframe')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Finding and handling null values
try:
    totalNumberOfRows = len(data)
    for feature in data.columns.values:
        nullCount = totalNumberOfRows - data[feature].isnull().value_counts()[0]
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        if nullCount > 0:
            imputer.fit(data[feature])
            data[feature] = imputer.transform(data[feature])
    log('process', 'Null values are found and replaced with median value of feature')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Separating discrete and continous features
try:
    categorical = [feature for feature in data.columns.values if len(data[feature].unique()) <= 10]
    continuous = [feature for feature in data.columns.values if feature not in categorical and feature != 'LC50']
    log('process', 'Separated categorical and continuous features')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Detecting and handling outliers
try:
    overallIndeces = set()
    outlierIndeces = {}
    
    for feature in data[continuous].columns.values:
        q1 = np.percentile(data[feature], 25)
        q3 = np.percentile(data[feature], 75)
        iqr = q3 - q1
        outliers = data.loc[(data[feature] < q1 - 1.5*iqr) | (data[feature] > q3 + 1.5*iqr)]
        outlierIndeces[feature] = outliers.index.values
        overallIndeces.update(outliers.index.values)
    
    if len(outlierIndeces) > 0:
        data.drop(overallIndeces, inplace = True)
    log('process', 'Detected and handled outliers by univariate analysis')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Separating target feature
try:
    target = data['LC50']
    data.drop(columns = ['LC50'], inplace = True)
    log('process', 'Separated dependent feature')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Constant features detection and elimination
try:
    feature_selector = VarianceThreshold(threshold=0)
    feature_selector.fit(data)
    feature_selector.get_support()
    constantFeatures = [x for x in data.columns if x not in data.columns[feature_selector.get_support()]]
    log('process', 'Constant features detected and eliminated')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Quasi Constant features detection and elimination
try:
    feature_selector = VarianceThreshold(threshold=0.01)
    feature_selector.fit(data)
    quasiConstant = [x for x in data.columns if x not in data.columns[feature_selector.get_support()]]
    log('process', 'Quasi constant features detected and eliminated')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Duplicate features detection and elimination
try:
    data.T.duplicated().sum()
    duplicatedFeatures = data.T[data.T.duplicated()].index.values
    log('process', 'Duplicate features detected and eliminated')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Correlated features detection and elimination
try:
    def correlation(data, threshold=None):
        # Set of all names of correlated columns
        col_corr = set()
        corr_mat = data.corr()
        for i in range(len(corr_mat.columns)):
            for j in range(i):
                if (abs(corr_mat.iloc[i,j]) > threshold):
                    colname = corr_mat.columns[i]
                    col_corr.add(colname)
        return col_corr
    correlatedFeatures = correlation(data=data, threshold=0.8)
    log('process', 'Correlated set of features detected and best feature retained')
    
    # Splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.20, random_state=355)
    log('process', 'Train and test data separated to prevent data leakage')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Scaling train data
try:
    sc = MinMaxScaler()
    sc.fit(x_train[continuous])
    x_train[continuous] = sc.transform(x_train[continuous])
    log('process', 'Train data scaling performed')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Normalising train data
try:
    skewness = data[continuous].skew() > 0.5
    skewedFeatures = skewness[skewness == True].index
    x_train[skewedFeatures] = np.sqrt(x_train[skewedFeatures])
    log('process', 'Train data\'s skewed features normalised')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Dimentionality reduction using PCA on train data
try:
    pca = PCA()
    principalComponents = pca.fit_transform(x_train)
    log('process', 'Beginning dimensionality reduction')
    
    #selecting the number of components that explain more than 99% variance of data
    components = 0
    maxVariance = 99
    
    for index in range(len(np.cumsum(pca.explained_variance_ratio_))):
        if np.cumsum(pca.explained_variance_ratio_)[index] >= (maxVariance/100):
            components = index + 1
            break
    log('process', 'Chosing number of dimensions covering '+str(maxVariance)+'% variance')
    
    #generating PCA data frame
    pca = PCA(n_components=components)
    pca.fit(x_train)
    new_train_data = pca.transform(x_train)
    columns = ['PC '+str(x) for x in range(1, components+1)]
    # This will be the new data fed to the algorithm.
    x_train = pd.DataFrame(data = new_train_data, columns = columns)
    log('process', 'Generated input data with dimensionality reduction performed')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Scaling test data
try:
    sc.fit(x_test[continuous])
    x_test[continuous] = sc.transform(x_test[continuous])
    log('process', 'Test data scaling performed')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Normalising test data
try:
    x_test[skewedFeatures] = np.sqrt(x_test[skewedFeatures])
    log('process', 'Test data\'s skewed features normalised')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Applying dimentionality reduction using PCA on test data
try:
    new_test_data = pca.transform(x_test)
    columns = ['PC '+str(x) for x in range(1, components+1)]
    x_test = pd.DataFrame(data = new_test_data, columns = columns)
    log('process', 'Applying dimensionality reduction on test data using previously built model')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

#Training the model
try:
    if needsRetrain(data):
        log('process', 'Retraining model')
        # Building model and hyperparameter turning using GridSearch
        datasets = {}
        datasets[0] = {'X_train': x_train, 
                       'X_test' : x_test,
                       'y_train': y_train, 
                       'y_test' : y_test}
        
        
        def train_test_linear_regression(X_train,
                                         X_test,
                                         y_train,
                                         y_test,
                                         cv_count,
                                         scorer,
                                         dataset_id):
            linear_regression = LinearRegression()
            grid_parameters_linear_regression = {'fit_intercept' : [False, True]}
            start_time = time.time()
            grid_obj = GridSearchCV(linear_regression,
                                    param_grid=grid_parameters_linear_regression,
                                    cv=cv_count,
                                    n_jobs=-1,
                                    scoring=scorer,
                                    verbose=2)
            grid_fit = grid_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_linear_regression = grid_fit.best_estimator_
            infStartTime = time.time()
            prediction = best_linear_regression.predict(X_test)
            prediction_time = time.time() - infStartTime 
            r2 = r2_score(y_true=y_test, y_pred=prediction)
            mse = mean_squared_error(y_true=y_test, y_pred=prediction)
            mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
            medae = median_absolute_error(y_true=y_test, y_pred=prediction)
            
        
        
            # metrics for true values
            # r2 remains unchanged, mse, mea will change and cannot be scaled
            # because there is some physical meaning behind it
            if 1==1:
                prediction_true_scale = prediction
                prediction = prediction
                y_test_true_scale = y_test
                mae_true_scale = mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                medae_true_scale = median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                mse_true_scale = mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
        
            return {'Regression type' : 'Linear Regression',
                    'model' : grid_fit,
                    'Predictions' : prediction,
                    'R2' : r2,
                    'MSE' : mse,
                    'MAE' : mae,
                    'MSE_true_scale' : mse_true_scale,
                    'RMSE_true_scale' : np.sqrt(mse_true_scale),
                    'MAE_true_scale' : mae_true_scale,
                    'MedAE_true_scale' : medae_true_scale,
                    'Training time' : training_time,
                    'Prediction time' : prediction_time,
                    'dataset' : dataset_id}
        
        def train_test_bRidge_regression(X_train,
                                         X_test,
                                         y_train,
                                         y_test,
                                         cv_count,
                                         scorer,
                                         dataset_id):
            bRidge_regression = BayesianRidge()
            grid_parameters_BayesianRidge_regression = {'fit_intercept' : [False, True], 
                                                        'n_iter':[300,1000,5000]}
            start_time = time.time()
            grid_obj = GridSearchCV(bRidge_regression,
                                    param_grid=grid_parameters_BayesianRidge_regression,
                                    cv=cv_count,
                                    n_jobs=-1,
                                    scoring=scorer,
                                    verbose=2)
            grid_fit = grid_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_linear_regression = grid_fit.best_estimator_
            infStartTime = time.time()
            prediction = best_linear_regression.predict(X_test)
            prediction_time = time.time() - infStartTime
            r2 = r2_score(y_true=y_test, y_pred=prediction)
            mse = mean_squared_error(y_true=y_test, y_pred=prediction)
            mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
            medae = median_absolute_error(y_true=y_test, y_pred=prediction)
            
        
            if 1==1:
                prediction_true_scale = prediction
                prediction = prediction
                y_test_true_scale = y_test
                mae_true_scale = mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                medae_true_scale = median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                mse_true_scale = mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
        
            return {'Regression type' : 'Bayesian Ridge Regression',
                    'model' : grid_fit,
                    'Predictions' : prediction,
                    'R2' : r2,
                    'MSE' : mse,
                    'MAE' : mae,
                    'MSE_true_scale' : mse_true_scale,
                    'RMSE_true_scale' : np.sqrt(mse_true_scale),
                    'MAE_true_scale' : mae_true_scale,
                    'MedAE_true_scale' : medae_true_scale,
                    'Training time' : training_time,
                    'Prediction time' : prediction_time,
                    'dataset' : dataset_id}
        
        
        def train_test_decision_tree_regression(X_train,
                                                X_test,
                                                y_train,
                                                y_test,
                                                cv_count,
                                                scorer,
                                                dataset_id):
            decision_tree_regression = DecisionTreeRegressor(random_state=42)
            grid_parameters_decision_tree_regression = {'max_depth' : [None, 3,5,7,9,10,11]}
            start_time = time.time()
            grid_obj = GridSearchCV(decision_tree_regression,
                                    param_grid=grid_parameters_decision_tree_regression,
                                    cv=cv_count,
                                    n_jobs=-1,
                                    scoring=scorer,
                                    verbose=2)
            grid_fit = grid_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_linear_regression = grid_fit.best_estimator_
            infStartTime = time.time()
            prediction = best_linear_regression.predict(X_test)
            prediction_time = time.time() - infStartTime
            r2 = r2_score(y_true=y_test, y_pred=prediction)
            mse = mean_squared_error(y_true=y_test, y_pred=prediction)
            mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
            medae = median_absolute_error(y_true=y_test, y_pred=prediction)
            
        
            if 1==1:
                prediction_true_scale = prediction
                prediction = prediction
                y_test_true_scale = y_test
                mae_true_scale = mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                medae_true_scale = median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                mse_true_scale = mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
        
            return {'Regression type' : 'Decision Tree Regression',
                    'model' : grid_fit,
                    'Predictions' : prediction,
                    'R2' : r2,
                    'MSE' : mse,
                    'MAE' : mae,
                    'MSE_true_scale' : mse_true_scale,
                    'RMSE_true_scale' : np.sqrt(mse_true_scale),
                    'MAE_true_scale' : mae_true_scale,
                    'MedAE_true_scale' : medae_true_scale,
                    'Training time' : training_time,
                    'Prediction time' : prediction_time,
                    'dataset' : dataset_id}
        
        def train_test_knn_regression(X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      cv_count,
                                      scorer,
                                      dataset_id):
            knn_regression = KNeighborsRegressor()
            grid_parameters_knn_regression = {'n_neighbors' : [1,2,3],
                                              'weights': ['uniform', 'distance'],
                                              'algorithm': ['ball_tree', 'kd_tree'],
                                              'leaf_size': [30,90,100,110],
                                              'p': [1,2]}
            start_time = time.time()
            grid_obj = GridSearchCV(knn_regression,
                                    param_grid=grid_parameters_knn_regression,
                                    cv=cv_count,
                                    n_jobs=-1,
                                    scoring=scorer,
                                    verbose=2)
            grid_fit = grid_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_linear_regression = grid_fit.best_estimator_
            infStartTime = time.time()
            prediction = best_linear_regression.predict(X_test)
            prediction_time = time.time() - infStartTime
            r2 = r2_score(y_true=y_test, y_pred=prediction)
            mse = mean_squared_error(y_true=y_test, y_pred=prediction)
            mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
            medae = median_absolute_error(y_true=y_test, y_pred=prediction)
            
            # metrics for true values
            # r2 remains unchanged, mse, mea will change and cannot be scaled
            # because there is some physical meaning behind it
            if 1==1:
                prediction_true_scale = prediction
                prediction = prediction
                y_test_true_scale = y_test
                mae_true_scale = mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                medae_true_scale = median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                mse_true_scale = mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
        
            return {'Regression type' : 'KNN Regression',
                    'model' : grid_fit,
                    'Predictions' : prediction,
                    'R2' : r2,
                    'MSE' : mse,
                    'MAE' : mae,
                    'MSE_true_scale' : mse_true_scale,
                    'RMSE_true_scale' : np.sqrt(mse_true_scale),
                    'MAE_true_scale' : mae_true_scale,
                    'MedAE_true_scale' : medae_true_scale,
                    'Training time' : training_time,
                    'Prediction time' : prediction_time,
                    'dataset' : dataset_id}
        
        def train_test_SVR_regression(X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      cv_count,
                                      scorer,
                                      dataset_id):
            SVR_regression = LinearSVR()
            grid_parameters_SVR_regression = {'C' : [1, 10, 50],
                                             'epsilon' : [0.01, 0.1],
                                             'fit_intercept' : [False, True]}
            start_time = time.time()
            grid_obj = GridSearchCV(SVR_regression,
                                    param_grid=grid_parameters_SVR_regression,
                                    cv=cv_count,
                                    n_jobs=-1,
                                    scoring=scorer,
                                    verbose=2)
            grid_fit = grid_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_linear_regression = grid_fit.best_estimator_
            infStarTime = time.time()
            prediction = best_linear_regression.predict(X_test)
            prediction_time = time.time() - infStarTime
            r2 = r2_score(y_true=y_test, y_pred=prediction)
            mse = mean_squared_error(y_true=y_test, y_pred=prediction)
            mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
            medae = median_absolute_error(y_true=y_test, y_pred=prediction)
            
            # metrics for true values
            # r2 remains unchanged, mse, mea will change and cannot be scaled
            # because there is some physical meaning behind it
            if 1==1:
                prediction_true_scale = prediction
                prediction = prediction
                y_test_true_scale = y_test
                mae_true_scale = mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                medae_true_scale = median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                mse_true_scale = mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
        
            return {'Regression type' : 'Linear SVM Regression',
                    'model' : grid_fit,
                    'Predictions' : prediction,
                    'R2' : r2,
                    'MSE' : mse,
                    'MAE' : mae,
                    'MSE_true_scale' : mse_true_scale,
                    'RMSE_true_scale' : np.sqrt(mse_true_scale),
                    'MAE_true_scale' : mae_true_scale,
                    'MedAE_true_scale' : medae_true_scale,
                    'Training time' : training_time,
                    'Prediction time' : prediction_time,
                    'dataset' : dataset_id}
        
        
        def train_test_random_forest_regression(X_train,
                                                X_test,
                                                y_train,
                                                y_test,
                                                cv_count,
                                                scorer,
                                                dataset_id):
            random_forest_regression = RandomForestRegressor(random_state=42)
            grid_parameters_random_forest_regression = {'n_estimators' : [3,5,10,15,18],
                                             'max_depth' : [None, 2,3,5,7,9]}
            start_time = time.time()
            grid_obj = GridSearchCV(random_forest_regression,
                                    param_grid=grid_parameters_random_forest_regression,
                                    cv=cv_count,
                                    n_jobs=-1,
                                    scoring=scorer,
                                    verbose=2)
            grid_fit = grid_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_linear_regression = grid_fit.best_estimator_
            infStartTime = time.time()
            prediction = best_linear_regression.predict(X_test)
            prediction_time = time.time() - infStartTime
            r2 = r2_score(y_true=y_test, y_pred=prediction)
            mse = mean_squared_error(y_true=y_test, y_pred=prediction)
            mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
            medae = median_absolute_error(y_true=y_test, y_pred=prediction)
            
            # metrics for true values
            # r2 remains unchanged, mse, mea will change and cannot be scaled
            # because there is some physical meaning behind it
            if 1==1:
                prediction_true_scale = prediction
                prediction = prediction
                y_test_true_scale = y_test
                mae_true_scale = mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                medae_true_scale = median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
                mse_true_scale = mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
        
            return {'Regression type' : 'Random Forest Regression',
                    'model' : grid_fit,
                    'Predictions' : prediction,
                    'R2' : r2,
                    'MSE' : mse,
                    'MAE' : mae,
                    'MSE_true_scale' : mse_true_scale,
                    'RMSE_true_scale' : np.sqrt(mse_true_scale),
                    'MAE_true_scale' : mae_true_scale,
                    'MedAE_true_scale' : medae_true_scale,
                    'Training time' : training_time,            
                    'Prediction time' : prediction_time,
                    'dataset' : dataset_id}
        
        # make scorer
        scorer = 'neg_mean_squared_error'
        results = {}
        counter = 0
        cv_count = 5
        
        for dataset in [0]:
            X_train, X_test, y_train, y_test = datasets[dataset]['X_train'], datasets[dataset]['X_test'], datasets[dataset]['y_train'], datasets[dataset]['y_test']
            results[counter] = train_test_linear_regression(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            cv_count,
                                                            scorer,
                                                            dataset)
            print("Linear Regression completed")
            counter += 1
            results[counter] = train_test_bRidge_regression(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            cv_count,
                                                            scorer,
                                                            dataset)
            print("Bayesian Ridge Regression completed")
            counter += 1
            results[counter] = train_test_decision_tree_regression(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            cv_count,
                                                            scorer,
                                                            dataset)
            print("Decision Trees completed")
            counter += 1
            results[counter] = train_test_knn_regression(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            cv_count,
                                                            scorer,
                                                            dataset)
            print("KNN completed")
            counter += 1
            results[counter] = train_test_SVR_regression(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            cv_count,
                                                            scorer,
                                                            dataset)
            print("SVR completed")
            counter += 1
            results[counter] = train_test_random_forest_regression(X_train,
                                                            X_test,
                                                            y_train,
                                                            y_test,
                                                            cv_count,
                                                            scorer,
                                                            dataset)
            print("Random Forest completed")
            counter += 1
        log('process', 'Model retraining completed')
        results_df = pd.DataFrame.from_dict(results, orient='index')
        # saving the model to the local file system
        regression = results_df.sort_values(by=['R2'], ascending = False).iloc[0,1]
        log('process', 'Model with highest R2 score selected as best model')
        filename = 'finalized_model.pickle'
        pickle.dump(regression, open(filename, 'wb'))
        log('process', 'Saved best model as pickle file')
    else:
        #predict using existing model
        log('process', 'Using saved model')       
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))




# Predict dependent feature

#loading saved model
try:
    filename = 'finalized_model.pickle'
    loaded_model = pickle.load(open(filename, 'rb'))
    log('process', 'Saved model loaded for prediction')
except:
    log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

#building web app and predicting values
    
def main():
    try:
        st.set_option('deprecation.showfileUploaderEncoding', False)
        #Building sidebar of web app
        log('process', 'Building sidebar of web app')
        st.write("""
        # LC50 predictor
        ***Lethal concentration 50*** (**LC50**) is the amount of a substance suspended
         in the air required to kills 50% of a test animals during a predetermined
         observation period. **LC50** values are frequently used as a general indicator
         of a substance's acute toxicity.
        """)
        
        st.sidebar.header('User Input Parameters')
        
        # Collects user input features into dataframe
        log('process', 'Building user input via batch upload or individual values in sidebar')
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV or Excel file", type=['csv', 'xls', 'xlsx'])
        st.sidebar.markdown("<p style='text-align: center; color: black;'>-- or --</p>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            def try_read_df(f):
                try:
                    return pd.read_csv(f)
                except:
                    return pd.read_excel(f)
            input_df = try_read_df(uploaded_file)
            #input_df = pd.read_csv(uploaded_file)
        else:
            def user_input_features(data):
                feature1 = st.sidebar.number_input("CIC0", value = 2.94)
                feature2 = st.sidebar.number_input("SM1_Dz(Z)", value = 0.56)
                feature3 = st.sidebar.number_input("GATS1i", value = 1.23)
                feature4 = st.sidebar.slider('NdsCH',  0, 4, 0, step = 1)
                feature5 = st.sidebar.slider('NdssC', 0, 6, 0, step = 1)
                feature6 = st.sidebar.number_input("MLOGP", value = 2.13)
                
                dataIn = {'CIC0': feature1,
                        'SM1_Dz(Z)': feature2,
                        'GATS1i': feature3,
                        'NdsCH': feature4,
                        'NdssC': feature5,
                        'MLOGP': feature6}
                features = pd.DataFrame(dataIn, index=[0])
                return features
            input_df = user_input_features(data)
        
        inputVals = input_df
        toPredict = inputVals.copy()
        
        log('process', 'Scaling, transforming and reducing dimensions for input values')
        #scale
        toPredict[continuous] = sc.transform(toPredict[continuous])
        #transform
        toPredict[skewedFeatures] = np.sqrt(toPredict[skewedFeatures])
        #dimensionality reduction
        toPredict = pca.transform(toPredict)
        #predict
        log('process', 'Predicting LC50 values')
        a = loaded_model.predict(toPredict)
        inputVals['LC50'] = a
        
        st.subheader('User Input parameters and predictions')
        log('process', 'Displaying input and predicted values')
        st.write(inputVals)
        
        nameHandle = open("logs/ProcessLog.txt", 'r')
        #st.write(*nameHandle.readlines())
        pLog = ''
        for line in nameHandle.readlines():
            pLog += line
        
        st.subheader('Process Log')
        pLog = '<div style="height:300px;width:700px;border:1px solid #ccc;font:16px/26px Georgia, Garamond, Serif;overflow:auto;">'+pLog+'</div>'
        st.markdown(pLog, unsafe_allow_html=True)
    except:
        log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

if __name__=='__main__':
    main()