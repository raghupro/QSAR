#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:44:40 2020

@author: raghu
"""

import sys
import os
import numpy as np
import time
import pickle

fileDir = os.path.dirname(__file__)
dirPath = os.path.abspath(os.path.join(fileDir, '..'))
sys.path.insert(0, dirPath)
from logBuilder.logger import AppLogger

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

updatelog = AppLogger()

class ModelProcess(object):
    def needsRetrain(self, dataFrame):
        try:
            updatelog.log('process', 'Checking if retraining is needed or not')
            filePath = os.path.join(dirPath, 'logs')
            fileName = os.path.join(filePath, 'RetrainLog.txt')
            status = os.path.isfile(fileName)
            if status:
                nameHandle = open(fileName, 'r')
                means = nameHandle.read().splitlines()[-1].split(',')[1:]
                means = list(map(float, means))
                newMeansStore = ''
                newMeans = []
                for column in dataFrame.columns.values:
                    currentMean = dataFrame[column].mean()
                    newMeans.append(currentMean)
                    newMeansStore += str(currentMean)+','
                newMeansStore = newMeansStore[:-1]
                if (means == newMeans):
                    nameHandle.close()
                    return False
                else:
                    updatelog.log('retrain', newMeansStore)
                    nameHandle.close()
                    return True
        
            else:
                means = ''
                for column in dataFrame.columns.values:
                    means += str(dataFrame[column].mean())+','
                means = means[:-1]
                updatelog.log('retrain', means)
                return True  
        except:
            updatelog.log('error', "Error checking if retraining is required:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def train_test_linear_regression(X_train, X_test, y_train, y_test, cv_count, scorer, dataset_id):
        try:
            updatelog.log('process', 'Training linear regression')
            linear_regression = LinearRegression()
            grid_parameters_linear_regression = {'fit_intercept' : [False, True]}
            start_time = time.time()
            grid_obj = GridSearchCV(linear_regression,
                                    param_grid=grid_parameters_linear_regression,
                                    cv=cv_count,
                                    n_jobs=2,
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
        except:
            updatelog.log('error', "Error training linear regression:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def train_test_bRidge_regression(X_train, X_test, y_train, y_test, cv_count, scorer, dataset_id):
        try:
            updatelog.log('process', 'Training bayesian ridge regression')
            bRidge_regression = BayesianRidge()
            grid_parameters_BayesianRidge_regression = {'fit_intercept' : [False, True],
                                                        'n_iter':[300,1000,5000]}
            start_time = time.time()
            grid_obj = GridSearchCV(bRidge_regression,
                                    param_grid=grid_parameters_BayesianRidge_regression,
                                    cv=cv_count,
                                    n_jobs=2,
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
        except:
            updatelog.log('error', "Error training bayesian ridge regression:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def train_test_decision_tree_regression(X_train, X_test, y_train, y_test, cv_count, scorer, dataset_id):
        try:
            updatelog.log('process', 'Decision tree regression')
            decision_tree_regression = DecisionTreeRegressor(random_state=42)
            grid_parameters_decision_tree_regression = {'max_depth' : [None, 3,5,7,9,10,11]}
            start_time = time.time()
            grid_obj = GridSearchCV(decision_tree_regression,
                                    param_grid=grid_parameters_decision_tree_regression,
                                    cv=cv_count,
                                    n_jobs=2,
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
        except:
            updatelog.log('error', "Error training decision tree regression:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def train_test_knn_regression(X_train, X_test, y_train, y_test, cv_count, scorer, dataset_id):
        try:
            updatelog.log('process', 'Training KNN regression')
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
                                    n_jobs=2,
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
        except:
            updatelog.log('error', "Error training KNN regression:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def train_test_SVR_regression(X_train, X_test, y_train, y_test, cv_count, scorer, dataset_id):
        try:
            updatelog.log('process', 'Training support vector regression')
            SVR_regression = LinearSVR()
            grid_parameters_SVR_regression = {'C' : [1, 10, 50],
                                             'epsilon' : [0.01, 0.1],
                                             'fit_intercept' : [False, True]}
            start_time = time.time()
            grid_obj = GridSearchCV(SVR_regression,
                                    param_grid=grid_parameters_SVR_regression,
                                    cv=cv_count,
                                    n_jobs=2,
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
        except:
            updatelog.log('error', "Error training support vector regression:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
    def train_test_random_forest_regression(X_train, X_test, y_train, y_test, cv_count, scorer, dataset_id):
        try:
            updatelog.log('process', 'Training random forest regression')
            random_forest_regression = RandomForestRegressor(random_state=42)
            grid_parameters_random_forest_regression = {'n_estimators' : [3,5,10,15,18],
                                             'max_depth' : [None, 2,3,5,7,9]}
            start_time = time.time()
            grid_obj = GridSearchCV(random_forest_regression,
                                    param_grid=grid_parameters_random_forest_regression,
                                    cv=cv_count,
                                    n_jobs=2,
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
        except:
            updatelog.log('error', "Error training random forest regression:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
         
    def trainModels(self, models, x_train, x_test, y_train, y_test):
        try:
            scorer = 'neg_mean_squared_error'
            cv_count = 5
            counter = 0
            dataset = 0
            results = {}
            for model in models:
                if model == 'linear':
                    results[counter] = ModelProcess.train_test_linear_regression(x_train, x_test, y_train, y_test, cv_count, scorer, dataset)
                    counter += 1
                if model == 'bayesian ridge':
                    results[counter] = ModelProcess.train_test_bRidge_regression(x_train, x_test, y_train, y_test, cv_count, scorer, dataset)
                    counter += 1
                if model == 'decision tree':
                    results[counter] = ModelProcess.train_test_decision_tree_regression(x_train, x_test, y_train, y_test, cv_count, scorer, dataset)
                    counter += 1
                if model == 'knn':
                    results[counter] = ModelProcess.train_test_knn_regression(x_train, x_test, y_train, y_test, cv_count, scorer, dataset)
                    counter += 1
                if model == 'svr':
                    results[counter] = ModelProcess.train_test_SVR_regression(x_train, x_test, y_train, y_test, cv_count, scorer, dataset)
                    counter += 1
                if model == 'random forest':
                    results[counter] = ModelProcess.train_test_random_forest_regression(x_train, x_test, y_train, y_test, cv_count, scorer, dataset)
                    counter += 1
            updatelog.log('process', 'Successfully trained all models')
            return results
        except:
            updatelog.log('error', "Error training models:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
    def saveBestModel(self, dataFrame, metric):
        try:
            updatelog.log('process', 'Successfully saved best model')
            modelPath = os.path.join(dirPath, 'models')
            regression = dataFrame.sort_values(by=metric, ascending = False).iloc[0,1]
            filename = os.path.join(modelPath, 'finalized_model.pickle')
            pickle.dump(regression, open(filename, 'wb'))
        except:
            updatelog.log('error', "Error saving best model:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
            
        
            
            
            
            
            
            
            
            
            