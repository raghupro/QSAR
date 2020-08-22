#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:20:49 2020

@author: raghu

"""
import os
import sys
from logBuilder.logger import AppLogger
from dataIngestion import loadData
from dataPreProcessing import process
from featureEngineering import featureProcessing
from featureSelection import featureSelecting
from dataTransformation import dataTransform
from modelTraining import model
#del sys.modules['logBuilder']
#del sys.modules['dataIngestion']
#del sys.modules['dataPreProcessing']
#del sys.modules['featureEngineering']
#del sys.modules['featureSelection']
#del sys.modules['dataTransformation']
#del sys.modules['modelTraining']
fileDir = os.path.dirname(__file__)
sys.path.insert(0, fileDir)

updatelog = AppLogger()

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import time
    updatelog.log('process', 'Imported necessary packages')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
masterBegin = time.time()

# Database ingestion
try:
    dataLoad = loadData.Load()
    inputFile = 'qsar_fish_toxicity.csv'
    database = 'QSAR'
    collection = 'fishinformation'
    dataLoad.pushIntoMongoDB(inputFile, database, collection)
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Loading data into DataFrame
try:
    #reading 'fishinformation' collection from 'QSAR' database and loading data into a dataframe
    database = 'QSAR'
    collection = 'fishinformation'
    data = dataLoad.pullFromMongoDB(database, collection)
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))


#dropping _id column
try:
    preProcess = process.PreProcess()
    data = preProcess.dropDummyCols(data)
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Typecasting data types for features
try:
    data = preProcess.typecastFeatures(data)
    updatelog.log('process', 'Typecasted data types to int and float from object')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Finding and handling duplicate rows
try:
    data = preProcess.dropDuplicateRows(data)
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
# Finding and handling null values
try:
    data = preProcess.replaceNullValues(data)
    updatelog.log('process', 'Null values are found and replaced with median value of feature')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Separating discrete and continous features
try:
    categorical = [feature for feature in data.columns.values if len(data[feature].unique()) <= 10]
    continuous = [feature for feature in data.columns.values if feature not in categorical and feature != 'LC50']
    updatelog.log('process', 'Separated categorical and continuous features')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Detecting and handling outliers
try:
    fProcessing = featureProcessing.featureEng()
    data = fProcessing.processOutliers(data, continuous)
    updatelog.log('process', 'Detected and handled outliers by univariate analysis')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Separating target feature
try:
    target = data['LC50']
    data.drop(columns = ['LC50'], inplace = True)
    updatelog.log('process', 'Separated dependent feature')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Constant features detection and elimination
try:
    fSelect = featureSelecting.featureSel()
    data = fSelect.constantFeatureElimination(data)
    updatelog.log('process', 'Constant features detected and eliminated')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Quasi Constant features detection and elimination
try:
    data = fSelect.quasiConstantFeatureElimination(data)
    updatelog.log('process', 'Quasi constant features detected and eliminated')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Duplicate features detection and elimination
try:
    data = fSelect.duplicateFeaturesElimination(data)
    updatelog.log('process', 'Duplicate features detected and eliminated')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Correlated features detection and elimination
try:
    data = fSelect.correlatedFeatureElimination(data)
    updatelog.log('process', 'Correlated set of features detected and best feature retained')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
# Splitting data into train and test sets
try:
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.20, random_state=355)
    updatelog.log('process', 'Train and test data separated to prevent data leakage')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Scaling train data
try:
    transform = dataTransform.Transform()
    transform.generateScaleModel(x_train[continuous])
    updatelog.log('process', 'Scaling model generated')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
try:
    x_train[continuous] = transform.scale(x_train[continuous])
    updatelog.log('process', 'Train data scaling performed')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Normalising train data
try:
    skewedFeatures = transform.getSkewedFeatures(data[continuous])
    x_train[skewedFeatures] = transform.normalise(x_train[skewedFeatures])
    updatelog.log('process', 'Train data\'s skewed features normalised')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Dimentionality reduction using PCA on train data
try:
    transform.generatePCAModel(x_train, 99)
    x_train = transform.transformPCA(x_train)
    updatelog.log('process', 'Generated input data with dimensionality reduction performed')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Scaling test data
try:
    x_test[continuous] = transform.scale(x_test[continuous])
    updatelog.log('process', 'Test data scaling performed')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Normalising test data
try:
    x_test[skewedFeatures] = transform.normalise(x_test[skewedFeatures])
    updatelog.log('process', 'Test data\'s skewed features normalised')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

# Applying dimentionality reduction using PCA on test data
try:
    x_test = transform.transformPCA(x_test)
    updatelog.log('process', 'Applying dimensionality reduction on test data using previously built model')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

#check if retraining is required

try:
    modelProcess = model.ModelProcess()
    checkRetrain = modelProcess.needsRetrain(data)
    updatelog.log('process', 'Checking if retraining is required')
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

#Training the model
try:
    if checkRetrain:
        results = modelProcess.trainModels(['linear', 'bayesian ridge', 'decision tree', 'knn', 'svr', 'random forest'], x_train, x_test, y_train, y_test)
        
        results_df = pd.DataFrame.from_dict(results, orient='index')
        print(results_df[['Regression type', 'R2']])
        modelProcess.saveBestModel(results_df, ['R2'])
        masterEnd = time.time() - masterBegin
        updatelog.log('train', str(round(masterEnd,2)))
    else:
        #predict using existing model
        updatelog.log('process', 'Using saved model')
        masterEnd = time.time() - masterBegin
        updatelog.log('retraining', str(round(masterEnd,2)))
except:
    updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))



