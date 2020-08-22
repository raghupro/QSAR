#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:55:52 2020

@author: raghu
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
from datetime import date

from logBuilder.logger import AppLogger
#import sys
#del sys.modules['logBuilder']

updatelog = AppLogger()

fileDir = os.path.dirname(os.path.realpath('__file__'))
modelsPath = os.path.join(fileDir, 'models')


def logFolderPath():
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    dirPath = os.path.join(fileDir, 'logs')
    return dirPath
    


        
def updateCount():
    fileName = os.path.join(logFolderPath(), 'CountLog.txt')
    countDf = pd.read_csv(fileName, header = None)
    countDf.columns = ['date', 'visits']
    
    lastDate = countDf.iloc[-1,0]
    lastCount = countDf.iloc[-1,1]
    
    if lastDate == str(date.today()):
        lastCount += 1
        countDf.iloc[-1,1] = lastCount
    else:
        countDf.iloc[0,1] = countDf.iloc[0,1] + lastCount
        countDf.iloc[1:-1,:] = countDf.iloc[2:,:].values
        countDf.iloc[-1,0] = str(date.today())
        countDf.iloc[-1,1] = 1
    countDf.to_csv(fileName, header = False, index = False)

def showChart(choice):
    if choice == 'Visitor count':
        fileName = os.path.join(logFolderPath(), 'CountLog.txt')
        
        countDf = pd.read_csv(fileName, header = None)
        countDf.columns = ['date', 'visits']
        totalPredictions = countDf.iloc[0,1] +countDf.iloc[-1,1]
        
        st.write('Total number of visits so far: %s' %totalPredictions)
        plt.bar(countDf.iloc[1:,0].values, countDf.iloc[1:,1].values)
        plt.xticks(rotation = 90)
        plt.xlabel('Date')
        plt.ylabel('Number of predictions')
        plt.title('Last 10 days count of predictions made')
        st.pyplot()
    elif choice == 'Training times':
        fileName = os.path.join(logFolderPath(), 'TrainTimeLog.txt')
        
        logs = pd.read_csv(fileName, header = None)
        logs.columns = ['type', 'time']
        trainTimes = logs[logs['type'] == 'train'].tail(10)['time'].values
        retrainTimes = logs[logs['type'] == 'retraining'].tail(10)['time'].values
        xTrain = list(range(len(trainTimes)))
        xRetrain = list(range(len(retrainTimes)))
        
        plt.plot(xTrain, trainTimes, label = 'train')
        plt.plot(xRetrain, retrainTimes, label = 'Re-train')
        plt.legend()
        plt.xlabel('Training ID')
        plt.ylabel('Time in seconds')
        plt.title('Last 10 training and retraining times')
        st.pyplot()
    elif choice == 'Predictions count':
        fileName = os.path.join(logFolderPath(), 'PredictLog.txt')
        maxPlotDuration = 10
        
        logs = pd.read_csv(fileName, header = None)
        logs.columns = ['date', 'count']
        logs['date'] = pd.to_datetime(logs['date'])
        totalPredictions = logs['count'].sum()
        
        results = logs.groupby('date').sum().sort_values('date')
        if len(results > 10):
            results = results.iloc[-maxPlotDuration:,:]
        results.index = results.index.date
        results.index = [str(val) for val in results.index]
        
        plt.bar(results.index,  results['count'].values)
        plt.xticks(rotation = 90)
        plt.xlabel('Date')
        plt.ylabel('Number of predictions')
        plt.title('Last 10 days count of predictions made')
        st.write('Total predictions made so far %s'%totalPredictions)
        st.pyplot()

def showLog(choice):
    if choice == 'Process log':
        nameHandle = open("logs/ProcessLog.txt", 'r')
        pLog = ''
        for line in nameHandle.readlines():
            pLog += line + '<br>'
        nameHandle.close()
        pLog = '<div style="height:300px;width:700px;border:1px solid #ccc;font:16px/26px Georgia, Garamond, Serif;overflow:auto;">'+pLog+'</div>'
        return pLog
    elif choice == 'Error log':
        nameHandle = open("logs/ErrorLog.txt", 'r')
        eLog = ''
        for line in nameHandle.readlines():
            eLog += line + '<br>'
        nameHandle.close()
        eLog = '<div style="height:300px;width:700px;border:1px solid #ccc;font:16px/26px Georgia, Garamond, Serif;overflow:auto;">'+eLog+'</div>'
        return eLog
    elif choice == 'Retrain log':
        nameHandle = open("logs/RetrainLog.txt", 'r')
        rLog = ''
        for line in nameHandle.readlines():
            rLog += line + '<br><br>'
        nameHandle.close()
        rLog = '<div style="height:300px;width:700px;border:1px solid #ccc;font:16px/26px Georgia, Garamond, Serif;overflow:auto;">'+rLog+'</div>'
        return rLog   

def readDf(f):
    try:
        return pd.read_csv(f)
    except:
        return pd.read_excel(f)
            
def main():

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Building sidebar of web app
    st.write("""
    # LC50 predictor
    ***Lethal concentration 50*** (**LC50**) is the amount of a substance suspended
     in the water required to kills 50% of a test fish during a predetermined
     observation period. **LC50** values are frequently used as a general indicator
     of a substance's acute toxicity.
    """)

    st.sidebar.header('User Input Parameters')
    
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV or Excel file", type=['csv', 'xls', 'xlsx'], key = 'predict')
    st.sidebar.markdown("<p style='text-align: center; color: black;'>-- or --</p>", unsafe_allow_html=True)

    if uploaded_file is not None:
        input_df = readDf(uploaded_file)
        if len(input_df.columns.values) != 6:
            st.subheader('Input file does not have 6 features, please try again')
            return None
        elif 'object' in input_df.dtypes.values:
            st.subheader('Input file has non-numeric values, please try again.')
            return None
    else:
        def user_input_features():
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
        input_df = user_input_features()
    updatelog.log('predict', str(len(input_df)))    
    inputVals = input_df
    toPredict = inputVals.copy()

    continuous = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'MLOGP']
    skewedFeatures = ['SM1_Dz(Z)', 'GATS1i']
    #scale
    scalePicklePath = os.path.join(modelsPath, 'scale.pickle')
    scaleSaved = pickle.load(open(scalePicklePath, 'rb'))
    toPredict[continuous] = scaleSaved.transform(toPredict[continuous])
    #transform
    toPredict[skewedFeatures] = np.sqrt(toPredict[skewedFeatures])
    #dimensionality reduction
    pcaPicklePath = os.path.join(modelsPath, 'pca.pickle')
    pca = pickle.load(open(pcaPicklePath, 'rb'))
    toPredict = pca.transform(toPredict)
    #predict
    modelPicklePath = os.path.join(modelsPath, 'finalized_model.pickle')
    modelSaved = pickle.load(open(modelPicklePath, 'rb'))
    a = modelSaved.predict(toPredict)
    inputVals['LC50'] = a

    st.subheader('User Input parameters and predictions')
    st.write(inputVals)
    
    ###### retrain file upload
    st.subheader('Upload retrain file')
    retrain_file = st.file_uploader("Upload your input CSV or Excel file", type=['csv', 'xls', 'xlsx'], key='retrain')
    if retrain_file:
        inputDf = readDf(retrain_file)
        if len(inputDf.columns.values) == 7 and 'object' not in inputDf.dtypes.values:    
            inputDf.to_csv('qsar_fish_toxicity.csv', index = False)
            inputDf.drop(columns = ['LC50'], inplace = True)
            st.success('File uploaded successfully')
            os.system('python maincode.py')
            st.success('Model retrained successfully')
        else:
            st.error('Wrong file uploaded, please try again')    
    
    dashboard = st.selectbox('Dashboard Charts', ['Visitor count', 'Training times', 'Predictions count'], 0)
    showChart(dashboard)

    selectLog = st.selectbox('Select log', ['Select log', 'Process log', 'Error log', 'Retrain log'], 0)
    finalLog = showLog(selectLog)
    if selectLog != 'Select log':
        st.subheader('%s' %selectLog)
    if finalLog != None:
        st.markdown(finalLog, unsafe_allow_html=True)

if __name__=='__main__':
    updateCount()
    main()