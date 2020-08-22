#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:02:49 2020

@author: raghu
"""

import sys
import os
import numpy as np
from sklearn.impute import SimpleImputer

fileDir = os.path.dirname(__file__)
dirPath = os.path.abspath(os.path.join(fileDir, '..'))
sys.path.insert(0, dirPath)
from logBuilder.logger import AppLogger

updatelog = AppLogger()

class PreProcess(object):
    
    def dropDummyCols(self, dataFrame):
        try:
            dataFrame.drop(columns = ['_id'], inplace = True)
            updatelog.log('process', 'Removed unnecessary label from dataframe')
            return dataFrame
        except:
            updatelog.log('error', "Error dropping unnecessary columns: "+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def typecastFeatures(self, dataFrame):
        try:
            dataFrame['CIC0'] = dataFrame['CIC0'].astype(float)
            dataFrame['SM1_Dz(Z)'] = dataFrame['SM1_Dz(Z)'].astype(float)
            dataFrame['GATS1i'] = dataFrame['GATS1i'].astype(float)
            dataFrame['NdsCH'] = dataFrame['NdsCH'].astype(int)
            dataFrame['NdssC'] = dataFrame['NdssC'].astype(int)
            dataFrame['MLOGP'] = dataFrame['MLOGP'].astype(float)
            dataFrame['LC50'] = dataFrame['LC50'].astype(float)
            updatelog.log('process', 'Typecasted data types to int and float from object')
            return dataFrame
        except:
            updatelog.log('error', "Error typecasting features: "+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def dropDuplicateRows(self, dataFrame):
        try:
            dataFrame.drop_duplicates(keep='first', inplace=True)
            updatelog.log('process', 'Dropping duplicate rows from dataframe')
            return dataFrame
        except:
            updatelog.log('error', "Error deleting duplicate rows: "+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

    def replaceNullValues(self, dataFrame):
        try:
            totalNumberOfRows = len(dataFrame)
            for feature in dataFrame.columns.values:
                nullCount = totalNumberOfRows - dataFrame[feature].isnull().value_counts()[0]
                imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                if nullCount > 0:
                    imputer.fit(dataFrame[feature])
                    dataFrame[feature] = imputer.transform(dataFrame[feature])
            updatelog.log('process', 'Null values that are found and replaced with median value of feature')
            return dataFrame
        except:
            updatelog.log('error', "Error handling null values: "+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
       
            