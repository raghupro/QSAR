#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 20:12:47 2020

@author: raghu
"""

import sys
import os
from sklearn.feature_selection import VarianceThreshold

fileDir = os.path.dirname(__file__)
dirPath = os.path.abspath(os.path.join(fileDir, '..'))
sys.path.insert(0, dirPath)
from logBuilder.logger import AppLogger

updatelog = AppLogger()

class featureSel(object):
    def constantFeatureElimination(self, dataFrame):
        try:
            feature_selector = VarianceThreshold(threshold=0)
            feature_selector.fit(dataFrame)
            feature_selector.get_support()
            constantFeatures = [x for x in dataFrame.columns if x not in dataFrame.columns[feature_selector.get_support()]]
            dataFrame.drop(columns = constantFeatures, inplace = True)
            updatelog.log('process', 'Constant features detected and eliminated')
            return dataFrame
        except:
            updatelog.log('error', "Error deleting constant features:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
    def quasiConstantFeatureElimination(self, dataFrame):
        try:
            feature_selector = VarianceThreshold(threshold=0.01)
            feature_selector.fit(dataFrame)
            quasiConstant = [x for x in dataFrame.columns if x not in dataFrame.columns[feature_selector.get_support()]]
            dataFrame.drop(columns = quasiConstant, inplace = True)
            updatelog.log('process', 'Quasi constant features detected and eliminated')
            return dataFrame
        except:
            updatelog.log('error', "Error deleting quasi constant features:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

    def duplicateFeaturesElimination(self, dataFrame):
        try:
            dataFrame.T.duplicated().sum()
            duplicatedFeatures = dataFrame.T[dataFrame.T.duplicated()].index.values
            dataFrame.drop(columns = duplicatedFeatures, inplace = True)
            updatelog.log('process', 'Duplicate features detected and eliminated')
            return dataFrame
        except:
            updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def correlatedFeatureElimination(self, dataFrame):
        try:
            def correlation(data, threshold=None):
                # Set of all names of correlated columns
                col_corr = set()
                corr_mat = dataFrame.corr()
                for i in range(len(corr_mat.columns)):
                    for j in range(i):
                        if (abs(corr_mat.iloc[i,j]) > threshold):
                            colname = corr_mat.columns[i]
                            col_corr.add(colname)
                return col_corr
            correlatedFeatures = correlation(data=dataFrame, threshold=0.8)
            if len(correlatedFeatures) > 1 :
                #enter code for correlated feature handling
                pass
            updatelog.log('process', 'Correlated set of features detected and best feature retained')
            return dataFrame
        except:
            updatelog.log('error', "Unexpected error:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
            
            
            
            