#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:03:03 2020

@author: raghu
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle


fileDir = os.path.dirname(__file__)
dirPath = os.path.abspath(os.path.join(fileDir, '..'))
sys.path.insert(0, dirPath)
from logBuilder.logger import AppLogger

updatelog = AppLogger()

class Transform(object):
    modelPath = os.path.join(dirPath, 'models')
    def generateScaleModel(self, dataFrame):
        try:
            sc = MinMaxScaler()
            sc.fit(dataFrame)
            dataFrame = sc.transform(dataFrame)
            filename = os.path.join(Transform.modelPath, 'scale.pickle')
            pickle.dump(sc, open(filename, 'wb'))
            updatelog.log('process', 'Scaling model generated')
        except:
            updatelog.log('error', "Error generating scaling model:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def scale(self, dataFrame):
        try:
            filename = os.path.join(Transform.modelPath, 'scale.pickle')
            scaleSaved = pickle.load(open(filename, 'rb'))
            dataFrame = scaleSaved.transform(dataFrame)
            updatelog.log('process', 'Data scaling performed')
            return dataFrame
        except:
            updatelog.log('error', "Error scaling data:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
        
    def getSkewedFeatures(self, dataFrame):
        try:
            skewness = dataFrame.skew() > 0.5
            skewedFeatures = skewness[skewness == True].index
            updatelog.log('process', 'Skewed features generated')
            return skewedFeatures
        except:
            updatelog.log('error', "Error generating skewed features:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            
    def normalise(self, dataFrame):
        try:
            dataFrame = np.sqrt(dataFrame)
            updatelog.log('process', 'Normalisation performed')
            return dataFrame
        except:
            updatelog.log('error', "Error normalising data:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def generatePCAModel(self, dataFrame, maxVariance):
        try:
            filename = os.path.join(Transform.modelPath, 'pca.pickle')
            
            pca = PCA()
            pca.fit_transform(dataFrame)
            components = 0
            for index in range(len(np.cumsum(pca.explained_variance_ratio_))):
                if np.cumsum(pca.explained_variance_ratio_)[index] >= (maxVariance/100):
                    components = index + 1
                    break
            pca = PCA(n_components=components)
            pca.fit(dataFrame)
            pickle.dump(pca, open(filename, 'wb'))
            updatelog.log('process', 'PCA model generated')
        except:
            updatelog.log('error', "Error generating scaling model:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
    
    def transformPCA(self, dataFrame):
        try:
            filename = os.path.join(Transform.modelPath, 'pca.pickle')
            pcaSaved = pickle.load(open(filename, 'rb'))
            dataFrame = pcaSaved.transform(dataFrame)
            components = len(dataFrame[0])
            columns = ['PC '+str(x) for x in range(1, components+1)]
            dataFrame = pd.DataFrame(data = dataFrame, columns = columns)
            updatelog.log('process', 'Data scaling performed')
            return dataFrame
        except:
            updatelog.log('error', "Error scaling data:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            