#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 20:02:24 2020

@author: raghu
"""

import sys
import os
import numpy as np

fileDir = os.path.dirname(__file__)
dirPath = os.path.abspath(os.path.join(fileDir, '..'))
sys.path.insert(0, dirPath)
from logBuilder.logger import AppLogger

updatelog = AppLogger()


class featureEng(object):
    def processOutliers(self, dataFrame, continuous):
        try:
            overallIndeces = set()
            outlierIndeces = {}
        
            for feature in dataFrame[continuous].columns.values:
                q1 = np.percentile(dataFrame[feature], 25)
                q3 = np.percentile(dataFrame[feature], 75)
                iqr = q3 - q1
                outliers = dataFrame.loc[(dataFrame[feature] < q1 - 1.5*iqr) | (dataFrame[feature] > q3 + 1.5*iqr)]
                outlierIndeces[feature] = outliers.index.values
                overallIndeces.update(outliers.index.values)
        
            if len(outlierIndeces) > 0:
                dataFrame.drop(overallIndeces, inplace = True)
            updatelog.log('process', 'Detected and handled outliers by univariate analysis')
            return dataFrame
        except:
            updatelog.log('error', "Error handling outliers:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
        