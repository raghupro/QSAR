#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:46:04 2020

@author: raghu
"""

import os
from datetime import datetime
from datetime import date

class AppLogger(object):
    def __init__(self):
        pass


    def log(self, logType, logData):
        fileDir = os.path.dirname(__file__)
        dirPath = os.path.abspath(os.path.join(fileDir, '..', 'logs'))
    
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
        if logType == 'train' or logType == 'retraining':
            fileName = os.path.join(dirPath, 'TrainTimeLog.txt')
            nameHandle = open(fileName, 'a')
            nameHandle.write(logType+','+logData + '\n')
            nameHandle.close()
        if logType == 'predict':
            fileName = os.path.join(dirPath, 'PredictLog.txt')
            nameHandle = open(fileName, 'a')
            nameHandle.write(str(date.today())+','+logData + '\n')
            nameHandle.close()
        if logType == 'count':
            fileName = os.path.join(dirPath, 'CountLog.txt')
            nameHandle = open(fileName, 'r')
            count = int(nameHandle.read())
            count += 1
            print(count)
            nameHandle.close()
            nameHandle = open(fileName, 'w')
            nameHandle.write(str(count))
            nameHandle.close()

        
updatelog = AppLogger()
updatelog.log('process', 'Initiated logger')