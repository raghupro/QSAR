#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:04:41 2020

@author: raghu
"""
import csv
import pymongo
import pandas as pd
import sys
import os

fileDir = os.path.dirname(__file__)
dirPath = os.path.abspath(os.path.join(fileDir, '..'))
sys.path.insert(0, dirPath)
from logBuilder.logger import AppLogger

updatelog = AppLogger()

class Load(object):
    def __init__(self):
        pass
    
    def pushIntoMongoDB(self, file, database, collection):
        try:
            client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
            mydb = client[database]
            information = mydb[collection]
            information.drop()
            reader = csv.DictReader(open(file))
            for raw in reader:
                information.insert_one(raw)
            updatelog.log('process', 'Database ingestion completed')
        except:
            updatelog.log('error', "Unexpected error while inserting into MongoDB:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))

        
    def pullFromMongoDB(self, database, collection):
        try:
            client = pymongo.MongoClient()
            db = client[database]
            data = pd.DataFrame(list(db[collection].find()))
            updatelog.log('process', 'Successfully loaded data from database to dataframe')
            return data
        except:
            updatelog.log('error', "Unexpected error while reading from MongoDB:"+str(sys.exc_info()[0])+str(sys.exc_info()[1]))
            