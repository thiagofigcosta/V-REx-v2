#!/bin/python

from DataProcessor import DataProcessor
from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB

Utils.createFolderIfNotExists(DataProcessor.TMP_FOLDER)
LOGGER=Logger(DataProcessor.TMP_FOLDER,verbose=True,name='processor')
Utils(DataProcessor.TMP_FOLDER,LOGGER)

mongo=MongoDB('127.0.0.1',27017,LOGGER,user='root',password='123456')
mongo.startQueue(id=0)
print(mongo.getQueueConsumerId())

processor=DataProcessor(mongo,LOGGER)

processor.filterAndNormalizeFullDataset()