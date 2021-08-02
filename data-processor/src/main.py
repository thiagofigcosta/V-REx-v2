#!/bin/python

from DataProcessor import DataProcessor
from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB

Utils.createFolderIfNotExists(DataProcessor.TMP_FOLDER)
LOGGER=Logger(DataProcessor.TMP_FOLDER,verbose=False,name='processor')
Utils(DataProcessor.TMP_FOLDER,LOGGER)

LOGGER.info('Starting Data Processor...')
if Utils.runningOnDockerContainer(){
    mongo_addr='mongo'
}else{
    mongo_addr='127.0.0.1'
}
mongo=MongoDB(mongo_addr,27017,LOGGER,user='root',password='123456')
processor=DataProcessor(mongo,LOGGER)
mongo.startQueue(id=0)
LOGGER.info('Started Data Processor...OK')
LOGGER.info('Listening on queue as {}'.format(mongo.getQueueConsumerId()))
processor.loopOnQueue()