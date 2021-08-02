#!/bin/python

from Utils import Utils
from DataCrawler import DataCrawler
from Logger import Logger
from MongoDB import MongoDB

Utils.createFolderIfNotExists(DataCrawler.TMP_FOLDER)
LOGGER=Logger(DataCrawler.TMP_FOLDER,name='crawler')
Utils(DataCrawler.TMP_FOLDER,LOGGER)

LOGGER.info('Starting Data Crawler...')

if Utils.runningOnDockerContainer(){
    mongo_addr='mongo'
}else{
    mongo_addr='127.0.0.1'
}
mongo=MongoDB(mongo_addr,27017,LOGGER,user='root',password='123456')
crawler=DataCrawler(mongo,LOGGER)
mongo.startQueue(id=0)
LOGGER.info('Started Data Crawler...OK')
LOGGER.info('Listening on queue as {}'.format(mongo.getQueueConsumerId()))
crawler.loopOnQueue()