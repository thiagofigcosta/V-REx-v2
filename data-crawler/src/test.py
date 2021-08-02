#!/bin/python

from Utils import Utils
from DataCrawler import DataCrawler
from Logger import Logger
from MongoDB import MongoDB
from XmlDictParser import XmlDictParser

Utils.createFolderIfNotExists(DataCrawler.TMP_FOLDER)
LOGGER=Logger(DataCrawler.TMP_FOLDER,name='crawler')
Utils(DataCrawler.TMP_FOLDER,LOGGER)

LOGGER.info('oi')
LOGGER.warn('oi')
LOGGER.error('oi')
try{
    vaidarerro+10
}except Exception as e{
    LOGGER.exception(e)
}
LOGGER.exception(Exception('TEST exception printing'))




mongo=MongoDB.asDummy(LOGGER)
crawler=DataCrawler(mongo,LOGGER)

# crawler.downloadRawDataFromSources(sources=['CVE_MITRE'])
crawler.downloadRawDataFromSources(sources=['CWE_MITRE'])
# crawler.downloadRawDataFromSources(sources=['CVE_NVD'])
# crawler.downloadRawDataFromSources(sources=['CAPEC_MITRE'])
# crawler.downloadRawDataFromSources(sources=['OVAL'])
# crawler.downloadRawDataFromSources(sources=['EXPLOIT_DB'])
# crawler.downloadRawDataFromSources(sources=['CVE_DETAILS'])


# crawler.downloadRawDataFromSources()

mongo=MongoDB('127.0.0.1',27017,LOGGER,user='root',password='123456')
mongo.dumpDB(mongo.getDB('queue'),'tmp/crawler')
mongo.restoreDB('tmp/crawler/queue.zip',db_name='queue_dump')

LOGGER.fatal('GG')