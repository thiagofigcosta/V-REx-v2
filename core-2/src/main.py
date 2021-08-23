#!/bin/python

import sys, getopt
from Core import Core
from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB

TMP_FOLDER='tmp/core/'

Utils.createFolderIfNotExists(TMP_FOLDER)
LOGGER=Logger(TMP_FOLDER,name='core')
Utils(TMP_FOLDER,LOGGER)


def loopOnQueue(core){
    while True{
        job=core.mongo.getQueues()[MongoDB.QUEUE_COL_CORE_NAME].next()
        if job is not None{
            payload=job.payload
            task=payload['task']
            try{
                LOGGER.info('Running job {}-{}...'.format(task,job.job_id))
                if task=='Genetic'{
                    core.runGeneticSimulation(payload['args']['simulation_id'])
                }elif task=='Train SNN'{
                    core.trainNeuralNetwork(payload['args']['independent_net_id'],False,True)
                    core.trainNeuralNetwork(payload['args']['independent_net_id'],True,False)
                }elif task=='Eval SNN'{
                    core.predictNeuralNetwork(payload['args']['independent_net_id'],payload['args']['result_id'],payload['args']['eval_data'])
                }
                if job{
                    job.complete()
                    LOGGER.info('Runned job {}-{}...'.format(task,job.job_id))
                }
            }except Exception as e{
                job.error(str(e))
                LOGGER.error('Failed to run job {}-{}...'.format(task,job.job_id))      
                LOGGER.exception(e)
            }
        }
    }
}

if __name__ == "__main__"{
    LOGGER.info('Starting Core...')
    if Utils.runningOnDockerContainer(){
        mongo_addr='mongo'
    }else{
        mongo_addr='127.0.0.1'
    }
    mongo=MongoDB(mongo_addr,27017,LOGGER,user='root',password='123456')
    mongo.startQueue(id=0)
    LOGGER.info('Started Core...OK')
    LOGGER.info('Writting on queue as {}'.format(mongo.getQueueConsumerId()))
    core=Core(mongo,LOGGER)
    loopOnQueue(core)
}
