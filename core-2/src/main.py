#!/bin/python

import sys, getopt, bson, re
from Core import Core
from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB

TMP_FOLDER='tmp/core/'

ITERATIVE=False

Utils.createFolderIfNotExists(TMP_FOLDER)
LOGGER=Logger(TMP_FOLDER,name='core')
Utils(TMP_FOLDER,LOGGER)


def loopOnQueue(core){
    while True{
        job=mongo.getQueues()[MongoDB.QUEUE_COL_CORE_NAME].next()
        if job is not None{
            payload=job.payload
            task=payload['task']
            try{
                LOGGER.info('Running job {}-{}...'.format(task,job.job_id))
                if task=='Genetic'{
                    done=False
                    core.runGeneticSimulation(payload['args']['simulation_id'])
                    if not done{
                        raise Exception('Failed to run genetic experiment using simulation: {}'.format(payload['args']['simulation_id']))
                    }
                }elif task=='Train SNN'{
                    done=False
                    core.trainNeuralNetwork(payload['args']['independent_net_id'],False,True)
                    core.trainNeuralNetwork(payload['args']['independent_net_id'],True,False)
                    if not done{
                        raise Exception('Failed to train network ({}) [step 1/2] and eval network ({}) train [step 2/2]'.format(payload['args']['independent_net_id'],payload['args']['independent_net_id']))
                    }
                }elif task=='Eval SNN'{
                    done=False
                    core.evalNeuralNetwork(payload['args']['independent_net_id'],payload['args']['result_id'],payload['args']['eval_data'])
                    if not done{
                        raise Exception('Failed to eval neural network ({}) with data: {}'.format(payload['args']['independent_net_id'],payload['args']['eval_data']))
                    }else{
                        LOGGER.info('Result stored at {}/neural_db/eval_results/{}'.format(mongo_addr,payload['args']['result_id']))
                    }
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
