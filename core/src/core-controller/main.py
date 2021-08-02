#!/bin/python

import sys,os
from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB

TMP_FOLDER='/tmp/controller/'

trap_signals=False
set_stack_size=False
debug_args=False
verbose_neural=False
avoid_genetic_cache=False
mongo_addr=''

Utils.createFolderIfNotExists(TMP_FOLDER)
LOGGER=Logger(TMP_FOLDER,name='controller')
Utils(TMP_FOLDER,LOGGER)

def runCore(bin_cmd,args,print_out=True){
    setup_cmd='ulimit -s unlimited'
    bin_cmd='{} ; {}'.format(setup_cmd,bin_cmd)
    if trap_signals{
        args+=' --trap-signals'
    }
    if set_stack_size{
        args+=' --set-stack-size'
    }
    if debug_args{
        args+=' --debug-args'
    }
    if verbose_neural{
        args+=' --verbose-neural'
    }
    if avoid_genetic_cache{
        args+=' --do-not-cache-genetic'
    }
    cmd='{} {}'.format(bin_cmd,args)
    if not print_out{
        cmd+=' 1> /dev/null'
    }
    LOGGER.info('Running CORE with cmd: {}...'.format(cmd))
    try{
        return_value=os.system(cmd) # use subprocess.Popen if system not work
        return_value=int(bin(return_value).replace('0b', '').rjust(16, '0')[:8], 2)
    }except{
        return_value=-1
    }
    if return_value==0{
        LOGGER.info('Runned CORE and got return code: {}...OK'.format(return_value))
        return True
    }else{
        LOGGER.error('Runned CORE and got return code: {}...FAIL'.format(return_value))
        return False
    }
}


def loopOnQueue(){
    core_bin=os.getenv('CORE_BIN_PATH',default='/vrex/src/bin/vrex')
    while True{
        job=mongo.getQueues()[MongoDB.QUEUE_COL_CORE_NAME].next()
        if job is not None{
            payload=job.payload
            task=payload['task']
            try{
                LOGGER.info('Running job {}-{}...'.format(task,job.job_id))
                if task=='Genetic'{
                    single_thread=False
                    done=False
                    args='--run-genetic --simulation-id {}'.format(payload['args']['simulation_id'])
                    if single_thread{
                        args+=' --single-thread'
                    }
                    max_attempts=10
                    cur_try=0
                    while cur_try<max_attempts and not done{
                        if runCore(core_bin,args){
                            done=True
                        }else{
                            cur_try+=1
                            if (cur_try<max_attempts) {
                                LOGGER.info('Failed to run core due to exception, trying again {} of {}...'.format(cur_try,max_attempts))
                            }
                        }
                    }
                    if not done{
                        raise Exception('Failed to run genetic experiment using simulation: {}'.format(payload['args']['simulation_id']))
                    }
                }elif task=='Train SNN'{
                    done=False
                    args='--train-neural --network-id {} --just-train'.format(payload['args']['independent_net_id'])
                    max_attempts=10
                    cur_try=0
                    while cur_try<max_attempts and not done{
                        if runCore(core_bin,args){
                            done=True
                        }else{
                            cur_try+=1
                            if (cur_try<max_attempts) {
                                LOGGER.info('Failed to run core due to exception, trying again {} of {}...'.format(cur_try,max_attempts))
                            }
                        }
                    }
                    if not done{
                        raise Exception('Failed to train network ({}) [step 1/2]'.format(payload['args']['independent_net_id']))
                    }else{
                        done=False
                        args='--train-neural --network-id {} --continue'.format(payload['args']['independent_net_id'])
                        max_attempts=100
                        cur_try=0
                        while cur_try<max_attempts and not done{
                            if runCore(core_bin,args){
                                done=True
                            }else{
                                cur_try+=1
                                if (cur_try<max_attempts) {
                                    LOGGER.info('Failed to run core due to exception, trying again {} of {}...'.format(cur_try,max_attempts))
                                }
                            }
                        }
                        if not done{
                            raise Exception('Failed to eval network ({}) train [step 2/2]'.format(payload['args']['independent_net_id']))
                        }
                    }
                }elif task=='Eval SNN'{
                    done=False
                    args='--eval-neural --network-id {} --eval-result-id {} --eval-data {}'.format(payload['args']['independent_net_id'],payload['args']['result_id'],payload['args']['eval_data'])
                    max_attempts=33
                    cur_try=0
                    while cur_try<max_attempts and not done{
                        if runCore(core_bin,args){
                            done=True
                        }else{
                            cur_try+=1
                            if (cur_try<max_attempts) {
                                LOGGER.info('Failed to run core due to exception, trying again {} of {}...'.format(cur_try,max_attempts))
                            }
                        }
                    }
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
    LOGGER.info('Starting Core controller...')
    if Utils.runningOnDockerContainer(){
        mongo_addr='mongo'
    }else{
        mongo_addr='127.0.0.1'
    }
    mongo=MongoDB(mongo_addr,27017,LOGGER,user='root',password='123456')
    mongo.startQueue(id=0)
    LOGGER.info('Started Core controller...OK')
    LOGGER.info('Writting on queue as {}'.format(mongo.getQueueConsumerId()))
    loopOnQueue()
}