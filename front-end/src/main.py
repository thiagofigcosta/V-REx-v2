#!/bin/python

import sys, getopt, bson, re
from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB

TMP_FOLDER='tmp/front/'

ITERATIVE=False

Utils.createFolderIfNotExists(TMP_FOLDER)
LOGGER=Logger(TMP_FOLDER,name='front')
Utils(TMP_FOLDER,LOGGER)

def inputNumber(is_float=False,greater_or_eq=0,lower_or_eq=None){
    out=0
    converted=False
    while not converted{
        try{
            if is_float{
                out=float(input())
            }else{
                out=int(input())
            }
            if (lower_or_eq==None or out <= lower_or_eq) and (greater_or_eq==None or out >= greater_or_eq){ 
                converted=True
            }else{
                print('ERROR. Out of boundaries [{},{}], type again: '.format(greater_or_eq,lower_or_eq))
            }
        }except ValueError{
            if not is_float{
                print('ERROR. Not an integer, type again: ')
            }else{
                print('ERROR. Not a float number, type again: ')
            } 
        }
    }
    return out
}

def inputArrayNumber(is_float=False,greater_or_eq=0,lower_or_eq=None){
    out=''
    not_converted=True
    while not_converted{
        try{
            out=input()
            out_test=out.split(',')
            for test in out_test{
                if is_float{
                    test=float(test)
                }else{
                    test=int(test)
                }
                if (lower_or_eq==None or test <= lower_or_eq) and (greater_or_eq==None or test >= greater_or_eq){ 
                    not_converted=False
                }else{
                    print('ERROR. Out of boundaries [{},{}], type again: '.format(greater_or_eq,lower_or_eq))
                    not_converted=True
                    break
                }
            }
        }except ValueError{
            if not is_float{
                print('ERROR. Not an integer, type again: ')
            }else{
                print('ERROR. Not a float number, type again: ')
            } 
        }
    }
    return out
}

def main(argv){
    HELP_STR='main.py [-h]\n\t[--check-jobs]\n\t[--create-genetic-env]\n\t[--list-genetic-envs]\n\t[--run-genetic]\n\t[--show-genetic-results]\n\t[--rm-genetic-env <env name>]\n\t[--parse-dna-to-hyperparams]\n\t[--create-smart-neural-hyperparams]\n\t[--list-smart-neural-hyperparams]\n\t[--rm-smart-neural-hyperparams <hyper name>]\n\t[--train-smart-neural]\n\t[--train-smart-neural-gen-mode]\n\t[--train-smart-neural-gen-mode-continue <simulation_id>:<independent_net_id>]\n\t[--eval-smart-neural]\n\t[--get-queue-names]\n\t[--get-all-db-names]\n\t[-q | --quit]\n\t[--run-processor-pipeline]\n\t[--run-merge-cve]\n\t[--run-flattern-and-simplify-all]\n\t[--run-flattern-and-simplify [cve|oval|capec|cwe]]\n\t[--run-filter-exploits]\n\t[--run-transform-all]\n\t[--run-transform [cve|oval|capec|cwe|exploits]]\n\t[--run-enrich]\n\t[--run-analyze]\n\t[--run-filter-and-normalize]\n\t[--download-source <source ID>]\n\t[--download-all-sources]\n\t[--empty-queue <queue name>]\n\t[--empty-all-queues]\n\t[--dump-db <db name>#<folder path to export> | --dump-db <db name> {saves on default tmp folder} \n\t\te.g. --dump-db queue#/home/thiago/Desktop]\n\t[--restore-db <file path to import>#<db name> | --restore-db <file path to import> {saves db under file name} \n\t\te.g. --restore-db /home/thiago/Desktop/queue.zip#restored_queue]\n\t[--keep-alive-as-zombie]'
    args=[]
    zombie=False
    global ITERATIVE
    to_run=[]
    try{ 
        opts, args = getopt.getopt(argv,"hq",["keep-alive-as-zombie","download-source=","download-all-sources","check-jobs","quit","get-queue-names","empty-queue=","empty-all-queues","get-all-db-names","dump-db=","restore-db=","run-processor-pipeline","run-flattern-and-simplify-all","run-flattern-and-simplify=","run-filter-exploits","run-transform-all","run-transform=","run-enrich","run-analyze","run-filter-and-normalize","run-merge-cve","create-genetic-env","list-genetic-envs","rm-genetic-env=","run-genetic","show-genetic-results","create-smart-neural-hyperparams","list-smart-neural-hyperparams","rm-smart-neural-hyperparams=","train-smart-neural","eval-smart-neural","parse-dna-to-hyperparams","train-smart-neural-gen-mode","train-smart-neural-gen-mode-continue="])
    }except getopt.GetoptError{
        print (HELP_STR)
        if not ITERATIVE {
            sys.exit(2)
        }
    }
    try{
        for opt, arg in opts{ 
            if opt == '-h'{
                print (HELP_STR)
            }elif opt in ('-q','--quit'){
                exit()
            }elif opt == "--keep-alive-as-zombie"{
                zombie=True
            }elif opt == "--dump-db"{
                splited_arg=arg.split('#')
                if len(splited_arg)>0 and len(splited_arg)<=2{
                    path_to_export=TMP_FOLDER
                    if len(splited_arg)>1{
                        path_to_export=splited_arg[1]
                    }
                    mongo.dumpDB(mongo.getDB(splited_arg[0]),path_to_export)
                }else{
                    LOGGER.error('Invalid argument, type the db_name or "db_name#path": {}'.format(arg))
                }
            }elif opt == "--list-smart-neural-hyperparams"{
                LOGGER.info('Getting hyperparameters...')
                for hyp in mongo.findAllOnDB(mongo.getDB('neural_db'),'snn_hyperparameters',wait_unlock=False){
                    for k,v in hyp.items(){
                        if k!='_id'{
                            LOGGER.clean('{}: {}'.format(k,str(v)))
                        }
                    }
                    LOGGER.clean('\n')
                }
                LOGGER.info('Gotten hyperparameters...OK')
            }elif opt == "--parse-dna-to-hyperparams"{
                LOGGER.info('Casting dna...')
                print('Enter the int dna (e.g. [ 2, 5, 1, 1, 16, 2, 45 ]): ', end = '')
                not_filled=True
                int_dna=''
                while not_filled {
                    int_dna=''.join(input().split()).replace('[','',1).replace(']','',1)
                    if re.match(r'^(:?[0-9]*,?)*$',int_dna){
                        not_filled=False
                    }else{
                        print('ERROR - Wrong int DNA format')
                    }
                }
                print('Enter the float dna (e.g. [ 0.028238 ]): ', end = '')
                not_filled=True
                float_dna=''
                while not_filled {
                    float_dna=''.join(input().split()).replace('[','',1).replace(']','',1)
                    if re.match(r'^(:?[0-9\.]*,?)*$',float_dna){
                        not_filled=False
                    }else{
                        print('ERROR - Wrong float DNA format')
                    }
                }
                int_dna=int_dna.split(',')
                float_dna=float_dna.split(',')
                int_dna=[int(i) for i in int_dna]
                float_dna=[float(i) for i in float_dna]
                LOGGER.info('Considering border_sparsity=1 and output_size=OUT')
                epochs=int_dna[0]
                batch_size=int_dna[1]
                layers=int_dna[2]
                max_layers=int_dna[3]
                layer_sizes=[0]*layers
                range_pow=[0]*layers
                K=[0]*layers
                L=[0]*layers
                node_types=[0]*layers
                sparcity=[0]*layers
                sparcity[0]=1
                sparcity[layers-1]=1
                layer_sizes[layers-1]='OUT'
                node_types[layers-1]=1 #NodeType::Softmax
                l=4
                i=0
                while(i<max_layers-1){
                    if (i+1<layers){
                        layer_sizes[i]=int_dna[l]
                    }
                    i+=1
                    l+=1
                }
                i=0
                while(i<max_layers){
                    if (i<layers){
                        range_pow[i]=int_dna[l]
                    }
                    i+=1
                    l+=1
                }
                i=0
                while(i<max_layers){
                    if (i<layers){
                        dna_value=int_dna[l]
                        if (type(layer_sizes[i]) is int){
                            if (dna_value>layer_sizes[i]){  # ??? K cannot be higher than layer size?
                                dna_value=layer_sizes[i]
                            }
                        }
                        K[i]=dna_value
                    }
                    i+=1
                    l+=1
                }
                i=0
                while(i<max_layers){
                    if (i<layers){
                        L[i]=int_dna[l]
                    }
                    i+=1
                    l+=1
                }
                i=0
                while(i<max_layers-1){
                    if (i+1<layers){
                        node_types[i]=int_dna[l]
                    }
                    i+=1
                    l+=1
                }
                alpha=float_dna[0]
                l=1
                i=1
                while(i<max_layers-1){
                    if (i+1<layers){
                        sparcity[i]=float_dna[l]
                    }
                    i+=1
                    l+=1
                }
                LOGGER.info('epochs: {}'.format(epochs))
                LOGGER.info('batch_size: {}'.format(batch_size))
                LOGGER.info('layers: {}'.format(layers))
                LOGGER.info('alpha: {}'.format(alpha))
                LOGGER.info('layer_sizes: {}'.format(str(layer_sizes)))
                LOGGER.info('range_pow: {}'.format(str(range_pow)))
                LOGGER.info('K: {}'.format(str(K)))
                LOGGER.info('L: {}'.format(str(L)))
                LOGGER.info('node_types: {}'.format(str(node_types)))
                LOGGER.info('sparcity: {}'.format(str(sparcity)))
                LOGGER.info('Cast dna...OK')
            }elif opt == "--create-smart-neural-hyperparams"{
                print('Now type the hyperparameters for the Smart Neural Network...')
                print('Enter the hyperparameters config name (unique): ', end = '')
                hyper_name=input().strip()
                submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                print('Enter the batch size: ', end = '')
                batch_size=inputNumber()
                print('Enter the alpha: ', end = '')
                alpha=inputNumber(is_float=True,lower_or_eq=1)
                print('Enter 1 to shuffle train data or 0 otherwise: ', end = '')
                shuffle=inputNumber(lower_or_eq=1)==1
                print('Enter 1 to use adam optimizer or 0 otherwise: ', end = '')
                adam=inputNumber(lower_or_eq=1)==1
                print('Enter the rehash (6400): ', end = '')
                rehash=inputNumber()
                print('Enter the rebuild (128000): ', end = '')
                rebuild=inputNumber()
                print('Enter the label type (0-2):')
                print('\t0 - INT_CLASS')
                print('\t1 - NEURON_BY_NEURON')
                print('\t2 - NEURON_BY_NEURON_LOG_LOSS')
                print('value: ', end='')
                label_type=inputNumber(lower_or_eq=2)
                print('Enter amount of layers: ', end = '')
                layers=inputNumber(greater_or_eq=1)
                layer_sizes=[]
                for i in range(layers){
                    if i==0{
                        print('Enter the layer size for layer 0 (output size): ', end = '')
                    }else{
                        print('Enter the layer size for layer {}: '.format(i), end = '')
                    }
                    layer_sizes.append(inputNumber())
                }
                range_pow=[]
                for i in range(layers){
                    print('Enter the range pow for layer {}: '.format(i), end = '')
                    range_pow.append(inputNumber())
                }
                K=[]
                for i in range(layers){
                    print('Enter the K for layer {}: '.format(i), end = '')
                    K.append(inputNumber())
                }
                L=[]
                for i in range(layers){
                    print('Enter the L for layer {}: '.format(i), end = '')
                    L.append(inputNumber())
                }
                if layers > 1 {
                    print('Node types (0-2):')
                    print('\t0 - ReLU')
                    print('\t1 - Softmax')
                    print('\t2 - Sigmoid')
                    print('value: ', end='')
                }
                node_types=[]
                for i in range(layers-1){
                    print('Enter the node type for layer {}: '.format(i), end = '')
                    node_types.append(inputNumber(lower_or_eq=2))
                }
                node_types.append(1) # softmax
                sparcity=[1] # border
                for i in range(layers-2){
                    print('Enter the sparcity for layer {}: '.format(i), end = '')
                    sparcity.append(inputNumber(is_float=True,lower_or_eq=1))
                }
                if layers > 1 {
                    sparcity.append(1); # border
                }
                hyperparams={'name':hyper_name,'submitted_at':submitted_at,'batch_size':batch_size,'alpha':alpha,'shuffle':shuffle,'adam':adam,'rehash':rehash,'rebuild':rebuild,'label_type':label_type,'layers':layers,'layer_sizes':layer_sizes,'range_pow':range_pow,'K':K,'L':L,'node_types':node_types,'sparcity':sparcity}
                LOGGER.info('Writting hyperparameters on neural_db...')
                mongo.insertOneOnDB(mongo.getDB('neural_db'),hyperparams,'snn_hyperparameters',index='name',ignore_lock=True)
                LOGGER.info('Wrote hyperparameters on neural_db...OK')
            }elif opt == "--list-genetic-envs"{
                LOGGER.info('Getting genetic environments...')
                for env in mongo.findAllOnDB(mongo.getDB('genetic_db'),'environments',wait_unlock=False){
                    LOGGER.clean('Name: {}'.format(env['name']))
                    LOGGER.clean('Submitted At: {}'.format(env['submitted_at']))
                    LOGGER.clean('Space Search:')
                    for k,v in env['space_search'].items() {
                        LOGGER.clean('\t{}: {}'.format(k,str(v)))
                    }
                    LOGGER.clean('\n')
                }
                LOGGER.info('Gotten genetic environments...OK')
            }elif opt == "--eval-smart-neural"{
                print('Enter a existing independent network name to be used: ', end = '')
                independent_net_name=input().strip()
                independent_net=mongo.findOneOnDBFromIndex(mongo.getDB('neural_db'),'independent_net','name',independent_net_name,wait_unlock=False)
                if independent_net==None{
                    LOGGER.error('Not found a independent network for the given name!')
                }else{
                    result_info={'total_test_cases':0,'matching_preds':0,'warning':'matching predictions are not ground truth','result_stats':None,'results':None}
                    LOGGER.info('Writting eval result holder...')
                    result_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),result_info,'eval_results')
                    LOGGER.info('Wrote eval result holder...OK')
                    if (result_id==None){
                        LOGGER.error('Failed to insert eval result holder!')
                    }else{
                        print('Eval single CVE or multiple (0 - single | 1 - multiple): ')
                        if inputNumber(lower_or_eq=1)==1{
                            print('Enter the years to be used as eval data splitted by comma (,) (1999-2020):')
                            eval_data=inputArrayNumber(greater_or_eq=1999,lower_or_eq=2020)
                            print('Enter a limit of CVEs for each year (0 = unlimitted): ')
                            eval_data+=':{}'.format(inputNumber())
                            job_args={'eval_data':eval_data,'result_id':result_id,'independent_net_id':str(independent_net['_id'])}
                        }else{
                            print('Enter a CVE id following the format CVE-####-#*: ')
                            not_filled=True
                            eval_cve=''
                            while not_filled {
                                eval_cve=input().strip()
                                if re.match(r'^CVE-[0-9]{4}-[0-9]*$',eval_cve){
                                    not_filled=False
                                }else{
                                    print('ERROR - Wrong CVE format')
                                }
                            }
                            job_args={'eval_data':eval_cve,'result_id':result_id,'independent_net_id':str(independent_net['_id'])}
                        }
                        LOGGER.info('Writting on Core queue to eval network...')
                        mongo.insertOnCoreQueue('Eval SNN',job_args,priority=1)
                        LOGGER.info('Wrote on Core queue to eval network...OK')
                    }
                } 
            }elif opt == "--train-smart-neural-gen-mode-continue"{
                arg=arg.strip().split(':')
                simulation_id=arg[0]
                independent_net_id=arg[1]
                query={'_id':bson.ObjectId(simulation_id)}
                simulation=mongo.findOneOnDB(mongo.getDB('genetic_db'),'simulations',query)

                if simulation!=None{
                    query={'_id':bson.ObjectId(simulation['population_id'])}
                    population=mongo.findOneOnDB(mongo.getDB('neural_db'),'populations',query)
                    if population!=None{
                        LOGGER.info('Updating individual_net on neural_db...')
                        query={'_id':bson.ObjectId(independent_net_id)}
                        update={'$set':{'started_by':simulation['started_by'],'started_at':simulation['started_at'],'finished_at':simulation['finished_at'],'weights':population['neural_genomes'][0]['weights']}}
                        mongo.getDB('neural_db')['independent_net'].find_one_and_update(query,update)
                        LOGGER.info('Updated individual_net on neural_db...OK')
                    }else{
                        LOGGER.error('Population not found!')
                    }
                }else{
                    LOGGER.error('Simulation not found!')
                }
            }elif opt == "--train-smart-neural-gen-mode"{
                print('Now enter the data to train the neural network...')
                print('Enter a existing hyperparameters name to be used: ', end = '')
                hyper_name=input().strip()
                hyper=mongo.findOneOnDBFromIndex(mongo.getDB('neural_db'),'snn_hyperparameters','name',hyper_name,wait_unlock=False)
                if hyper==None{
                    LOGGER.error('Not found a hyperparameter for the given name!')
                }else{
                    print('Enter the epochs: ',end='')
                    epochs=inputNumber()
                    print('Enter the cross validation method (0-4):')
                    print('\t0 - NONE')
                    print('\t1 - ROLLING_FORECASTING_ORIGIN')
                    print('\t2 - KFOLDS')
                    print('\t3 - TWENTY_PERCENT')
                    print('value: ', end='')
                    cross_validation=inputNumber(lower_or_eq=3)
                    print('Enter the metric to be used during training/val (0-4):')
                    print('\t0 - RAW_LOSS (cheaper)')
                    print('\t1 - F1')
                    print('\t2 - RECALL')
                    print('\t3 - ACCURACY')
                    print('\t4 - PRECISION')
                    print('value: ',end='')
                    train_metric=inputNumber(lower_or_eq=4)
                    print('Enter the years to be used as train data splitted by comma (,) (1999-2020):')
                    train_data=inputArrayNumber(greater_or_eq=1999,lower_or_eq=2020)
                    print('Enter a limit of CVEs for each year (0 = unlimitted): ')
                    train_data+=':{}'.format(inputNumber())
                    test_metric=0
                    test_data=''
                    gen_name='{}_ss'.format(hyper_name)
                    space_search=mongo.findOneOnDBFromIndex(mongo.getDB('genetic_db'),'environments','name',gen_name,wait_unlock=False)
                    if space_search==None{
                        space_search={'name':gen_name,'submitted_at':Utils.getTodayDate('%d/%m/%Y %H:%M:%S'),'space_search':{'amount_of_layers':{'min':hyper['layers'],'max':hyper['layers']},'epochs':{'min':epochs,'max':epochs},'batch_size':{'min':hyper['batch_size'],'max':hyper['batch_size']},'layer_sizes':{'min':hyper['layers'],'max':hyper['layers']},'range_pow':{'min':hyper['range_pow'][0],'max':hyper['range_pow'][0]},'K':{'min':hyper['K'][0],'max':hyper['K'][0]},'L':{'min':hyper['L'][0],'max':hyper['L'][0]},'activation_functions':{'min':hyper['node_types'][0],'max':hyper['node_types'][0]},'sparcity':{'min':hyper['sparcity'][0],'max':hyper['sparcity'][0]},'alpha':{'min':hyper['alpha'],'max':hyper['alpha']}}}
                        LOGGER.info('Writting environment on genetic_db...')
                        mongo.insertOneOnDB(mongo.getDB('genetic_db'),space_search,'environments',index='name',ignore_lock=True)
                        LOGGER.info('Wrote environment on genetic_db...OK')
                    }
                    print('Enter a name for the train (unique): ', end = '')
                    train_name=input().strip()
                    submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                    started_by=None
                    started_at=None
                    finished_at=None
                    weights=None
                    train_metadata={'name':train_name,'hyperparameters_name':hyper_name,'submitted_at':submitted_at,'started_by':started_by,'started_at':started_at,'finished_at':finished_at,'epochs':epochs,'cross_validation':cross_validation,'train_metric':train_metric,'train_data':train_data,'test_metric':test_metric,'test_data':test_data,'weights':weights}
                    LOGGER.info('Writting individual_net on neural_db...')
                    independent_net_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),train_metadata,'independent_net',index='name')
                    if (independent_net_id==None){
                        LOGGER.error('Failed to insert individual_net!')
                    }else{
                        LOGGER.info('Wrote individual_net on neural_db...OK')
                        simulation_name='{}_s'.format(train_name)
                        hall_of_fame_id=None
                        population_id=None
                        best={'output':None,'at_gen':None}
                        results=[]
                        simulation_data={'name':simulation_name,'env_name':gen_name,'submitted_at':submitted_at,'started_by':started_by,'started_at':started_at,'finished_at':finished_at,'hall_of_fame_id':hall_of_fame_id,'population_id':population_id,'pop_start_size':1,'max_gens':1,'algorithm':1,'max_age':0,'max_children':0,'mutation_rate':0,'recycle_rate':0,'sex_rate':0,'max_notables':1,'cross_validation':cross_validation,'label_type':hyper['label_type'],'metric':train_metric,'train_data':train_data,'best':best,'results':results}
                        LOGGER.info('Writting simulation config on genetic_db...')
                        simulation_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('genetic_db'),simulation_data,'simulations')
                        if (simulation_id==None){
                            LOGGER.error('Failed to insert simulation!')
                        }else{
                            LOGGER.info('Wrote simulation config on genetic_db...OK')
                            halloffame_data={'simulation_id':simulation_id,'env_name':gen_name,'updated_at':None,'neural_genomes':[]}
                            halloffame_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),halloffame_data,'hall_of_fame')
                            generation_data={'simulation_id':simulation_id,'env_name':gen_name,'updated_at':None,'neural_genomes':[]}
                            population_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),generation_data,'populations')
                            if halloffame_id == None or population_id == None{
                                LOGGER.error('Failed to insert hall of fame or/and generation!')
                            }else{
                                query={'_id':bson.ObjectId(simulation_id)}
                                update={'$set':{'hall_of_fame_id':halloffame_id,'population_id':population_id}}
                                mongo.getDB('genetic_db')['simulations'].find_one_and_update(query,update)

                                LOGGER.info('Writting on Core queue to train network...')
                                job_args={'simulation_id':simulation_id}
                                mongo.insertOnCoreQueue('Genetic',job_args,priority=3)
                                LOGGER.info('Wrote on Core queue to train network...OK')
                                LOGGER.info('\n\n\n\n\nType `--train-smart-neural-gen-mode-continue {}:{}` when the core simulation {} finishes.\n'.format(simulation_id,independent_net_id,simulation_id))
                            }
                        }
                    }
                }
            }elif opt == "--train-smart-neural"{
                print('Now enter the data to train the neural network...')
                print('Enter a existing hyperparameters name to be used: ', end = '')
                hyper_name=input().strip()
                hyper=mongo.findOneOnDBFromIndex(mongo.getDB('neural_db'),'snn_hyperparameters','name',hyper_name,wait_unlock=False)
                if hyper==None{
                    LOGGER.error('Not found a hyperparameter for the given name!')
                }else{
                    print('Enter a name for the train (unique): ', end = '')
                    train_name=input().strip()
                    submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                    started_by=None
                    started_at=None
                    finished_at=None
                    weights=None
                    print('Enter the epochs: ',end='')
                    epochs=inputNumber()
                    print('Enter the cross validation method (0-4):')
                    print('\t0 - NONE')
                    print('\t1 - ROLLING_FORECASTING_ORIGIN')
                    print('\t2 - KFOLDS')
                    print('\t3 - TWENTY_PERCENT')
                    print('value: ', end='')
                    cross_validation=inputNumber(lower_or_eq=3)
                    print('Enter the metric to be used during training/val (0-4):')
                    print('\t0 - RAW_LOSS (cheaper)')
                    print('\t1 - F1')
                    print('\t2 - RECALL')
                    print('\t3 - ACCURACY')
                    print('\t4 - PRECISION')
                    print('value: ',end='')
                    train_metric=inputNumber(lower_or_eq=4)
                    print('Enter the years to be used as train data splitted by comma (,) (1999-2020):')
                    train_data=inputArrayNumber(greater_or_eq=1999,lower_or_eq=2020)
                    print('Enter a limit of CVEs for each year (0 = unlimitted): ')
                    train_data+=':{}'.format(inputNumber())
                    print('Run test data also (0 - no | 1 - yes): ')
                    if inputNumber(lower_or_eq=1)==1{
                        print('Enter the metric to be used during test (0-4):')
                        print('\t0 - RAW_LOSS (cheaper)')
                        print('\t1 - F1')
                        print('\t2 - RECALL')
                        print('\t3 - ACCURACY')
                        print('\t4 - PRECISION')
                        print('value: ',end='')
                        test_metric=inputNumber(lower_or_eq=4)

                        print('Enter the years to be used as test data splitted by comma (,) (1999-2020):')
                        test_data=inputArrayNumber(greater_or_eq=1999,lower_or_eq=2020)
                        print('Enter a limit of CVEs for each year (0 = unlimitted): ')
                        test_data+=':{}'.format(inputNumber())
                    }else{
                        test_metric=0
                        test_data=''
                    }
                    train_metadata={'name':train_name,'hyperparameters_name':hyper_name,'submitted_at':submitted_at,'started_by':started_by,'started_at':started_at,'finished_at':finished_at,'epochs':epochs,'cross_validation':cross_validation,'train_metric':train_metric,'train_data':train_data,'test_metric':test_metric,'test_data':test_data,'weights':weights}
                    LOGGER.info('Writting individual_net on neural_db...')
                    independent_net_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),train_metadata,'independent_net',index='name')
                    if (independent_net_id==None){
                        LOGGER.error('Failed to insert individual_net!')
                    }else{
                        LOGGER.info('Wrote individual_net on neural_db...OK')

                        LOGGER.info('Writting on Core queue to train network...')
                        job_args={'independent_net_id':independent_net_id}
                        mongo.insertOnCoreQueue('Train SNN',job_args,priority=2)
                        LOGGER.info('Wrote on Core queue to train network...OK')
                    }
                }
            }elif opt == "--run-genetic"{
                print('Now enter the data to run the genetic experiment...')
                print('Enter a existing genetic environment name to be used: ', end = '')
                env_name=input().strip()
                env=mongo.findOneOnDBFromIndex(mongo.getDB('genetic_db'),'environments','name',env_name,wait_unlock=False)
                if env==None{
                    LOGGER.error('Not found an environment for the given name!')
                }else{
                    print('Enter a name for the simulation (not unique, just for reference): ', end = '')
                    simulation_name=input().strip()
                    submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                    started_by=None
                    started_at=None
                    finished_at=None
                    hall_of_fame_id=None
                    population_id=None
                    print('Enter the population start size: ')
                    pop_start_size=inputNumber()
                    print('Enter the amount of generations: ')
                    max_gens=inputNumber()
                    print('Enter the algorithm to use (0 - Enchanced | 1 - Standard): ')
                    algorithm=inputNumber(lower_or_eq=1)
                    if algorithm==0{
                        print('Enter the genome max age: ')
                        max_age=inputNumber(greater_or_eq=1)
                        print('Enter the max amount of children at once: ')
                        max_children=inputNumber(greater_or_eq=2)
                        print('Enter the recycle rate: ')
                        recycle_rate=inputNumber(is_float=True,lower_or_eq=1)
                    }else{
                        max_age=0
                        max_children=0
                        recycle_rate=0.0
                    }
                    print('Enter the mutation rate: ')
                    mutation_rate=inputNumber(is_float=True,lower_or_eq=1)
                    print('Enter the sex rate: ')
                    sex_rate=inputNumber(is_float=True,lower_or_eq=1)
                    print('Enter the max amount of notable individuals at the Hall Of Fame: ')
                    max_notables=inputNumber()
                    print('Enter the cross validation method (0-3):')
                    print('\t0 - NONE')
                    print('\t1 - ROLLING_FORECASTING_ORIGIN')
                    print('\t2 - KFOLDS')
                    print('\t3 - TWENTY_PERCENT')
                    print('value: ')
                    cross_validation=inputNumber(lower_or_eq=3)
                    print('Enter the metric to be used (0-4):')
                    print('\t0 - RAW_LOSS (cheaper)')
                    print('\t1 - F1')
                    print('\t2 - RECALL')
                    print('\t3 - ACCURACY')
                    print('\t4 - PRECISION')
                    print('value: ')
                    metric=inputNumber(lower_or_eq=4)
                    print('Enter the label type (0-2):')
                    print('\t0 - INT_CLASS')
                    print('\t1 - NEURON_BY_NEURON')
                    print('\t2 - NEURON_BY_NEURON_LOG_LOSS')
                    print('value: ', end='')
                    label_type=inputNumber(lower_or_eq=2)
                    print('Enter the years to be used as train data splitted by comma (,) (1999-2020):')
                    train_data=inputArrayNumber(greater_or_eq=1999,lower_or_eq=2020)
                    print('Enter a limit of CVEs for each year (0 = unlimitted): ')
                    train_data+=':{}'.format(inputNumber())
                    best={'output':None,'at_gen':None}
                    results=[]
                    simulation_data={'name':simulation_name,'env_name':env_name,'submitted_at':submitted_at,'started_by':started_by,'started_at':started_at,'finished_at':finished_at,'hall_of_fame_id':hall_of_fame_id,'population_id':population_id,'pop_start_size':pop_start_size,'max_gens':max_gens,'algorithm':algorithm,'max_age':max_age,'max_children':max_children,'mutation_rate':mutation_rate,'recycle_rate':recycle_rate,'sex_rate':sex_rate,'max_notables':max_notables,'cross_validation':cross_validation,'label_type':label_type,'metric':metric,'train_data':train_data,'best':best,'results':results}
                    LOGGER.info('Writting simulation config on genetic_db...')
                    simulation_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('genetic_db'),simulation_data,'simulations')
                    if (simulation_id==None){
                        LOGGER.error('Failed to insert simulation!')
                    }else{
                        LOGGER.info('Wrote simulation config on genetic_db...OK')
                        halloffame_data={'simulation_id':simulation_id,'env_name':env_name,'updated_at':None,'neural_genomes':[]}
                        halloffame_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),halloffame_data,'hall_of_fame')
                        generation_data={'simulation_id':simulation_id,'env_name':env_name,'updated_at':None,'neural_genomes':[]}
                        population_id=mongo.quickInsertOneIgnoringLockAndRetrieveId(mongo.getDB('neural_db'),generation_data,'populations')
                        if halloffame_id == None or population_id == None{
                            LOGGER.error('Failed to insert hall of fame or/and generation!')
                        }else{
                            query={'_id':bson.ObjectId(simulation_id)}
                            update={'$set':{'hall_of_fame_id':halloffame_id,'population_id':population_id}}
                            mongo.getDB('genetic_db')['simulations'].find_one_and_update(query,update)

                            LOGGER.info('Writting on Core queue to run genetic simulation...')
                            job_args={'simulation_id':simulation_id}
                            mongo.insertOnCoreQueue('Genetic',job_args,priority=3)
                            LOGGER.info('Wrote on Core queue to run genetic simulation...OK')
                        }
                    }
                }
            }elif opt == "--show-genetic-results"{
                LOGGER.info('Getting genetic results...')
                for sim in mongo.findAllOnDB(mongo.getDB('genetic_db'),'simulations',wait_unlock=False){
                    for k,v in sim.items(){
                        if k=="hall_of_fame_id"{
                            hall=mongo.findOneOnDBFromIndex(mongo.getDB('neural_db'),'hall_of_fame','_id',bson.ObjectId(v),wait_unlock=False)
                            LOGGER.clean('Hall Of Fame:')
                            if (len( hall['neural_genomes'])==0){
                                LOGGER.clean('\t[]')
                            }else{
                                for el in hall['neural_genomes']{
                                    LOGGER.clean('\t{}'.format(str(el))) 
                                }
                            }
                        }elif k=="population_id"{
                            pop=mongo.findOneOnDBFromIndex(mongo.getDB('neural_db'),'populations','_id',bson.ObjectId(v),wait_unlock=False)
                            LOGGER.clean('Population:')
                            if (len(pop['neural_genomes'])==0){
                                LOGGER.clean('\t[]')
                            }else{
                                for el in pop['neural_genomes']{
                                    LOGGER.clean('\t{}:'.format(str(el))) 
                                }
                            }
                        }else{
                             if type(v) is list{
                                if (len(v)==0){
                                    LOGGER.clean('{}: []'.format(k))
                                }else{
                                    LOGGER.clean('{}:'.format(k))
                                    for el in v{
                                        LOGGER.clean('\t{}:'.format(str(el))) 
                                    }
                                }
                            }else{
                                LOGGER.clean('{}: {}'.format(k,str(v)))
                            }
                        }
                    }
                    LOGGER.clean('\n')
                }
                LOGGER.info('Gotten genetic results...OK')
            }elif opt == "--create-genetic-env"{
                print('Now type the minimum and maximums for each item of the Smart Neural Search Space...')
                print('Enter the genetic environment name: ', end = '')
                gen_name=input().strip()
                submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                print('Enter the amount of layers: min: ')
                amount_of_layers_min=inputNumber()
                print("Max: ")
                amount_of_layers_max=inputNumber(greater_or_eq=amount_of_layers_min)
                print('Enter the epochs: min: ')
                epochs_min=inputNumber()
                print("Max: ")
                epochs_max=inputNumber(greater_or_eq=epochs_min)
                print('Enter the alpha: min: ')
                alpha_min=inputNumber(is_float=True,lower_or_eq=1)
                print("Max: ")
                alpha_max=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=alpha_min)
                print('Enter the batch size: min: ')
                batch_size_min=inputNumber()
                print("Max: ")
                batch_size_max=inputNumber(greater_or_eq=batch_size_min)
                print('Enter the layer sizes: min: ')
                layer_size_min=inputNumber()
                print("Max: ")
                layer_size_max=inputNumber(greater_or_eq=layer_size_min)
                print('Enter the range pow: min: ')
                range_pow_min=inputNumber()
                print("Max: ")
                range_pow_max=inputNumber(greater_or_eq=range_pow_min)
                print('Enter the K values: min: ')
                k_min=inputNumber()
                print("Max: ")
                k_max=inputNumber(greater_or_eq=k_min)
                print('Enter the L values: min: ')
                l_min=inputNumber()
                print("Max: ")
                l_max=inputNumber(greater_or_eq=l_min)
                print('Enter the sparcity: min: ')
                sparcity_min=inputNumber(is_float=True,lower_or_eq=1)
                print("Max: ")
                sparcity_max=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=sparcity_min)
                print('Enter the activation functions (0-2):')
                print('\t0 - ReLU')
                print('\t1 - Softmax')
                print('\t2 - Sigmoid')
                print('min: ')
                activation_min=inputNumber(lower_or_eq=2)
                print("Max: ")
                activation_max=inputNumber(lower_or_eq=2,greater_or_eq=activation_min)
                space_search={'name':gen_name,'submitted_at':submitted_at,'space_search':{'amount_of_layers':{'min':amount_of_layers_min,'max':amount_of_layers_max},'epochs':{'min':epochs_min,'max':epochs_max},'batch_size':{'min':batch_size_min,'max':batch_size_max},'layer_sizes':{'min':layer_size_min,'max':layer_size_max},'range_pow':{'min':range_pow_min,'max':range_pow_max},'K':{'min':k_min,'max':k_max},'L':{'min':l_min,'max':l_max},'activation_functions':{'min':activation_min,'max':activation_max},'sparcity':{'min':sparcity_min,'max':sparcity_max},'alpha':{'min':alpha_min,'max':alpha_max}}}
                LOGGER.info('Writting environment on genetic_db...')
                mongo.insertOneOnDB(mongo.getDB('genetic_db'),space_search,'environments',index='name',ignore_lock=True)
                LOGGER.info('Wrote environment on genetic_db...OK')
            }elif opt == "--rm-genetic-env"{
                arg=arg.strip()
                LOGGER.info('Removing {} from genetic environments...'.format(arg))
                query={'name':arg}
                mongo.rmOneFromDB(mongo.getDB('genetic_db'),'environments',query=query)
                LOGGER.info('Removed {} from genetic environments...OK'.format(arg))
            }elif opt == "--rm-smart-neural-hyperparams"{
                arg=arg.strip()
                LOGGER.info('Removing {} from hyperparameters...'.format(arg))
                query={'name':arg}
                mongo.rmOneFromDB(mongo.getDB('neural_db'),'snn_hyperparameters',query=query)
                LOGGER.info('Removed {} from genetic environments...OK'.format(arg))
            }elif opt == "--run-processor-pipeline"{
                LOGGER.info('Writting on Processor to Run the entire Pipeline...')
                mongo.insertOnProcessorQueue('Run Pipeline')
                LOGGER.info('Wrote on Processor to Run the entire Pipeline...OK')
            }elif opt == "--run-merge-cve"{ 
                LOGGER.info('Writting on Processor to Run Merge CVE step...')
                mongo.insertOnProcessorQueue('Merge')
                LOGGER.info('Wrote on Processor to  Run Merge CVE step...OK')
            }elif opt == "--run-flattern-and-simplify-all"{
                LOGGER.info('Writting on Processor to Flattern and Simplify all types of data...')
                job_args={'type':'CVE'}
                mongo.insertOnProcessorQueue('Flattern and Simplify',job_args)
                job_args={'type':'OVAL'}
                mongo.insertOnProcessorQueue('Flattern and Simplify',job_args)
                job_args={'type':'CAPEC'}
                mongo.insertOnProcessorQueue('Flattern and Simplify',job_args)
                job_args={'type':'CWE'}
                mongo.insertOnProcessorQueue('Flattern and Simplify',job_args)
                LOGGER.info('Wrote on Processor to Flattern and Simplify all types of data...OK')
            }elif opt == "--run-flattern-and-simplify"{
                if arg.lower() in ('cve','oval','capec','cwe'){
                    job_args={'type':arg.upper()}
                    LOGGER.info('Writting on Processor to Flattern and Simplify {}...'.format(arg))
                    mongo.insertOnProcessorQueue('Flattern and Simplify',job_args)
                    LOGGER.info('Wrote on Processor to Flattern and Simplify {}...'.format(arg))
                }else{
                    LOGGER.error('Invalid argument, arg must be cve, oval, capec or cwe')
                }
            }elif opt == "--run-filter-exploits"{
                LOGGER.info('Writting on Processor to Filter Exploits...')
                mongo.insertOnProcessorQueue('Filter Exploits')
                LOGGER.info('Wrote on Processor to Filter Exploits...OK')
            }elif opt == "--run-transform-all"{
                LOGGER.info('Writting on Processor to Transform all types of data...')
                job_args={'type':'CVE'}
                mongo.insertOnProcessorQueue('Transform',job_args)
                job_args={'type':'OVAL'}
                mongo.insertOnProcessorQueue('Transform',job_args)
                job_args={'type':'CAPEC'}
                mongo.insertOnProcessorQueue('Transform',job_args)
                job_args={'type':'CWE'}
                mongo.insertOnProcessorQueue('Transform',job_args)
                job_args={'type':'EXPLOITS'}
                mongo.insertOnProcessorQueue('Transform',job_args)
                LOGGER.info('Wrote on Processor to Transform all types of data...OK')
            }elif opt == "--run-transform"{
                if arg.lower() in ('cve','oval','capec','cwe','exploits'){
                    job_args={'type':arg.upper()}
                    LOGGER.info('Writting on Processor to Transform {}...'.format(arg))
                    mongo.insertOnProcessorQueue('Transform',job_args)
                    LOGGER.info('Wrote on Processor to Transform {}...'.format(arg))
                }else{
                    LOGGER.error('Invalid argument, arg must be cve, oval, capec or cwe')
                }
            }elif opt == "--run-enrich"{
                LOGGER.info('Writting on Processor to Enrich Data...')
                mongo.insertOnProcessorQueue('Enrich')
                LOGGER.info('Wrote on Processor to Enrich Data...OK')
            }elif opt == "--run-analyze"{
                LOGGER.info('Writting on Processor to Analyze Data...')
                mongo.insertOnProcessorQueue('Analyze')
                LOGGER.info('Wrote on Processor to Analyze Data...OK')
            }elif opt == "--run-filter-and-normalize"{
                LOGGER.info('Writting on Processor to Filter and Normalize Data...')
                mongo.insertOnProcessorQueue('Filter and Normalize')
                LOGGER.info('Wrote on Processor to Filter and Normalize Data...OK')
            }elif opt == "--restore-db"{
                splited_arg=arg.split('#')
                if len(splited_arg)>0 and len(splited_arg)<=2{
                    db_name=None
                    if len(splited_arg)>1{
                        db_name=splited_arg[1]
                    }
                    mongo.restoreDB(splited_arg[0],db_name)
                }else{
                    LOGGER.error('Invalid argument, type the compressed_file_path or "compressed_file_path#db_name": {}'.format(arg))
                }
            }elif opt == "--empty-queue"{
                LOGGER.info('Erasing queue {}...'.format(arg))
                mongo.clearQueue(arg)
                LOGGER.info('Erased queue {}...OK'.format(arg))
            }elif opt == "--empty-all-queues"{
                LOGGER.info('Erasing all queues...')
                for queue in mongo.getQueueNames(){
                    LOGGER.info('Erasing queue {}...'.format(queue))
                    mongo.clearQueue(queue)
                    LOGGER.info('Erased queue {}...OK'.format(queue))
                }
                LOGGER.info('Erased all queues...OK')
            }elif opt == "--get-queue-names"{
                LOGGER.info('Getting queue names...')
                for queue in mongo.getQueueNames(){
                    LOGGER.clean('\t{}'.format(queue))
                }
                LOGGER.info('Gotten queue names...OK')
            }elif opt == "--get-all-db-names"{
                LOGGER.info('Getting db names...')
                for db in mongo.getAllDbNames(){
                    LOGGER.clean('\t{}'.format(db))
                }
                LOGGER.info('Gotten db names...OK')
            }elif opt == "--download-source"{
                LOGGER.info('Writting on Crawler to Download {}...'.format(arg))
                job_args={'id':arg}
                mongo.insertOnCrawlerQueue('Download',job_args)
                LOGGER.info('Wrote on Crawler to Download {}...OK'.format(arg))
            }elif opt == "--download-all-sources"{
                LOGGER.info('Writting on Crawler to Download All Sources...')
                mongo.insertOnCrawlerQueue('DownloadAll')
                LOGGER.info('Wrote on Crawler to Download All Sources...OK')
            }elif opt == "--check-jobs"{
                LOGGER.info('Checking jobs on Queue...')
                queues=mongo.getAllQueueJobs()
                for queue,jobs in queues.items(){
                    LOGGER.clean('Under \'{}\' queue:'.format(queue))
                    for job in jobs{
                        for k,v in job.items(){
                            tab='\t'
                            if k!='task'{
                                tab+=tab
                            }
                            LOGGER.clean('{}{}: {}'.format(tab,k.capitalize(),v))
                        }
                    }
                }
                LOGGER.info('Checked jobs on Queue...OK')
            }
        }
    } except Exception as e{
        if ITERATIVE {
            print(e)
        }else{
            raise e
        }
    }

    if zombie{
        LOGGER.info('Keeping Front end alive on Zombie Mode...')
        LOGGER.info('Tip: Use `front-end` command on container to run commands.')
        while(True){
            pass
        }
    }else{
        print()
        ITERATIVE=True
        value = input("Keeping Front end alive on Iterative Mode...\nEnter a command (e.g. -h):")
        args=value.split(' ')
        args[0]=args[0].strip()
        if len(args[0])==1{
            if not args[0].startswith('-'){
                args[0]='-{}'.format(args[0])
            }
        }else{
            if not args[0].startswith('--'){
                args[0]='--{}'.format(args[0])
            }
        }
        main(args)
    }
}


if __name__ == "__main__"{
    LOGGER.info('Starting Front end...')
    if Utils.runningOnDockerContainer(){
        mongo_addr='mongo'
    }else{
        mongo_addr='127.0.0.1'
    }
    mongo=MongoDB(mongo_addr,27017,LOGGER,user='root',password='123456')
    mongo.startQueue(id=0)
    LOGGER.info('Started Front end...OK')
    LOGGER.info('Writting on queue as {}'.format(mongo.getQueueConsumerId()))
    main(sys.argv[1:])
}
