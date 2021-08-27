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
    HELP_STR='main.py [-h]\n\t[--check-jobs]\n\t[--create-genetic-env]\n\t[--list-genetic-envs]\n\t[--run-genetic]\n\t[--show-genetic-results]\n\t[--rm-genetic-env <env name>]\n\t[--parse-dna-to-hyperparams]\n\t[--create-neural-hyperparams]\n\t[--list-neural-hyperparams]\n\t[--rm-neural-hyperparams <hyper name>]\n\t[--train-neural]\n\t[--eval-neural]\n\t[--get-queue-names]\n\t[--get-all-db-names]\n\t[-q | --quit]\n\t[--run-processor-pipeline]\n\t[--run-merge-cve]\n\t[--run-flattern-and-simplify-all]\n\t[--run-flattern-and-simplify [cve|oval|capec|cwe]]\n\t[--run-filter-exploits]\n\t[--run-transform-all]\n\t[--run-transform [cve|oval|capec|cwe|exploits]]\n\t[--run-enrich]\n\t[--run-analyze]\n\t[--run-filter-and-normalize]\n\t[--download-source <source ID>]\n\t[--download-all-sources]\n\t[--empty-queue <queue name>]\n\t[--empty-all-queues]\n\t[--count-features]\n\t[--dump-db <db name>#<folder path to export> | --dump-db <db name> {saves on default tmp folder} \n\t\te.g. --dump-db queue#/home/thiago/Desktop]\n\t[--restore-db <file path to import>#<db name> | --restore-db <file path to import> {saves db under file name} \n\t\te.g. --restore-db /home/thiago/Desktop/queue.zip#restored_queue]\n\t[--keep-alive-as-zombie]'
    args=[]
    zombie=False
    global ITERATIVE
    to_run=[]
    try{ 
        opts, args = getopt.getopt(argv,"hq",["keep-alive-as-zombie","download-source=","download-all-sources","check-jobs","quit","get-queue-names","empty-queue=","empty-all-queues","get-all-db-names","dump-db=","restore-db=","run-processor-pipeline","run-flattern-and-simplify-all","run-flattern-and-simplify=","run-filter-exploits","run-transform-all","run-transform=","run-enrich","run-analyze","run-filter-and-normalize","run-merge-cve","create-genetic-env","list-genetic-envs","rm-genetic-env=","run-genetic","show-genetic-results","create-neural-hyperparams","list-neural-hyperparams","rm-neural-hyperparams=","train-neural","eval-neural","parse-dna-to-hyperparams","count-features"])
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
            }elif opt == "--count-features"{
                processed_db=mongo.getProcessedDB()
                query={}
                cve=mongo.findOneOnDB(processed_db,'dataset',query)
                if cve is not None{
                    total=0
                    cvss_enum=0
                    description=0
                    reference=0
                    vendor=0
                    other=0
                    for k,_ in cve['features'].items(){
                        total+=1
                        if 'cvss_' in k and '_ENUM_' in k {
                            cvss_enum+=1
                        }elif 'description_' in k {
                            description+=1
                        }elif 'reference_' in k {
                            reference+=1
                        }elif 'vendor_' in k {
                            vendor+=1
                        }else{
                            other+=1
                        }
                    }
                    LOGGER.info('Total features on dataset: {}'.format(total))
                    LOGGER.info('')
                    LOGGER.info('Grouped features count:')
                    LOGGER.info('\tCVSS/ENUM features: {}'.format(cvss_enum))
                    LOGGER.info('\tDescription features: {}'.format(description))
                    LOGGER.info('\tReference features: {}'.format(reference))
                    LOGGER.info('\tVendor features: {}'.format(vendor))
                    LOGGER.info('\tOther features: {}'.format(other))
                }else{
                    LOGGER.warn('No dataset found...')
                }
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
            }elif opt == "--list-neural-hyperparams"{
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
                print('Select your core version (1-2):')
                print('\t1 - C++')
                print('\t2 - Pytho{\}')
                version=inputNumber(greater_or_eq=1,lower_or_eq=2)
                print()
                if version==2{
                    LOGGER.info('Casting DNA...')
                    print('Enter the DNA (e.g. [ 10 0.001 False 10 80 0 1 1 1 True 1 0 2 0.1 True ]): ', end = '')
                    not_filled=True
                    dna=''
                    while not_filled {
                        dna=input().strip().replace('[','',1).replace(']','',1).strip()
                        if re.match(r'^(:?([0-9\.]|True|False)* ?)*$',dna){
                            not_filled=False
                        }else{
                            print('ERROR - Wrong DNA format')
                        }
                    }
                    dna=dna.split()
                    for s,gene in enumerate(dna){
                        if '.' in gene {
                            dna[s]=float(gene)
                        }elif any(c.isalpha() for c in gene){
                            dna[s]=gene.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
                        }else{
                            dna[s]=int(gene)
                        }
                    }
                    is_ok=True
                    if type(dna[2]) is bool { # single network
                        batch_size=int(dna[0])
                        alpha=float(dna[1])
                        shuffle=bool(dna[2])
                        patience_epochs=int(dna[3])
                        max_epochs=int(dna[4])
                        loss=int(dna[5])
                        label_type=int(dna[6])

                        optimizer=int(dna[7])
                        monitor_metric=int(dna[8])
                        model_checkpoint=bool(dna[9])

                        layers=int(dna[10])
                        first_layer_dependent=11
                        layer_sizes=[]
                        node_types=[]
                        dropouts=[]
                        bias=[]
                        amount_of_dependent=4
                        for l in range(layers){
                            layer_sizes.append(int(dna[(first_layer_dependent+0)+amount_of_dependent*l]))
                            node_types.append(int(dna[(first_layer_dependent+1)+amount_of_dependent*l]))
                            dropouts.append(float(dna[(first_layer_dependent+2)+amount_of_dependent*l]))
                            bias.append(bool(dna[(first_layer_dependent+3)+amount_of_dependent*l]))
                        }
                        layer_sizes[-1]='Output Size'
                        node_types[-1]='NodeType.SIGMOID(2)'
                        LOGGER.info('amount_of_networks: {}'.format(1))
                        LOGGER.info('batch_size: {}'.format(batch_size))
                        LOGGER.info('alpha: {}'.format(alpha))
                        LOGGER.info('shuffle: {}'.format(shuffle))
                        LOGGER.info('max_epochs: {}'.format(max_epochs))
                        LOGGER.info('patience_epochs: {}'.format(patience_epochs))
                        LOGGER.info('loss: {}'.format(loss))
                        LOGGER.info('label_type: {}'.format(label_type))
                        LOGGER.info('optimizer: {}'.format(optimizer))
                        LOGGER.info('monitor_metric: {}'.format(monitor_metric))
                        LOGGER.info('model_checkpoint: {}'.format(model_checkpoint))
                        LOGGER.info('layers: {}'.format(layers))
                        LOGGER.info('layer_sizes: {}'.format(str(layer_sizes)))
                        LOGGER.info('node_types: {}'.format(str(node_types)))
                        LOGGER.info('dropouts: {}'.format(str(dropouts)))
                        LOGGER.info('bias: {}'.format(str(bias)))
                    }elif type(dna[5]) is bool { # multi network
                        batch_size=int(dna[0])
                        patience_epochs=int(dna[1])
                        max_epochs=int(dna[2])
                        label_type=int(dna[3])
                        monitor_metric=int(dna[4])
                        model_checkpoint=bool(dna[5])
                        shuffle=bool(dna[6])

                        networks=int(dna[7])
                        last_index=8
                        alpha=[]
                        loss=[]
                        optimizer=[]
                        layers=[]
                        layer_sizes=[]
                        node_types=[]
                        dropouts=[]
                        bias=[]
                        network_parameters=9
                        layer_parameters=4
                        offset=0
                        for n in range(networks){
                            alpha.append(float(dna[(last_index+0)+network_parameters*n+offset]))
                            loss.append(int(dna[(last_index+1)+network_parameters*n+offset]))
                            optimizer.append(int(dna[(last_index+2)+network_parameters*n+offset]))
                            layers.append(int(dna[(last_index+3)+network_parameters*n+offset]))
                            max_layers=int(dna[(last_index+4)+network_parameters*n+offset])
                            layer_sizes.append([])
                            node_types.append([])
                            dropouts.append([])
                            bias.append([])
                            for l in range(layers[-1]){
                                layer_sizes[-1].append(int(dna[(last_index+5)+network_parameters*n+offset]))
                                node_types[-1].append(int(dna[(last_index+6)+network_parameters*n+offset]))
                                dropouts[-1].append(float(dna[(last_index+7)+network_parameters*n+offset]))
                                bias[-1].append(bool(dna[(last_index+8)+network_parameters*n+offset]))
                                offset+=layer_parameters
                            }
                            offset+=(max_layers-layers[-1]-1)*layer_parameters
                        }
                        layer_sizes[-1][-1]='Output Size'
                        node_types[-1][-1]='NodeType.SIGMOID(2)'
                        LOGGER.info('amount_of_networks: {}'.format(networks))
                        LOGGER.info('batch_size: {}'.format(batch_size))
                        LOGGER.info('max_epochs: {}'.format(max_epochs))
                        LOGGER.info('patience_epochs: {}'.format(patience_epochs))
                        LOGGER.info('label_type: {}'.format(label_type))
                        LOGGER.info('monitor_metric: {}'.format(monitor_metric))
                        LOGGER.info('model_checkpoint: {}'.format(model_checkpoint))
                        LOGGER.info('shuffle: {}'.format(shuffle))
                        for n in range(networks){
                            if n!=networks-1{
                                LOGGER.info('Network {} of {}'.format(n+1,networks))
                            }else{
                                LOGGER.info('Concatenation Network')
                            }
                            LOGGER.info('\talpha: {}'.format(str(alpha[n])))
                            LOGGER.info('\tloss: {}'.format(str(loss[n])))
                            LOGGER.info('\toptimizer: {}'.format(str(optimizer[n])))
                            LOGGER.info('\tlayers: {}'.format(str(layers[n])))
                            LOGGER.info('\tlayer_sizes: {}'.format(str(layer_sizes[n])))
                            LOGGER.info('\tnode_types: {}'.format(str(node_types[n])))
                            LOGGER.info('\tdropouts: {}'.format(str(dropouts[n])))
                            LOGGER.info('\tbias: {}'.format(str(bias[n])))
                        }
                    }else{
                        is_ok=False
                        LOGGER.error('Unkown DNA format!')
                        LOGGER.error('Cast DNA...FAIL')
                    }
                    if is_ok {
                        LOGGER.info('Cast DNA...OK')
                    }
                }else{
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
                }
            }elif opt == "--create-neural-hyperparams"{
                print('Select your core version (1-2):')
                print('\t1 - C++')
                print('\t2 - Pytho{\}')
                version=inputNumber(greater_or_eq=1,lower_or_eq=2)
                print()
                if version==2{
                    print('Use multiple networks (0 [False] - 1 [True]):')
                    multiple_networks=inputNumber(lower_or_eq=1)==1
                    if multiple_networks{
                        print('Now type the hyperparameters for the Neural Networks...')
                        print('Enter the hyperparameters config name (unique): ', end = '')
                        hyper_name=input().strip()
                        submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                        print('Enter the batch size: ', end = '')
                        batch_size=inputNumber()
                        print('Enter the label type ([0-1]+[3-8]):')
                        print('\t0 - INCREMENTAL')
                        print('\t1 - BINARY')
                        # print('\t2 - NEURON_BY_NEURON_LOG_LOSS DEPRECATED')
                        print('\t3 - BINARY_PLUS_ONE')
                        print('\t4 - SPARSE')
                        print('\t5 - DISTINCT_SPARSE')
                        print('\t6 - DISTINCT_SPARSE_PLUS_ONE')
                        print('\t7 - INCREMENTAL_PLUS_ONE')
                        print('\t8 - EXPONENTIAL')
                        print('value: ', end='')
                        label_type=inputNumber(lower_or_eq=8)
                        while label_type==2{
                            print('Label type 2 - NEURON_BY_NEURON_LOG_LOSS is deprecated, try another number:')
                            label_type=inputNumber(lower_or_eq=8)
                        }
                        print('Enter shuffle train data (0 [False] - 1 [True]): ', end = '')
                        shuffle=inputNumber(lower_or_eq=1)==1
                        print()
                        print('We\'ll use 5 networks for each group of feature and one final network to concatenate every other, now we\'ll define the parameters for them')
                        network_names=['Main features','CVSS ENUM features','Description features','Reference Features','Vendor Features','Concatenation']
                        amount_of_networks=len(network_names)

                        use_same_alpha=None
                        alpha=[None for _ in range(len(network_names))]
                        use_same_optimizer=None
                        optimizer=[None for _ in range(len(network_names))]
                        use_same_loss=None
                        loss=[None for _ in range(len(network_names))]
                        layers=[None for _ in range(len(network_names))]
                        layer_sizes=[None for _ in range(len(network_names))]
                        node_types=[None for _ in range(len(network_names))]
                        dropouts=[None for _ in range(len(network_names))]
                        bias=[None for _ in range(len(network_names))]
                        for n in range(len(network_names)){
                            print('Now enter data regarding the {} network'.format(network_names[n]))
                            if use_same_alpha is None {
                                print('Should we use the same alpha for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_alpha=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_alpha or alpha[0] is None {
                                print('Enter the alpha: ', end = '')
                                alpha[n]=inputNumber(is_float=True,lower_or_eq=1)
                            }else{
                                alpha[n]=alpha[0]
                            }
                            if use_same_optimizer is None {
                                print('Should we use the same optmizer for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_optimizer=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_optimizer or optimizer[0] is None {
                                print('Enter the optimizer (0-2): ')
                                print('\t0 - SGD')
                                print('\t1 - Adam')
                                print('\t2 - RMSProp')
                                optimizer[n]=inputNumber(lower_or_eq=2)
                            }else{
                                optimizer[n]=optimizer[0]
                            }
                            if use_same_loss is None {
                                print('Should we use the same loss for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_loss=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_loss or loss[0] is None {
                                print('Enter the loss function (0-3):')
                                print('\t0 - Binary Crossentropy')
                                print('\t1 - Categorical Crossentropy')
                                print('\t2 - Mean Squared Error')
                                print('\t3 - Mean Absolute Error')
                                print('value: ')
                                loss[n]=inputNumber(lower_or_eq=3)
                            }else{
                                loss[n]=loss[0]
                            }
                            print('Enter amount of layers: ', end = '')
                            layers[n]=inputNumber(greater_or_eq=1)
                            tmp_layer_sizes=[]
                            if n!=len(network_names)-1{
                                for i in range(layers[n]){
                                    print('Enter the layer size for layer {}: '.format(i), end = '')
                                    tmp_layer_sizes.append(inputNumber())
                                }
                            }else{
                                for i in range(layers[n]-1){
                                    print('Ignoring last layer, its size is set on core...')
                                    print('Enter the layer size for layer {}: '.format(i), end = '')
                                    tmp_layer_sizes.append(inputNumber())
                                }
                                tmp_layer_sizes.append(0) # output layer
                            }
                            layer_sizes[n]=tmp_layer_sizes
                            print('Enter the nodes activation functions (0-9):')
                            print('\t0 - ReLU')
                            print('\t1 - Softmax')
                            print('\t2 - Sigmoid')
                            print('\t3 - Tanh')
                            print('\t4 - Softplus')
                            print('\t5 - Softsign')
                            print('\t6 - Selu')
                            print('\t7 - Elu')
                            print('\t8 - Exponential')
                            print('\t9 - Linear')
                            tmp_node_types=[]
                            if n!=len(network_names)-1{
                                for i in range(layers[n]){
                                    print('Enter the node type for layer {}: '.format(i), end = '')
                                    tmp_node_types.append(inputNumber(lower_or_eq=9))
                                }
                            }else{
                                for i in range(layers[n]-1){
                                    print('Enter the node type for layer {}: '.format(i), end = '')
                                    tmp_node_types.append(inputNumber(lower_or_eq=9))
                                }
                                print('Enter the node type for the OUTPUT layer (recommended softmax or sigmoid): ', end = '')
                                tmp_node_types.append(inputNumber(lower_or_eq=9))
                            }
                            node_types[n]=tmp_node_types
                            print('Should we use the same dropouts for every layer of this network instead of specify one for each? (0 [False] - 1 [True]):')
                            use_same_dropouts=inputNumber(lower_or_eq=1)==1
                            tmp_dropouts=[]
                            for i in range(layers[n]){
                                if not use_same_dropouts or len(tmp_dropouts)==0{
                                    print('Enter the dropout for layer {}: '.format(i), end = '')
                                    tmp_dropouts.append(inputNumber(is_float=True,lower_or_eq=1))
                                }else{
                                    tmp_dropouts.append(tmp_dropouts[0])
                                }
                            }
                            dropouts[n]=tmp_dropouts
                            print('Should we use the same bias for every layer of this network instead of specify one for each? (0 [False] - 1 [True]):')
                            use_same_bias=inputNumber(lower_or_eq=1)==1
                            tmp_bias=[]
                            for i in range(layers[n]){
                                if not use_same_bias or len(tmp_bias)==0{
                                    print('Enter use bias for layer {} (0 [False] - 1 [True]): '.format(i), end = '')
                                    tmp_bias.append(inputNumber(lower_or_eq=1)==1)
                                }else{
                                    tmp_bias.append(tmp_bias[0])
                                }
                            }
                            bias[n]=tmp_bias

                            if n!=len(network_names)-1{
                                print()
                            }
                        }
                    }else{
                        amount_of_networks=1
                        print('Now type the hyperparameters for the Neural Network...')
                        print('Enter the hyperparameters config name (unique): ', end = '')
                        hyper_name=input().strip()
                        submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                        print('Enter the batch size: ', end = '')
                        batch_size=inputNumber()
                        print('Enter the alpha: ', end = '')
                        alpha=inputNumber(is_float=True,lower_or_eq=1)
                        print('Enter shuffle train data (0 [False] - 1 [True]): ', end = '')
                        shuffle=inputNumber(lower_or_eq=1)==1
                        print('Enter the optimizer (0-2): ')
                        print('\t0 - SGD')
                        print('\t1 - Adam')
                        print('\t2 - RMSProp')
                        optimizer=inputNumber(lower_or_eq=2)
                        print('Enter the loss function (0-3):')
                        print('\t0 - Binary Crossentropy')
                        print('\t1 - Categorical Crossentropy')
                        print('\t2 - Mean Squared Error')
                        print('\t3 - Mean Absolute Error')
                        print('value: ')
                        loss=inputNumber(lower_or_eq=3)
                        print('Enter the label type ([0-1]+[3-8]):')
                        print('\t0 - INCREMENTAL')
                        print('\t1 - BINARY')
                        # print('\t2 - NEURON_BY_NEURON_LOG_LOSS DEPRECATED')
                        print('\t3 - BINARY_PLUS_ONE')
                        print('\t4 - SPARSE')
                        print('\t5 - DISTINCT_SPARSE')
                        print('\t6 - DISTINCT_SPARSE_PLUS_ONE')
                        print('\t7 - INCREMENTAL_PLUS_ONE')
                        print('\t8 - EXPONENTIAL')
                        print('value: ', end='')
                        label_type=inputNumber(lower_or_eq=8)
                        while label_type==2{
                            print('Label type 2 - NEURON_BY_NEURON_LOG_LOSS is deprecated, try another number:')
                            label_type=inputNumber(lower_or_eq=8)
                        }
                        print('Enter amount of layers: ', end = '')
                        layers=inputNumber(greater_or_eq=1)
                        layer_sizes=[]
                        for i in range(layers-1){
                            print('Ignoring last layer, its size is set on core...')
                            print('Enter the layer size for layer {}: '.format(i), end = '')
                            layer_sizes.append(inputNumber())
                        }
                        layer_sizes.append(0) # output layer
                        print('Enter the nodes activation functions (0-9):')
                        print('\t0 - ReLU')
                        print('\t1 - Softmax')
                        print('\t2 - Sigmoid')
                        print('\t3 - Tanh')
                        print('\t4 - Softplus')
                        print('\t5 - Softsign')
                        print('\t6 - Selu')
                        print('\t7 - Elu')
                        print('\t8 - Exponential')
                        print('\t9 - Linear')
                        node_types=[]
                        for i in range(layers-1){
                            print('Enter the node type for layer {}: '.format(i), end = '')
                            node_types.append(inputNumber(lower_or_eq=9))
                        }
                        print('Enter the node type for the OUTPUT layer (recommended softmax or sigmoid): ', end = '')
                        node_types.append(inputNumber(lower_or_eq=9))
                        dropouts=[]
                        for i in range(layers){
                            print('Enter the dropout for layer {}: '.format(i), end = '')
                            dropouts.append(inputNumber(is_float=True,lower_or_eq=1))
                        }
                        bias=[]
                        for i in range(layers){
                            print('Enter use bias for layer {} (0 [False] - 1 [True]): '.format(i), end = '')
                            bias.append(inputNumber(lower_or_eq=1)==1)
                        }
                    }
                    hyperparams={'core_version':'v2','name':hyper_name,'submitted_at':submitted_at,'amount_of_networks':amount_of_networks,'batch_size':batch_size,'alpha':alpha,'shuffle':shuffle,'optimizer':optimizer,'loss':loss,'label_type':label_type,'layers':layers,'layer_sizes':layer_sizes,'bias':bias,'node_types':node_types,'dropouts':dropouts}
                }else{ # v1
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
                        print('Enter the node type for layer {}: '.format(i+1), end = '')
                        node_types.append(inputNumber(lower_or_eq=9))
                    }
                    node_types.append(1) # output layer softmax/sigmoid
                    sparcity=[1] # border
                    for i in range(layers-2){
                        print('Enter the sparcity for layer {}: '.format(i), end = '')
                        sparcity.append(inputNumber(is_float=True,lower_or_eq=1))
                    }
                    if layers > 1 {
                        sparcity.append(1); # border
                    }
                    hyperparams={'core_version':'v1','name':hyper_name,'submitted_at':submitted_at,'batch_size':batch_size,'alpha':alpha,'shuffle':shuffle,'adam':adam,'rehash':rehash,'rebuild':rebuild,'label_type':label_type,'layers':layers,'layer_sizes':layer_sizes,'range_pow':range_pow,'K':K,'L':L,'node_types':node_types,'sparcity':sparcity}
                }
                LOGGER.info('Writting hyperparameters on neural_db...')
                mongo.insertOneOnDB(mongo.getDB('neural_db'),hyperparams,'snn_hyperparameters',index='name',ignore_lock=True)
                LOGGER.info('Wrote hyperparameters on neural_db...OK')
            }elif opt == "--list-genetic-envs"{
                LOGGER.info('Getting genetic environments...')
                for env in mongo.findAllOnDB(mongo.getDB('genetic_db'),'environments',wait_unlock=False){
                    LOGGER.clean('Name: {}'.format(env['name']))
                    LOGGER.clean('Submitted At: {}'.format(env['submitted_at']))
                    LOGGER.clean('Search Space:')
                    if 'space_search' in env{
                        for k,v in env['space_search'].items() {
                            LOGGER.clean('\t{}: {}'.format(k,str(v)))
                        }
                    }elif 'search_space' in env{
                        for k,v in env['search_space'].items() {
                            LOGGER.clean('\t{}: {}'.format(k,str(v)))
                        }
                    }
                    LOGGER.clean('\n')
                }
                LOGGER.info('Gotten genetic environments...OK')
            }elif opt == "--eval-neural"{
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
            }elif opt == "--train-neural"{
                print('Now enter the data to train the neural network...')
                print('Enter a existing hyperparameters name to be used: ', end = '')
                hyper_name=input().strip()
                hyper=mongo.findOneOnDBFromIndex(mongo.getDB('neural_db'),'snn_hyperparameters','name',hyper_name,wait_unlock=False)
                if hyper==None{
                    LOGGER.error('Not found a hyperparameter for the given name!')
                }else{
                    print('Select your core version (1-2):')
                    print('\t1 - C++')
                    print('\t2 - Pytho{\}')
                    version=inputNumber(greater_or_eq=1,lower_or_eq=2)
                    print()
                    print('Enter a name for the train (unique): ', end = '')
                    train_name=input().strip()
                    submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                    started_by=None
                    started_at=None
                    finished_at=None
                    weights=None
                    print('Enter the max epochs: ',end='')
                    epochs=inputNumber(greater_or_eq=1)
                    if version==2{
                        print('Enter the patience epochs (0 means no patience): ',end='')
                        patience_epochs=epochs=inputNumber()
                    }else{
                        patience_epochs=None
                    }
                    print('Enter the cross validation method (0-4):')
                    print('\t0 - NONE')
                    print('\t1 - ROLLING_FORECASTING_ORIGIN')
                    print('\t2 - KFOLDS')
                    print('\t3 - FIXED_PERCENT')
                    print('value: ', end='')
                    cross_validation=inputNumber(lower_or_eq=3)
                    print('Enter the metric to be used during training/val (0-4):')
                    print('\t0 - RAW_LOSS')
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
                        print('\t0 - RAW_LOSS')
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
                    train_metadata={'name':train_name,'hyperparameters_name':hyper_name,'submitted_at':submitted_at,'started_by':started_by,'started_at':started_at,'finished_at':finished_at,'epochs':epochs,'patience_epochs':patience_epochs,'cross_validation':cross_validation,'train_metric':train_metric,'train_data':train_data,'test_metric':test_metric,'test_data':test_data,'weights':weights}
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
                    print('Enter the algorithm to use (0 - Enhanced | 1 - Standard): ')
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
                    print('\t3 - FIXED_PERCENT')
                    print('value: ')
                    cross_validation=inputNumber(lower_or_eq=3)
                    print('Enter the metric to be used (0-4):')
                    print('\t0 - RAW_LOSS')
                    print('\t1 - F1')
                    print('\t2 - RECALL')
                    print('\t3 - ACCURACY')
                    print('\t4 - PRECISION')
                    print('value: ')
                    metric=inputNumber(lower_or_eq=4)
                    if 	'core_version' not in env or env['core_version']=='v1'{
                        print('Enter the label type (0-2):')
                        print('\t0 - INT_CLASS')
                        print('\t1 - NEURON_BY_NEURON')
                        print('\t2 - NEURON_BY_NEURON_LOG_LOSS') # DEPRECATED
                        print('value: ', end='')
                        label_type=inputNumber(lower_or_eq=2)
                    }else{
                        print('Now enter label type ([0-1]+[3-8]):')
                        print('\t0 - INCREMENTAL')
                        print('\t1 - BINARY')
                        # print('\t2 - NEURON_BY_NEURON_LOG_LOSS DEPRECATED')
                        print('\t3 - BINARY_PLUS_ONE')
                        print('\t4 - SPARSE')
                        print('\t5 - DISTINCT_SPARSE')
                        print('\t6 - DISTINCT_SPARSE_PLUS_ONE')
                        print('\t7 - INCREMENTAL_PLUS_ONE')
                        print('\t8 - EXPONENTIAL')
                        print('value: ', end='')
                        label_type=inputNumber(lower_or_eq=8)
                        while label_type==2{
                            print('Label type 2 - NEURON_BY_NEURON_LOG_LOSS is deprecated, try another number:')
                            label_type=inputNumber(lower_or_eq=8)
                        }
                    }
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
                print('Select your core version (1-2):')
                print('\t1 - C++')
                print('\t2 - Pytho{\}')
                version=inputNumber(greater_or_eq=1,lower_or_eq=2)
                print()
                if version==2{
                    print('Use multiple networks (0 [False] - 1 [True]):')
                    multiple_networks=inputNumber(lower_or_eq=1)==1
                    if multiple_networks {
                        print('Now type the minimum and maximums for each item of the Smart Neural Search Space...')
                        print('Enter the genetic environment name: ', end = '')
                        gen_name=input().strip()
                        submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                        print('Enter the maximum epochs: min: ')
                        epochs_min=inputNumber(greater_or_eq=1)
                        print('Max: ')
                        epochs_max=inputNumber(greater_or_eq=epochs_min)
                        print('Enter the patience epochs (0 means no patience): min: ')
                        patience_epochs_min=inputNumber()
                        print('Max: ')
                        patience_epochs_max=inputNumber(greater_or_eq=patience_epochs_min)
                        print('Enter the batch size: min: ')
                        batch_size_min=inputNumber()
                        print('Max: ')
                        batch_size_max=inputNumber(greater_or_eq=batch_size_min)
                        print()
                        print('We\'ll use 5 networks for each group of feature and one final network to concatenate every other, now we\'ll define the parameters for them')
                        network_names=['Main features','CVSS ENUM features','Description features','Reference Features','Vendor Features','Concatenation']
                        amount_of_layers_min=[None for _ in range(len(network_names))]
                        amount_of_layers_max=[None for _ in range(len(network_names))]
                        layer_size_min=[None for _ in range(len(network_names))]
                        layer_size_max=[None for _ in range(len(network_names))]
                        activation_min=[None for _ in range(len(network_names))]
                        activation_max=[None for _ in range(len(network_names))]
                        dropouts_min=[None for _ in range(len(network_names))]
                        dropouts_max=[None for _ in range(len(network_names))]
                        use_same_alpha=None
                        use_same_bias=None
                        use_same_loss=None
                        use_same_optimizer=None
                        alpha_min=[None for _ in range(len(network_names))]
                        alpha_max=[None for _ in range(len(network_names))]
                        bias_min=[None for _ in range(len(network_names))]
                        bias_max=[None for _ in range(len(network_names))]
                        loss_min=[None for _ in range(len(network_names))]
                        loss_max=[None for _ in range(len(network_names))]
                        optimizer_min=[None for _ in range(len(network_names))]
                        optimizer_max=[None for _ in range(len(network_names))]
                        for n in range(len(network_names)){
                            print('Now enter data regarding the {} network'.format(network_names[n]))
                            print('Enter the amount of layers: min: ')
                            amount_of_layers_min[n]=inputNumber(greater_or_eq=1)
                            print('Max: ')
                            amount_of_layers_max[n]=inputNumber(greater_or_eq=amount_of_layers_min[n])
                            if amount_of_layers_min[n]>1 or amount_of_layers_max[n]>1 or n!=len(network_names)-1{
                                print('Enter the layer sizes: min: ')
                                layer_size_min[n]=inputNumber(greater_or_eq=1)
                                print('Max: ')
                                layer_size_max[n]=inputNumber(greater_or_eq=layer_size_min[n])
                            }else{
                                layer_size_min[n]=0
                                layer_size_max[n]=0
                            }
                            if amount_of_layers_min[n]>1 or amount_of_layers_max[n]>1 or n!=len(network_names)-1{
                                print('Activation functions (0-9):')
                                print('\t0 - ReLU')
                                print('\t1 - Softmax')
                                print('\t2 - Sigmoid')
                                print('\t3 - Tanh')
                                print('\t4 - Softplus')
                                print('\t5 - Softsign')
                                print('\t6 - Selu')
                                print('\t7 - Elu')
                                print('\t8 - Exponential')
                                print('\t9 - Linear')
                                if n!=len(network_names)-1 {
                                    print('Enter the activation functions for all nodes: min: ')
                                }else{
                                    print('Enter the activation functions for all nodes except output layer: min: ')
                                }
                                activation_min[n]=inputNumber(lower_or_eq=9)
                                print('Max: ')
                                activation_max[n]=inputNumber(lower_or_eq=9,greater_or_eq=activation_min[n])
                            }else{
                                activation_min[n]=0
                                activation_max[n]=0
                            }
                            print('Enter the dropouts layer value: min: ')
                            dropouts_min[n]=inputNumber(is_float=True,lower_or_eq=1)
                            print('Max: ')
                            dropouts_max[n]=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=dropouts_min[n])
                            if use_same_alpha is None {
                                print('Should we use the same alpha for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_alpha=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_alpha or alpha_min[0] is None {
                                print('Enter the alpha: min: ')
                                alpha_min[n]=inputNumber(is_float=True,lower_or_eq=1)
                                print('Max: ')
                                alpha_max[n]=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=alpha_min[n])
                            }else{
                                alpha_min[n]=alpha_min[0]
                                alpha_max[n]=alpha_max[0]
                            }
                            if use_same_bias is None {
                                print('Should we use the same bias for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_bias=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_bias or bias_min[0] is None {
                                print('Use bias on layers (0 [False] - 1 [True]):')
                                print('min: ')
                                bias_min[n]=inputNumber(lower_or_eq=1)==1
                                print('Max: ')
                                bias_max[n]=inputNumber(lower_or_eq=1,greater_or_eq=bias_min[n])==1
                            }else{
                                bias_min[n]=bias_min[0]
                                bias_max[n]=bias_max[0]
                            }
                            if use_same_loss is None {
                                print('Should we use the same loss for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_loss=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_loss or loss_min[0] is None {
                                print('Enter the loss functions (0-3):')
                                print('\t0 - Binary Crossentropy')
                                print('\t1 - Categorical Crossentropy')
                                print('\t2 - Mean Squared Error')
                                print('\t3 - Mean Absolute Error')
                                print('min: ')
                                loss_min[n]=inputNumber(lower_or_eq=3)
                                print('Max: ')
                                loss_max[n]=inputNumber(lower_or_eq=3,greater_or_eq=loss_min[n])
                            }else{
                                loss_min[n]=loss_min[0]
                                loss_max[n]=loss_max[0]
                            }
                            if use_same_optimizer is None {
                                print('Should we use the same optimizer for every network instead of specify one for each? (0 [False] - 1 [True]):')
                                use_same_optimizer=inputNumber(lower_or_eq=1)==1
                            }
                            if not use_same_optimizer or optimizer_min[0] is None {
                                print('Enter the optimizer (0-2): ')
                                print('\t0 - SGD')
                                print('\t1 - Adam')
                                print('\t2 - RMSProp')
                                print('min: ')
                                optimizer_min[n]=inputNumber(lower_or_eq=2)
                                print('Max: ')
                                optimizer_max[n]=inputNumber(lower_or_eq=2,greater_or_eq=optimizer_min[n])
                            }else{
                                optimizer_min[n]=optimizer_min[0]
                                optimizer_max[n]=optimizer_max[0]
                            }
                            print()
                        }
                        print('Activation functions (0-9):')
                        print('\t0 - ReLU')
                        print('\t1 - Softmax')
                        print('\t2 - Sigmoid')
                        print('\t3 - Tanh')
                        print('\t4 - Softplus')
                        print('\t5 - Softsign')
                        print('\t6 - Selu')
                        print('\t7 - Elu')
                        print('\t8 - Exponential')
                        print('\t9 - Linear')
                        print('Now enter the activation function for the OUTPUT layer of the Concatenation network (0-9) recommended (1-2):')
                        activation_out=inputNumber(lower_or_eq=9)
                        environment_to_insert={'core_version':'v2','name':gen_name,'submitted_at':submitted_at,'amount_of_networks':len(network_names),'search_space':{'output_layer_node_type':activation_out,'amount_of_layers':{'min':amount_of_layers_min,'max':amount_of_layers_max},'epochs':{'min':epochs_min,'max':epochs_max},'patience_epochs':{'min':patience_epochs_min,'max':patience_epochs_max},'batch_size':{'min':batch_size_min,'max':batch_size_max},'layer_sizes':{'min':layer_size_min,'max':layer_size_max},'activation_functions':{'min':activation_min,'max':activation_max},'dropouts':{'min':dropouts_min,'max':dropouts_max},'alpha':{'min':alpha_min,'max':alpha_max},'loss':{'min':loss_min,'max':loss_max},'bias':{'min':bias_min,'max':bias_max},'optimizer':{'min':optimizer_min,'max':optimizer_max}}}
                    }else{
                        print('Now type the minimum and maximums for each item of the Smart Neural Search Space...')
                        print('Enter the genetic environment name: ', end = '')
                        gen_name=input().strip()
                        submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                        print('Enter the amount of layers: min: ')
                        amount_of_layers_min=inputNumber(greater_or_eq=1)
                        print('Max: ')
                        amount_of_layers_max=inputNumber(greater_or_eq=amount_of_layers_min)
                        print('Enter the maximum epochs: min: ')
                        epochs_min=inputNumber(greater_or_eq=1)
                        print('Max: ')
                        epochs_max=inputNumber(greater_or_eq=epochs_min)
                        print('Enter the patience epochs (0 means no patience): min: ')
                        patience_epochs_min=inputNumber()
                        print('Max: ')
                        patience_epochs_max=inputNumber(greater_or_eq=patience_epochs_min)
                        print('Enter the alpha: min: ')
                        alpha_min=inputNumber(is_float=True,lower_or_eq=1)
                        print('Max: ')
                        alpha_max=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=alpha_min)
                        print('Enter the batch size: min: ')
                        batch_size_min=inputNumber()
                        print('Max: ')
                        batch_size_max=inputNumber(greater_or_eq=batch_size_min)
                        print('Enter the loss functions (0-3):')
                        print('\t0 - Binary Crossentropy')
                        print('\t1 - Categorical Crossentropy')
                        print('\t2 - Mean Squared Error')
                        print('\t3 - Mean Absolute Error')
                        print('min: ')
                        loss_min=inputNumber(lower_or_eq=3)
                        print('Max: ')
                        loss_max=inputNumber(lower_or_eq=3,greater_or_eq=loss_min)
                        print('Enter the optimizer (0-2): ')
                        print('\t0 - SGD')
                        print('\t1 - Adam')
                        print('\t2 - RMSProp')
                        print('min: ')
                        optimizer_min=inputNumber(lower_or_eq=2)
                        print('Max: ')
                        optimizer_max=inputNumber(lower_or_eq=2,greater_or_eq=optimizer_min)
                        if amount_of_layers_min>1 or amount_of_layers_max>1{
                            print('Enter the layer sizes: min: ')
                            layer_size_min=inputNumber(greater_or_eq=1)
                            print('Max: ')
                            layer_size_max=inputNumber(greater_or_eq=layer_size_min)
                        }else{
                            layer_size_min=0
                            layer_size_max=0
                        }
                        print('Use bias on layers (0 [False] - 1 [True]):')
                        print('min: ')
                        bias_min=inputNumber(lower_or_eq=1)==1
                        print('Max: ')
                        bias_max=inputNumber(lower_or_eq=1,greater_or_eq=bias_min)==1
                        print('Enter the dropouts layer value: min: ')
                        dropouts_min=inputNumber(is_float=True,lower_or_eq=1)
                        print('Max: ')
                        dropouts_max=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=dropouts_min)
                        print('Activation functions (0-9):')
                        print('\t0 - ReLU')
                        print('\t1 - Softmax')
                        print('\t2 - Sigmoid')
                        print('\t3 - Tanh')
                        print('\t4 - Softplus')
                        print('\t5 - Softsign')
                        print('\t6 - Selu')
                        print('\t7 - Elu')
                        print('\t8 - Exponential')
                        print('\t9 - Linear')
                        if amount_of_layers_min>1 or amount_of_layers_max>1{
                            print('Enter the activation functions for all nodes except output layer: min: ')
                            activation_min=inputNumber(lower_or_eq=9)
                            print('Max: ')
                            activation_max=inputNumber(lower_or_eq=9,greater_or_eq=activation_min)
                        }else{
                            activation_min=0
                            activation_max=0
                        }
                        print('Now enter the activation function for the OUTPUT layer (0-9) recommended (1-2):')
                        activation_out=inputNumber(lower_or_eq=9)
                        environment_to_insert={'core_version':'v2','name':gen_name,'submitted_at':submitted_at,'amount_of_networks':1,'search_space':{'output_layer_node_type':activation_out,'amount_of_layers':{'min':amount_of_layers_min,'max':amount_of_layers_max},'epochs':{'min':epochs_min,'max':epochs_max},'patience_epochs':{'min':patience_epochs_min,'max':patience_epochs_max},'batch_size':{'min':batch_size_min,'max':batch_size_max},'layer_sizes':{'min':layer_size_min,'max':layer_size_max},'activation_functions':{'min':activation_min,'max':activation_max},'dropouts':{'min':dropouts_min,'max':dropouts_max},'alpha':{'min':alpha_min,'max':alpha_max},'loss':{'min':loss_min,'max':loss_max},'bias':{'min':bias_min,'max':bias_max},'optimizer':{'min':optimizer_min,'max':optimizer_max}}}
                    }
                }else{
                    print('Now type the minimum and maximums for each item of the Smart Neural Search Space...')
                    print('Enter the genetic environment name: ', end = '')
                    gen_name=input().strip()
                    submitted_at=Utils.getTodayDate('%d/%m/%Y %H:%M:%S')
                    print('Enter the amount of layers: min: ')
                    amount_of_layers_min=inputNumber()
                    print('Max: ')
                    amount_of_layers_max=inputNumber(greater_or_eq=amount_of_layers_min)
                    print('Enter the epochs: min: ')
                    epochs_min=inputNumber()
                    print('Max: ')
                    epochs_max=inputNumber(greater_or_eq=epochs_min)
                    print('Enter the alpha: min: ')
                    alpha_min=inputNumber(is_float=True,lower_or_eq=1)
                    print('Max: ')
                    alpha_max=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=alpha_min)
                    print('Enter the batch size: min: ')
                    batch_size_min=inputNumber()
                    print('Max: ')
                    batch_size_max=inputNumber(greater_or_eq=batch_size_min)
                    print('Enter the layer sizes: min: ')
                    layer_size_min=inputNumber()
                    print('Max: ')
                    layer_size_max=inputNumber(greater_or_eq=layer_size_min)
                    print('Enter the range pow: min: ')
                    range_pow_min=inputNumber()
                    print('Max: ')
                    range_pow_max=inputNumber(greater_or_eq=range_pow_min)
                    print('Enter the K values: min: ')
                    k_min=inputNumber()
                    print('Max: ')
                    k_max=inputNumber(greater_or_eq=k_min)
                    print('Enter the L values: min: ')
                    l_min=inputNumber()
                    print('Max: ')
                    l_max=inputNumber(greater_or_eq=l_min)
                    print('Enter the sparcity: min: ')
                    sparcity_min=inputNumber(is_float=True,lower_or_eq=1)
                    print('Max: ')
                    sparcity_max=inputNumber(is_float=True,lower_or_eq=1,greater_or_eq=sparcity_min)
                    print('Enter the activation functions (0-2):')
                    print('\t0 - ReLU')
                    print('\t1 - Softmax')
                    print('\t2 - Sigmoid')
                    print('min: ')
                    activation_min=inputNumber(lower_or_eq=2)
                    print('Max: ')
                    activation_max=inputNumber(lower_or_eq=2,greater_or_eq=activation_min)
                    environment_to_insert={'core_version':'v1','name':gen_name,'submitted_at':submitted_at,'space_search':{'amount_of_layers':{'min':amount_of_layers_min,'max':amount_of_layers_max},'epochs':{'min':epochs_min,'max':epochs_max},'batch_size':{'min':batch_size_min,'max':batch_size_max},'layer_sizes':{'min':layer_size_min,'max':layer_size_max},'range_pow':{'min':range_pow_min,'max':range_pow_max},'K':{'min':k_min,'max':k_max},'L':{'min':l_min,'max':l_max},'activation_functions':{'min':activation_min,'max':activation_max},'sparcity':{'min':sparcity_min,'max':sparcity_max},'alpha':{'min':alpha_min,'max':alpha_max}}}
                }
                LOGGER.info('Writting environment on genetic_db...')
                mongo.insertOneOnDB(mongo.getDB('genetic_db'),environment_to_insert,'environments',index='name',ignore_lock=True)
                LOGGER.info('Wrote environment on genetic_db...OK')
            }elif opt == "--rm-genetic-env"{
                arg=arg.strip()
                LOGGER.info('Removing {} from genetic environments...'.format(arg))
                query={'name':arg}
                mongo.rmOneFromDB(mongo.getDB('genetic_db'),'environments',query=query)
                LOGGER.info('Removed {} from genetic environments...OK'.format(arg))
            }elif opt == "--rm-neural-hyperparams"{
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
        value = input("Keeping Front end alive on Iterative Mode...\nEnter a command (e.g. -h): ")
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
