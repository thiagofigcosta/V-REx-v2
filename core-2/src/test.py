#!/bin/python

import math
from SearchSpace import SearchSpace
from Utils import Utils
from Logger import Logger
from Core import Core
from Dataset import Dataset
from Genome import Genome
from Enums import LabelEncoding,NodeType,Loss,Metric,Optimizers,GeneticRankType
from Hyperparameters import Hyperparameters
from HallOfFame import HallOfFame
from NeuralNetwork import NeuralNetwork
from GeneticAlgorithm import GeneticAlgorithm
from StandardNeuralNetwork import StandardNeuralNetwork
from EnhancedNeuralNetwork import EnhancedNeuralNetwork
from StandardGeneticAlgorithm import StandardGeneticAlgorithm
from EnhancedGeneticAlgorithm import EnhancedGeneticAlgorithm
from PopulationManager import PopulationManager

from memory_profiler import profile

TMP_FOLDER='tmp/core/'
Utils.createFolderIfNotExists(TMP_FOLDER)
LOGGER=Logger(TMP_FOLDER,name='core')
Utils(TMP_FOLDER,LOGGER)
core=Core(None,LOGGER)

def testStdGenetic(){
    def eggHolder(genome){
        # https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
        x=genome.dna[0]
        y=genome.dna[1]
        return -(y+47)*math.sin(math.sqrt(abs(y+(x/2)+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
    }

    def easom(genome){
        # https://www.sfu.ca/~ssurjano/easom.html TIME MINUS ONE // maximum -> x1=x2=pi -> y(x1,x2)=1
        x=genome.dna[0]
        y=genome.dna[1]
        return -(-math.cos(x)*math.cos(y)*math.exp(-(math.pow(x-math.pi,2)+math.pow(y-math.pi,2))))
    }

    verbose=False
    print('Minimization')
    limits=SearchSpace()
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='x')
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='y')
    population_size=300
    max_gens=100
    mutation_rate=0.2
    sex_rate=0.6
    search_maximum=False
    max_notables=5
    elite_min=HallOfFame(max_notables, search_maximum)
    ga=StandardGeneticAlgorithm(search_maximum,mutation_rate, sex_rate)
    population=PopulationManager(ga,limits,eggHolder,population_size,neural_genome=False,print_deltas=verbose,after_gen_callback=lambda x:print('After gen'))
    population.hall_of_fame=elite_min
    population.naturalSelection(max_gens,verbose)
    print('Expected: (x: 512, y: 404.2319) = -959.6407')
    for individual in elite_min.notables{
        print(str(individual))
    }
    Utils.printDict(elite_min.best,'Elite')

    runned_after_gen=False
    def afterGen(args_list){
        nonlocal runned_after_gen
        if not runned_after_gen {
            print('After gen - only once')
            runned_after_gen=True
        }
    }

    print('Maximization')
    limits=SearchSpace()
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='x')
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='y')
    max_gens=100
    mutation_rate=0.1
    sex_rate=0.7
    search_maximum=True
    max_notables=5
    elite_max=HallOfFame(max_notables, search_maximum)
    ga=StandardGeneticAlgorithm(search_maximum,mutation_rate, sex_rate)
    population=PopulationManager(ga,limits,easom,population_size,neural_genome=False,print_deltas=verbose,after_gen_callback=afterGen)
    population.hall_of_fame=elite_max
    population.naturalSelection(max_gens,verbose)
    print('Expected: (x: 3.141592, y: 3.141592) = 1')
    for individual in elite_max.notables{
        print(str(individual))
    }
    Utils.printDict(elite_max.best,'Elite')
}

def testEnhGenetic(){
    def eggHolder(genome){
        # https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
        x=genome.dna[0]
        y=genome.dna[1]
        return -(y+47)*math.sin(math.sqrt(abs(y+(x/2)+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
    }

    def easom(genome){
        # https://www.sfu.ca/~ssurjano/easom.html TIME MINUS ONE // maximum -> x1=x2=pi -> y(x1,x2)=1
        x=genome.dna[0]
        y=genome.dna[1]
        return -(-math.cos(x)*math.cos(y)*math.exp(-(math.pow(x-math.pi,2)+math.pow(y-math.pi,2))))
    }

    PopulationManager.PRINT_REL_FREQUENCY=0

    verbose_natural_selection=False
    verbose_population_details=True
    add_callback_after_gen=False
    print('Minimization')
    limits=SearchSpace()
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='x')
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='y')
    population_start_size_enh=300
    population_start_size_std=780
    max_gens=100
    max_age=5
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=False
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,limits,eggHolder,population_start_size_enh,neural_genome=False,print_deltas=verbose_population_details,after_gen_callback=lambda x:print('After gen') if add_callback_after_gen else None)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    

    
    print('Expected: (x: 512, y: 404.2319) = -959.6407')
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')

    runned_after_gen=False
    def afterGen(){
        nonlocal runned_after_gen
        if not runned_after_gen {
            print('After gen - only once')
            runned_after_gen=True
        }
    }
    del enh_elite
    del enh_population

    print('Maximization')
    limits=SearchSpace()
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='x')
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='y')
    max_gens=100
    max_age=5
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=True
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,limits,easom,population_start_size_enh,neural_genome=False,print_deltas=verbose_population_details,after_gen_callback=afterGen if add_callback_after_gen else None)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)

    print('Expected: (x: 3.141592, y: 3.141592) = 1')
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')
}

def testStdVsEnhGenetic(mutation_rate=0.1,max_age=6,max_children=5,recycle_rate=0.13,sex_rate=0.7){
    def eggHolder(genome){
        # https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
        x=genome.dna[0]
        y=genome.dna[1]
        return -(y+47)*math.sin(math.sqrt(abs(y+(x/2)+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
    }

    def easom(genome){
        # https://www.sfu.ca/~ssurjano/easom.html TIME MINUS ONE // maximum -> x1=x2=pi -> y(x1,x2)=1
        x=genome.dna[0]
        y=genome.dna[1]
        return -(-math.cos(x)*math.cos(y)*math.exp(-(math.pow(x-math.pi,2)+math.pow(y-math.pi,2))))
    }

    Core.FREE_MEMORY_MANUALLY=False

    verbose=False
    print_deltas=False
    print('Standard vs Enhanced - EggHolder:')
    tests=100
    limits=SearchSpace()
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='x')
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='y')
    population_start_size_enh=300
    max_gens=100
    search_maximum=False
    max_notables=5
    results={'standard':[],'enhanced':[]}
    for x in range(tests){
        print('Test {} of {}'.format(x+1,tests))
        enh_elite=HallOfFame(max_notables, search_maximum)
        en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
        enh_population=PopulationManager(en_ga,limits,eggHolder,population_start_size_enh,print_deltas=print_deltas)
        enh_population.hall_of_fame=enh_elite
        run_time=enh_population.naturalSelection(max_gens)
        enh_result=enh_elite.best
        results['enhanced'].append(enh_result)
        results['enhanced'][-1]['run_time']=run_time
        enhanced_population_mean=int(sum(enh_population.last_run_population_sizes)/len(enh_population.last_run_population_sizes))
        if Utils.LazyCore.freeMemManually(){
            del enh_elite
            del enh_population
        }

        population_start_size_std=enhanced_population_mean

        std_elite=HallOfFame(max_notables, search_maximum)
        std_ga=StandardGeneticAlgorithm(search_maximum,mutation_rate, sex_rate)
        std_population=PopulationManager(std_ga,limits,eggHolder,population_start_size_std,print_deltas=print_deltas)
        std_population.hall_of_fame=std_elite
        run_time=std_population.naturalSelection(max_gens)
        std_result=std_elite.best
        results['standard'].append(std_result)
        results['standard'][-1]['run_time']=run_time
        if Utils.LazyCore.freeMemManually(){
            del std_elite
            del std_population
        }
    }
    std_mean=[0.0,0.0,0.0]
    for std_result in results['standard']{
        std_mean[0]+=std_result['generation']
        std_mean[1]+=std_result['output']
        std_mean[2]+=std_result['run_time']
        print('Standard Best ({}): {}'.format(std_result['generation'],std_result['output']))
    }
    std_mean[0]/=tests
    std_mean[1]/=tests
    std_mean[2]/=tests
    print()
    enh_mean=[0.0,0.0,0.0]
    for enh_result in results['enhanced']{
        enh_mean[0]+=enh_result['generation']
        enh_mean[1]+=enh_result['output']
        enh_mean[2]+=enh_result['run_time']
        print('Enhanced Best ({}): {}'.format(enh_result['generation'],enh_result['output']))
    }
    enh_mean[0]/=tests
    enh_mean[1]/=tests
    enh_mean[2]/=tests
    print()
    Utils.printDict(results,'Results')
    print()
    mean_result_str_egg_holder='Standard Mean:\n\tgeneration: {}\n\toutput: {}\n\trun_time: {}\nEnhanced Mean:\n\tgeneration: {}\n\toutput: {}\n\trun_time: {}'.format(std_mean[0],std_mean[1],std_mean[2],enh_mean[0],enh_mean[1],enh_mean[2])
    print(mean_result_str_egg_holder)

    print()
    print()
    print('Standard vs Enhanced - easom:')
    limits=SearchSpace()
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='x')
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='y')
    search_maximum=True
    results={'standard':[],'enhanced':[]}
    for x in range(tests){
        print('Test {} of {}'.format(x+1,tests))
        enh_elite=HallOfFame(max_notables, search_maximum)
        en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
        enh_population=PopulationManager(en_ga,limits,easom,population_start_size_enh,print_deltas=print_deltas)
        enh_population.hall_of_fame=enh_elite
        run_time=enh_population.naturalSelection(max_gens)
        enh_result=enh_elite.best
        results['enhanced'].append(enh_result)
        results['enhanced'][-1]['run_time']=run_time
        enhanced_population_mean=int(sum(enh_population.last_run_population_sizes)/len(enh_population.last_run_population_sizes))
        if Utils.LazyCore.freeMemManually(){
            del enh_elite
            del enh_population
        }

        population_start_size_std=enhanced_population_mean

        std_elite=HallOfFame(max_notables, search_maximum)
        std_ga=StandardGeneticAlgorithm(search_maximum,mutation_rate, sex_rate)
        std_population=PopulationManager(std_ga,limits,easom,population_start_size_std,print_deltas=print_deltas)
        std_population.hall_of_fame=std_elite
        run_time=std_population.naturalSelection(max_gens)
        std_result=std_elite.best
        results['standard'].append(std_result)
        results['standard'][-1]['run_time']=run_time
        if Utils.LazyCore.freeMemManually(){
            del std_elite
            del std_population
        }
    }
    std_mean=[0.0,0.0,0.0]
    for std_result in results['standard']{
        std_mean[0]+=std_result['generation']
        std_mean[1]+=std_result['output']
        std_mean[2]+=std_result['run_time']
        print('Standard Best ({}): {}'.format(std_result['generation'],std_result['output']))
    }
    std_mean[0]/=tests
    std_mean[1]/=tests
    std_mean[2]/=tests
    print()
    enh_mean=[0.0,0.0,0.0]
    for enh_result in results['enhanced']{
        enh_mean[0]+=enh_result['generation']
        enh_mean[1]+=enh_result['output']
        enh_mean[2]+=enh_result['run_time']
        print('Enhanced Best ({}): {}'.format(enh_result['generation'],enh_result['output']))
    }
    enh_mean[0]/=tests
    enh_mean[1]/=tests
    enh_mean[2]/=tests
    print()
    Utils.printDict(results,'Results')
    print()
    mean_result_str_easom='Standard Mean:\n\tgeneration: {}\n\toutput: {}\n\trun_time: {}\nEnhanced Mean:\n\tgeneration: {}\n\toutput: {}\n\trun_time: {}'.format(std_mean[0],std_mean[1],std_mean[2],enh_mean[0],enh_mean[1],enh_mean[2])
    print(mean_result_str_easom)

    Core.FREE_MEMORY_MANUALLY=True
    return mean_result_str_egg_holder,mean_result_str_easom
}


def testNNIntLabel(){
    label_type=LabelEncoding.INCREMENTAL

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    features,labels=Dataset.filterDataset(features,labels,'Iris-setosa')
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,label_type)
    features,labels=Dataset.balanceDataset(features,labels)
    print('First label:',Dataset.translateLabelFromOutput(labels[0],label_map,label_map_2))
    print('First label enum:',Dataset.translateLabelFromOutput(labels[0],label_map_2))
    print('First label encoding:',labels[0])
    features,scale=Dataset.normalizeDatasetFeatures(features)
    print('First feature',features[0])
    print('First labels',labels[:4])
    features,labels=Dataset.shuffleDataset(features,labels)
    print('First labels randomized',labels[:4])
    train,test=Dataset.splitDataset(features,labels,.7)
    train,val=Dataset.splitDataset(train[0],train[1],.7)

    input_size=len(train[0][0])
    output_size=len(train[1][0])
    layers=2
    dropouts=0
    bias=True
    layer_sizes=[5,output_size]
    node_types=[NodeType.TANH,NodeType.SOFTMAX]
    batch_size=5
    alpha=0.01
    shuffle=True
    optimizer=Optimizers.ADAM
    patience_epochs=0
    max_epochs=100
    loss=Loss.CATEGORICAL_CROSSENTROPY
    monitor_metric=Metric.F1
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric)

    nn=StandardNeuralNetwork(hyperparameters,name='iris',verbose=True)
    nn.buildModel(input_size=input_size)
    nn.train(train[0],train[1],val[0],val[1])
    history=nn.history
    Utils.printDict(history,'History')
    preds,activations=nn.predict(test[0],True,True)
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=nn.eval(test[0],test[1])
    nn.clearCache()
    del nn
    Utils.printDict(eval_res,'Eval')
    Dataset.compareAndPrintLabels(preds,activations,test[1],show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}


def testNNBinLabel_KFolds(){
    label_type=LabelEncoding.BINARY

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,label_type)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)
    train,test=Dataset.splitDataset(features,labels,.7)

    input_size=len(train[0][0])
    output_size=len(train[1][0])
    layers=2
    dropouts=0
    bias=True
    layer_sizes=[5,output_size]
    node_types=[NodeType.TANH,NodeType.SOFTMAX]
    batch_size=5
    alpha=0.01
    shuffle=True
    optimizer=Optimizers.ADAM
    patience_epochs=15
    max_epochs=100
    loss=Loss.BINARY_CROSSENTROPY
    monitor_metric=Metric.RAW_LOSS
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric)

    nn=StandardNeuralNetwork(hyperparameters,name='iris',verbose=True)
    nn.buildModel(input_size=input_size)
    # KFolds
    nn.trainKFolds(train[0],train[1],8)
    ############################################################
    # # Rolling Forecast Origin Technique
    # nn.trainRollingForecast(train[0],train[1])
    # # No Validation
    # nn.trainNoValidation(train[0],train[1])
    # # Custom Validation
    # nn.trainCustomValidation(train[0],train[1],test[0],test[1])
    ############################################################
    history=nn.history
    preds,activations=nn.predict(test[0],True,True)
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=nn.eval(test[0],test[1])
    del nn
    Utils.printDict(eval_res,'Eval')
    Dataset.compareAndPrintLabels(preds,activations,test[1],show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}

def testGeneticallyTunedNN(){
    metric=Metric.ACCURACY
    search_space=SearchSpace()
    search_space.add(1,4,SearchSpace.Type.INT,'layers')
    search_space.add(5,15,SearchSpace.Type.INT,'batch_size')
    search_space.add(0.0001,0.1,SearchSpace.Type.FLOAT,'alpha')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'shuffle')
    search_space.add(15,30,SearchSpace.Type.INT,'patience_epochs')
    search_space.add(20,150,SearchSpace.Type.INT,'max_epochs')
    search_space.add(Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY,SearchSpace.Type.INT,'loss')
    search_space.add(LabelEncoding.SPARSE,LabelEncoding.SPARSE,SearchSpace.Type.INT,'label_type')
    search_space.add(Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),SearchSpace.Type.INT,'optimizer')
    search_space.add(metric,metric,SearchSpace.Type.INT,'monitor_metric')
    search_space.add(True,True,SearchSpace.Type.BOOLEAN,'model_checkpoint')
    search_space.add(2,8,SearchSpace.Type.INT,'layer_sizes')
    search_space.add(Utils.getEnumBorder(NodeType,False),NodeType.TANH,SearchSpace.Type.INT,'node_types')
    search_space.add(0,0.995,SearchSpace.Type.FLOAT,'dropouts')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'bias')
    search_space=Genome.enrichSearchSpace(search_space)

    Genome.CACHE_WEIGHTS=False

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    features,labels=Dataset.filterDataset(features,labels,'Iris-setosa')
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    features,labels=Dataset.balanceDataset(features,labels)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)

    train,test=Dataset.splitDataset(features,labels,.7)

    def train_callback(genome){
        nonlocal train
        kfolds=5
        preserve_weights=False # TODO fix when true, to avoid nan outputs
        train_features=train[0]
        train_labels=train[1]
        train_labels,_=Dataset.encodeDatasetLabels(train_labels,genome.getHyperparametersEncoder(False))
        input_size=len(train_features[0])
        output_size=len(train_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX)
        search_maximum=hyperparameters.monitor_metric.isMaxMetric(hyperparameters.loss)
        nn=StandardNeuralNetwork(hyperparameters,name='iris_{}'.format(genome.id),verbose=False)
        nn.buildModel(input_size=input_size)
        nn.setWeights(genome.getWeights())
        nn.trainKFolds(train_features,train_labels,kfolds)
        if preserve_weights and hyperparameters.model_checkpoint{
            nn.restoreCheckpointWeights()
        }
        output=nn.getMetricMean(hyperparameters.monitor_metric.toKerasName(),True)
        if output!=output{ # Not a Number, ignore this genome
            Core.LOGGER.warn('Not a number metric ('+str(hyperparameters.monitor_metric.toKerasName())+') mean of '+str(nn.getMetric(hyperparameters.monitor_metric.toKerasName(),True)))
            output=float('-inf') if search_maximum else float('inf')
        }
        if preserve_weights {
            genome.setWeights(nn.mergeWeights(genome.getWeights()))
        }
        del nn
        return output
    }

    verbose_natural_selection=True
    verbose_population_details=True
    population_start_size_enh=10
    max_gens=10
    max_age=5
    max_children=3
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    max_notables=5
    search_maximum=metric.isMaxMetric()
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,search_space,train_callback,population_start_size_enh,neural_genome=True,print_deltas=verbose_population_details)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')
    print('Evaluating best')

    def test_callback(genome){
        nonlocal test
        test_features=test[0]
        test_labels=test[1]
        test_labels,label_map_2=Dataset.encodeDatasetLabels(test_labels,genome.getHyperparametersEncoder(False))
        input_size=len(test_features[0])
        output_size=len(test_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX)
        nn=StandardNeuralNetwork(hyperparameters,name='iris_{}'.format(genome.id),verbose=False)
        nn.buildModel(input_size=input_size)
        nn.saveModelSchemaToFile()
        nn.setWeights(genome.getWeights())
        print('Best genome encoded weights:',genome.getWeights(raw=True))
        preds,activations=nn.predict(test_features,True,True)
        del nn
        Dataset.compareAndPrintLabels(preds,activations,test_labels,show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
        Utils.printDict(Dataset.statisticalAnalysis(preds,test_labels),'Statistical Analysis')
    }

    test_callback(enh_elite.getBestGenome())
}

def testCustomEncodings(){
    base64='AAAAAAAAAAaBCDDDDDDE'
    print('base64--original',base64)
    base65=Utils.base64ToBase65(base64)
    print('base65-converted',base65)
    print('base64-converted',Utils.base65ToBase64(base65))
}

def testEnhancedNN_SingleNet(){
    label_type=LabelEncoding.SPARSE

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,label_type)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)
    train,test=Dataset.splitDataset(features,labels,.7)

    input_size=len(train[0][0])
    output_size=len(train[1][0])
    layers=2
    dropouts=0
    bias=True
    layer_sizes=[5,output_size]
    node_types=[NodeType.TANH,NodeType.SOFTMAX]
    batch_size=5
    alpha=0.01
    shuffle=True
    optimizer=Optimizers.SGD
    patience_epochs=15
    max_epochs=100
    loss=Loss.BINARY_CROSSENTROPY
    monitor_metric=Metric.RAW_LOSS
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric)

    enn=EnhancedNeuralNetwork(hyperparameters,name='iris',verbose=True)
    enn.buildModel(input_size=input_size)
    enn.trainKFolds(train[0],train[1],8)
    history=enn.history
    preds,activations=enn.predict(test[0],True,True)
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=enn.eval(test[0],test[1])
    del enn
    Utils.printDict(eval_res,'Eval')
    Dataset.compareAndPrintLabels(preds,activations,test[1],show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}


def testEnhancedNN_MultiNet(){
    label_type=LabelEncoding.SPARSE

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,label_type)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)
    train,test=Dataset.splitDataset(features,labels,.7)
    groups=[[0,2],[2,-1]]
    print('Before division [0]:',train[0][0])

    train[0]=Dataset.divideFeaturesIntoMultipleGroups(train[0],groups)
    test[0]=Dataset.divideFeaturesIntoMultipleGroups(test[0],groups)
    print('After division [0]:', train[0][0][0],'and',train[0][1][0])
    print()
    # BEFORE
        # in - hid - out
        # in - hid - out
        # in - hid - out
        # in - hid -
        #      hid -
        #
        # Hyper: input_size=4, output_size = 3, hidden_size= 5

    # NOW
        # in - hid - out  | in - hid - out
        # in - hid - out  | in - hid - out
        #      hid -             hid - out
    
        # in - hid - out  | in - hid -
        # in - hid -
        #      hid -
        #
        # Hyper: networks=3 (considering 1 to concatenate) 
        #   Net A: input_size=2, output_size = 2, hidden_size= 3
        #   Net B: input_size=2, output_size = 1, hidden_size= 3
        #
        #   Net Concat: intput size=(NetA_output_size+NetB_output_size), output_size=3, hidden_size=4
    #
    amount_of_networks=3
    input_size=[len(train[0][i][0]) for i in range(len(train[0]))]
    intermediary_size=[2,1]
    output_size=len(train[1][0])
    layers=[2,2,2]
    dropouts=[0,0,0]
    bias=[True,True,True]
    layer_sizes=[[3,intermediary_size[0]],[3,intermediary_size[1]],[4,output_size]]
    node_types=[[NodeType.TANH,NodeType.TANH],[NodeType.TANH,NodeType.TANH],[NodeType.TANH,NodeType.SOFTMAX]]
    batch_size=5
    alpha=[0.01,0.01,0.01]
    shuffle=True
    optimizer=[Optimizers.SGD,Optimizers.SGD,Optimizers.SGD]
    patience_epochs=30
    max_epochs=200
    loss=[Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY]
    monitor_metric=Metric.RAW_LOSS
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric,amount_of_networks=amount_of_networks)

    enn=EnhancedNeuralNetwork(hyperparameters,name='iris',verbose=True)
    enn.buildModel(input_size=input_size)
    enn.saveModelSchemaToFile()
    # enn.train(train[0],train[1])
    enn.trainKFolds(train[0],train[1],8)
    enn.restoreCheckpointWeights()
    enn.setWeights(enn.getWeights())
    history=enn.history
    preds,activations=enn.predict(test[0],True,True)
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=enn.eval(test[0],test[1])
    del enn
    Utils.printDict(eval_res,'Eval')
    Dataset.compareAndPrintLabels(preds,activations,test[1],show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}

def testGeneticallyTunedEnhancedNN_MultiNet(){
    metric=Metric.F1
    search_space=SearchSpace()
    search_space.add(3,3,SearchSpace.Type.INT,'networks') # amount of networks including the final
    # variables mandatory for every network
        # network a
    search_space.add(1,4,SearchSpace.Type.INT,'layers_0')
    search_space.add(2,8,SearchSpace.Type.INT,'layer_sizes_0')
    search_space.add(Utils.getEnumBorder(NodeType,False),NodeType.TANH,SearchSpace.Type.INT,'node_types_0')
    search_space.add(0,0.995,SearchSpace.Type.FLOAT,'dropouts_0')
        # network b
    search_space.add(1,4,SearchSpace.Type.INT,'layers_1')
    search_space.add(2,8,SearchSpace.Type.INT,'layer_sizes_1')
    search_space.add(Utils.getEnumBorder(NodeType,False),NodeType.TANH,SearchSpace.Type.INT,'node_types_1')
    search_space.add(0,0.995,SearchSpace.Type.FLOAT,'dropouts_1')
        # concatenation network
    search_space.add(1,4,SearchSpace.Type.INT,'layers_2')
    search_space.add(2,8,SearchSpace.Type.INT,'layer_sizes_2')
    search_space.add(Utils.getEnumBorder(NodeType,False),NodeType.TANH,SearchSpace.Type.INT,'node_types_2')
    search_space.add(0,0.995,SearchSpace.Type.FLOAT,'dropouts_2')
    # network variable that does not need to be specified for every single net
    search_space.add(0.0001,0.1,SearchSpace.Type.FLOAT,'alpha')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'bias')
    search_space.add(Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY,SearchSpace.Type.INT,'loss')
    search_space.add(Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),SearchSpace.Type.INT,'optimizer')
    # static variables for every net
    search_space.add(5,15,SearchSpace.Type.INT,'batch_size')
    search_space.add(15,30,SearchSpace.Type.INT,'patience_epochs')
    search_space.add(20,150,SearchSpace.Type.INT,'max_epochs')
    search_space.add(LabelEncoding.SPARSE,LabelEncoding.SPARSE,SearchSpace.Type.INT,'label_type')
    search_space.add(metric,metric,SearchSpace.Type.INT,'monitor_metric')
    search_space.add(True,True,SearchSpace.Type.BOOLEAN,'model_checkpoint')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'shuffle')
    search_space=Genome.enrichSearchSpace(search_space,multi_net_enhanced_nn=True)

    Genome.CACHE_WEIGHTS=False

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    features,labels=Dataset.filterDataset(features,labels,'Iris-setosa')
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    # labels,label_map_2=Dataset.encodeDatasetLabels(labels,label_type)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)
    train,test=Dataset.splitDataset(features,labels,.7)
    groups=[[0,2],[2,-1]]
    train[0]=Dataset.divideFeaturesIntoMultipleGroups(train[0],groups)
    test[0]=Dataset.divideFeaturesIntoMultipleGroups(test[0],groups)
    train=Dataset.balanceDataset(train[0],train[1])

    def train_callback(genome){
        nonlocal train
        kfolds=5
        preserve_weights=False # TODO fix when true, to avoid nan outputs
        train_features=train[0]
        train_labels=train[1]
        train_labels,_=Dataset.encodeDatasetLabels(train_labels,genome.getHyperparametersEncoder(True))
        input_size=[len(train_features[i][0]) for i in range(len(train_features))]
        output_size=len(train_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX,multi_net_enhanced_nn=True)
        search_maximum=hyperparameters.monitor_metric.isMaxMetric(hyperparameters.loss)
        enn=EnhancedNeuralNetwork(hyperparameters,name='iris_{}'.format(genome.id),verbose=False)
        enn.buildModel(input_size=input_size)
        enn.saveModelSchemaToFile('population_nets')
        enn.setWeights(genome.getWeights())
        enn.trainKFolds(train_features,train_labels,kfolds)
        if preserve_weights and hyperparameters.model_checkpoint{
            enn.restoreCheckpointWeights()
        }
        output=enn.getMetricMean(hyperparameters.monitor_metric.toKerasName(),True)
        if output!=output{ # Not a Number, ignore this genome
            Core.LOGGER.warn('Not a number metric mean')
            output=float('-inf') if search_maximum else float('inf')
        }
        if preserve_weights {
            genome.setWeights(enn.mergeWeights(genome.getWeights()))
        }
        del enn
        return output
    }

    verbose_natural_selection=True
    verbose_population_details=True
    population_start_size_enh=10
    max_gens=10
    max_age=5
    max_children=3
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    max_notables=5
    search_maximum=metric.isMaxMetric()
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,search_space,train_callback,population_start_size_enh,neural_genome=True,print_deltas=verbose_population_details)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')
    print('Evaluating best')

    def test_callback(genome){
        nonlocal test
        test_features=test[0]
        test_labels=test[1]
        test_labels,label_map_2=Dataset.encodeDatasetLabels(test_labels,genome.getHyperparametersEncoder(True))
        input_size=[len(test_features[i][0]) for i in range(len(test_features))]
        output_size=len(test_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX,multi_net_enhanced_nn=True)
        enn=EnhancedNeuralNetwork(hyperparameters,name='iris_{}'.format(genome.id),verbose=False)
        enn.buildModel(input_size=input_size)
        enn.saveModelSchemaToFile()
        enn.setWeights(genome.getWeights())
        print('Best genome encoded weights:',genome.getWeights(raw=True))
        preds,activations=enn.predict(test_features,True,True)
        del enn
        Dataset.compareAndPrintLabels(preds,activations,test_labels,show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
        Utils.printDict(Dataset.statisticalAnalysis(preds,test_labels),'Statistical Analysis')
    }

    test_callback(enh_elite.getBestGenome())
}

def testParallelEnhGenetic(){
    bkp=PopulationManager.SIMULTANEOUS_EVALUATIONS
    PopulationManager.SIMULTANEOUS_EVALUATIONS=0
    testEnhGenetic()
    PopulationManager.SIMULTANEOUS_EVALUATIONS=bkp
}


def testParallelGeneticallyTunedNN(){
    bkp=PopulationManager.SIMULTANEOUS_EVALUATIONS
    PopulationManager.SIMULTANEOUS_EVALUATIONS=0
    testGeneticallyTunedNN()
    PopulationManager.SIMULTANEOUS_EVALUATIONS=bkp
}

def testEnhGeneticStats(){
    def eggHolder(genome){
        # https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
        x=genome.dna[0]
        y=genome.dna[1]
        return -(y+47)*math.sin(math.sqrt(abs(y+(x/2)+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
    }

    def easom(genome){
        # https://www.sfu.ca/~ssurjano/easom.html TIME MINUS ONE // maximum -> x1=x2=pi -> y(x1,x2)=1
        x=genome.dna[0]
        y=genome.dna[1]
        return -(-math.cos(x)*math.cos(y)*math.exp(-(math.pow(x-math.pi,2)+math.pow(y-math.pi,2))))
    }

    PopulationManager.PRINT_REL_FREQUENCY=0

    verbose_natural_selection=False
    verbose_population_details=False
    print('Minimization')
    limits=SearchSpace()
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='x')
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='y')
    population_start_size_enh=300
    max_gens=100
    max_age=5
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=False
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,limits,eggHolder,population_start_size_enh,neural_genome=False,print_deltas=verbose_population_details)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    
    for g,stats in enumerate(enh_population.last_run_stats){
        print('Generation {}:'.format(g+1))
        Utils.printDict(stats,name=None,tabs=1)
    }

    
    print('Expected: (x: 512, y: 404.2319) = -959.6407')
    Utils.printDict(enh_elite.best,'Elite')
    del enh_elite
    del enh_population

    print()
    print('Maximization')
    limits=SearchSpace()
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='x')
    limits.add(-100,100,SearchSpace.Type.FLOAT,name='y')
    max_gens=100
    max_age=5
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=True
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,limits,easom,population_start_size_enh,neural_genome=False,print_deltas=verbose_population_details)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    for g,stats in enumerate(enh_population.last_run_stats){
        print('Generation {}:'.format(g+1))
        Utils.printDict(stats,name=None,tabs=1)
    }

    print('Expected: (x: 3.141592, y: 3.141592) = 1')
    Utils.printDict(enh_elite.best,'Elite')
    del enh_elite
    del enh_population
}

def runParallelGeneticallyTuneGeneticEnhancedAlgorithm(){
    bkp=PopulationManager.SIMULTANEOUS_EVALUATIONS
    PopulationManager.SIMULTANEOUS_EVALUATIONS=12

    def enhGenetic(genome){
        def eggHolder(genome){
            # https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
            x=genome.dna[0]
            y=genome.dna[1]
            return -(y+47)*math.sin(math.sqrt(abs(y+(x/2)+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
        }
        def easom(genome){
            # https://www.sfu.ca/~ssurjano/easom.html TIME MINUS ONE // maximum -> x1=x2=pi -> y(x1,x2)=1
            x=genome.dna[0]
            y=genome.dna[1]
            return -(-math.cos(x)*math.cos(y)*math.exp(-(math.pow(x-math.pi,2)+math.pow(y-math.pi,2))))
        }

        TUNE_ONE_INSTEAD_OF_ALL=False

        if TUNE_ONE_INSTEAD_OF_ALL {
            resetMtDnaPercentage=float(genome.dna[0])
            max_age=5
            max_children=4
            mutation_rate=0.1
            recycle_rate=0.13
            sex_rate=0.7
            willOfDPercentage=0.07
            recycleRateThreshold=0.05
            rank_type=GeneticRankType(2)
        }else{
            resetMtDnaPercentage=float(genome.dna[0])
            max_age=int(genome.dna[1])
            max_children=int(genome.dna[2])
            mutation_rate=float(genome.dna[3])
            recycle_rate=float(genome.dna[4])
            sex_rate=float(genome.dna[5])
            willOfDPercentage=float(genome.dna[6])
            recycleRateThreshold=float(genome.dna[7])
            rank_type=GeneticRankType(int(genome.dna[8]))
        }

        max_gens=80
        population_start_size_enh=200
        max_notables=1

        limits=SearchSpace()
        limits.add(-512,512,SearchSpace.Type.FLOAT,name='x')
        limits.add(-512,512,SearchSpace.Type.FLOAT,name='y')
        search_maximum=False
        enh_elite=HallOfFame(max_notables, search_maximum)
        en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate,rank_type=rank_type,resetMtDnaPercentage=resetMtDnaPercentage,willOfDPercentage=willOfDPercentage,recycleRateThreshold=recycleRateThreshold)
        enh_population=PopulationManager(en_ga,limits,eggHolder,population_start_size_enh,neural_genome=False,print_deltas=False,force_sequential=True)
        enh_population.hall_of_fame=enh_elite
        enh_population.naturalSelection(max_gens,False,False)
        part_1_value=enh_elite.best['output']/-959.6407*100.0
        part_1_speed=(max_gens-enh_elite.best['generation'])/max_gens*100.0
        del enh_elite
        del enh_population

        limits=SearchSpace()
        limits.add(-100,100,SearchSpace.Type.FLOAT,name='x')
        limits.add(-100,100,SearchSpace.Type.FLOAT,name='y')
        search_maximum=True
        enh_elite=HallOfFame(max_notables, search_maximum)
        en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate,rank_type=rank_type,resetMtDnaPercentage=resetMtDnaPercentage,willOfDPercentage=willOfDPercentage,recycleRateThreshold=recycleRateThreshold)
        enh_population=PopulationManager(en_ga,limits,easom,population_start_size_enh,neural_genome=False,print_deltas=False,force_sequential=True)
        enh_population.hall_of_fame=enh_elite
        enh_population.naturalSelection(max_gens,False,False)
        part_2_value=enh_elite.best['output']/1*100.0
        part_2_speed=(max_gens-enh_elite.best['generation'])/max_gens*100.0
        del enh_elite
        del enh_population

        return (part_1_value*5+part_2_value*5+part_1_speed+part_2_speed)/12.0
    }

    def after_gen_callback(args_list){
        pop_size=args_list[0]
        g=args_list[1]
        best_out=args_list[2]
        timestamp_s=args_list[3]
        population=args_list[4]
        hall_of_fame=args_list[5]
        print('Best at gen {} is {} - {}'.format(g,best_out,str(hall_of_fame.notables[0])))
    }

    verbose=True
    limits=SearchSpace()
    limits.add(1,50,SearchSpace.Type.FLOAT,name='reset_mt_dna')
    limits.add(2,10,SearchSpace.Type.INT,name='age')
    limits.add(2,6,SearchSpace.Type.INT,name='children')
    limits.add(0.05,0.35,SearchSpace.Type.FLOAT,name='mutation')
    limits.add(0.05,0.35,SearchSpace.Type.FLOAT,name='recycle')
    limits.add(0.5,0.9,SearchSpace.Type.FLOAT,name='sex')
    limits.add(0.01,0.2,SearchSpace.Type.FLOAT,name='D')
    limits.add(0.01,0.2,SearchSpace.Type.FLOAT,name='trash')
    limits.add(0,2,SearchSpace.Type.INT,name='rank')
    population_size=100
    max_gens=50
    mutation_rate=0.2
    sex_rate=0.6
    search_maximum=True
    max_notables=5
    elite_min=HallOfFame(max_notables, search_maximum)
    ga=StandardGeneticAlgorithm(search_maximum,mutation_rate, sex_rate)
    population=PopulationManager(ga,limits,enhGenetic,population_size,neural_genome=False,print_deltas=verbose,after_gen_callback=after_gen_callback)
    population.hall_of_fame=elite_min
    population.naturalSelection(max_gens,verbose)
    for individual in elite_min.notables{
        print(str(individual))
    }
    Utils.printDict(elite_min.best,'Elite')
    PopulationManager.SIMULTANEOUS_EVALUATIONS=bkp
}

@profile
def testParallelGeneticallyTunedNN_withSharedMemAndNumpy(){
    bkp=PopulationManager.SIMULTANEOUS_EVALUATIONS
    PopulationManager.SIMULTANEOUS_EVALUATIONS=0

    metric=Metric.ACCURACY
    search_space=SearchSpace()
    search_space.add(1,2,SearchSpace.Type.INT,'layers')
    search_space.add(5,15,SearchSpace.Type.INT,'batch_size')
    search_space.add(0.0001,0.1,SearchSpace.Type.FLOAT,'alpha')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'shuffle')
    search_space.add(5,10,SearchSpace.Type.INT,'patience_epochs')
    search_space.add(20,40,SearchSpace.Type.INT,'max_epochs')
    search_space.add(Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY,SearchSpace.Type.INT,'loss')
    search_space.add(LabelEncoding.SPARSE,LabelEncoding.SPARSE,SearchSpace.Type.INT,'label_type')
    search_space.add(Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),SearchSpace.Type.INT,'optimizer')
    search_space.add(metric,metric,SearchSpace.Type.INT,'monitor_metric')
    search_space.add(True,True,SearchSpace.Type.BOOLEAN,'model_checkpoint')
    search_space.add(2,8,SearchSpace.Type.INT,'layer_sizes')
    search_space.add(Utils.getEnumBorder(NodeType,False),NodeType.TANH,SearchSpace.Type.INT,'node_types')
    search_space.add(0,0.995,SearchSpace.Type.FLOAT,'dropouts')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'bias')
    search_space=Genome.enrichSearchSpace(search_space)

    Genome.CACHE_WEIGHTS=False

    dataset_increase_factor=5
    
    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    for i in range(dataset_increase_factor){
        features+=features
        labels+=labels
    }

    features,labels=Dataset.filterDataset(features,labels,'Iris-setosa')
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    features,labels=Dataset.balanceDataset(features,labels)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)

    train,test=Dataset.splitDataset(features,labels,.7)
    train_features=train[0]
    train_labels=train[1]
    train_labels,_=Dataset.encodeDatasetLabels(train_labels,LabelEncoding(search_space['label_type'].min_value)) # must be outside callback to be serialized

    train_features=NeuralNetwork.FormatData(train_features) # to np array
    train_labels=NeuralNetwork.FormatData(train_labels) # to np array

    train_features,train_labels=Dataset.shuffleDataset(train_features,train_labels) # shuffle again to test

    train_features=NeuralNetwork.createSharedNumpyArray(train_features) # put array in shared memory
    train_labels=NeuralNetwork.createSharedNumpyArray(train_labels) # put array in shared memory
    print('train_features',train_features.shape)
    print('train_labels',train_labels.shape)

    def train_callback(genome){
        nonlocal train_features,train_labels
        kfolds=5
        preserve_weights=False # TODO fix when true, to avoid nan outputs
        input_size=len(train_features[0])
        output_size=len(train_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX)
        search_maximum=hyperparameters.monitor_metric.isMaxMetric(hyperparameters.loss)
        nn=StandardNeuralNetwork(hyperparameters,name='iris_{}'.format(genome.id),verbose=False)
        nn.buildModel(input_size=input_size)
        nn.setWeights(genome.getWeights())
        nn.trainKFolds(train_features,train_labels,kfolds)
        nn.trainRollingForecast(train_features,train_labels) # just to test numpy arrays
        if preserve_weights and hyperparameters.model_checkpoint{
            nn.restoreCheckpointWeights()
        }
        output=nn.getMetricMean(hyperparameters.monitor_metric.toKerasName(),True)
        if output!=output{ # Not a Number, ignore this genome
            Core.LOGGER.warn('Not a number metric ('+str(hyperparameters.monitor_metric.toKerasName())+') mean of '+str(nn.getMetric(hyperparameters.monitor_metric.toKerasName(),True)))
            output=float('-inf') if search_maximum else float('inf')
        }
        if preserve_weights {
            genome.setWeights(nn.mergeWeights(genome.getWeights()))
        }
        del nn
        return output
    }

    verbose_natural_selection=True
    verbose_population_details=True
    population_start_size_enh=20
    max_gens=4
    max_age=5
    max_children=2
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    max_notables=3
    search_maximum=metric.isMaxMetric()
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,search_space,train_callback,population_start_size_enh,neural_genome=True,print_deltas=verbose_population_details)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')
    print('Evaluating best')

    def test_callback(genome){
        nonlocal test
        test_features=test[0]
        test_labels=test[1]
        test_labels,label_map_2=Dataset.encodeDatasetLabels(test_labels,genome.getHyperparametersEncoder(False))
        input_size=len(test_features[0])
        output_size=len(test_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX)
        nn=StandardNeuralNetwork(hyperparameters,name='iris_{}'.format(genome.id),verbose=False)
        nn.buildModel(input_size=input_size)
        nn.saveModelSchemaToFile()
        nn.setWeights(genome.getWeights())
        print('Best genome encoded weights:',genome.getWeights(raw=True))
        preds,activations=nn.predict(test_features,True,True)
        del nn
        Dataset.compareAndPrintLabels(preds,activations,test_labels,show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
        Utils.printDict(Dataset.statisticalAnalysis(preds,test_labels),'Statistical Analysis')
    }

    test_callback(enh_elite.getBestGenome())


    PopulationManager.SIMULTANEOUS_EVALUATIONS=bkp
}

def testEnhancedNN_MultiNet_withSharedMemAndNumpy(){
    label_type=LabelEncoding.SPARSE

    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,label_type)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)
    train,test=Dataset.splitDataset(features,labels,.7)
    groups=[[0,2],[2,-1]]
    train[0]=Dataset.divideFeaturesIntoMultipleGroups(train[0],groups)
    test[0]=Dataset.divideFeaturesIntoMultipleGroups(test[0],groups)

    train[0]=NeuralNetwork.createSharedNumpyArray(train[0]) # put array in shared memory
    train[1]=NeuralNetwork.createSharedNumpyArray(train[1]) # put array in shared memory

    amount_of_networks=3
    input_size=[len(train[0][i][0]) for i in range(len(train[0]))]
    intermediary_size=[2,1]
    output_size=len(train[1][0])
    layers=[2,2,2]
    dropouts=[0,0,0]
    bias=[True,True,True]
    layer_sizes=[[3,intermediary_size[0]],[3,intermediary_size[1]],[4,output_size]]
    node_types=[[NodeType.TANH,NodeType.TANH],[NodeType.TANH,NodeType.TANH],[NodeType.TANH,NodeType.SOFTMAX]]
    batch_size=5
    alpha=[0.01,0.01,0.01]
    shuffle=True
    optimizer=[Optimizers.SGD,Optimizers.SGD,Optimizers.SGD]
    patience_epochs=15
    max_epochs=100
    loss=[Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY,Loss.CATEGORICAL_CROSSENTROPY]
    monitor_metric=Metric.RAW_LOSS
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric,amount_of_networks=amount_of_networks)

    enn=EnhancedNeuralNetwork(hyperparameters,name='iris',verbose=True)
    enn.buildModel(input_size=input_size)
    enn.saveModelSchemaToFile()
    # enn.train(train[0],train[1])
    enn.trainKFolds(train[0],train[1],8)
    enn.trainRollingForecast(train[0],train[1]) # just to test numpy arrays
    enn.trainNoValidation(train[0],train[1]) # just to test numpy arrays
    enn.restoreCheckpointWeights()
    enn.setWeights(enn.getWeights())
    history=enn.history
    preds,activations=enn.predict(test[0],True,True)
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=enn.eval(test[0],test[1])
    del enn
    Utils.printDict(eval_res,'Eval')
    Dataset.compareAndPrintLabels(preds,activations,test[1],show_positives=False,equivalence_table_1=label_map,equivalence_table_2=label_map_2,logger=None)
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}

def runGenExperimentsOnMath(){
    results=[]
    mutation_rates=(0.1,0.2)
    for mutation_rate in mutation_rates{
        print('Mutation rate of: {}'.format(mutation_rate))
        results.append(testStdVsEnhGenetic(mutation_rate,max_age=5,max_children=4,recycle_rate=0.33,sex_rate=0.83))
        print()
        print()
        print()
    }
    print()
    print()
    print()
    print()
    print('Summary:')
    for r,result in enumerate(results){
        print('Mutation rate {}:'.format(mutation_rates[r]))
        print('\tEggHolder (-959.6407):')
        for line in result[0].split('\n'){
            print('\t\t'+line)
        }
        print()
        print('\tEasom (1):')
        for line in result[1].split('\n'){
            print('\t\t'+line)
        }
        print()
        print()
    }
}


# testStdGenetic()
# testEnhGenetic()
# testStdVsEnhGenetic()
# testNNIntLabel()
# testNNBinLabel_KFolds()
# testGeneticallyTunedNN()
# testCustomEncodings()
# testEnhancedNN_SingleNet()
# testEnhancedNN_MultiNet()
# testGeneticallyTunedEnhancedNN_MultiNet()
# testParallelEnhGenetic()
# testParallelGeneticallyTunedNN()
# testEnhGeneticStats()
# runParallelGeneticallyTuneGeneticEnhancedAlgorithm()
# testParallelGeneticallyTunedNN_withSharedMemAndNumpy()
# testEnhancedNN_MultiNet_withSharedMemAndNumpy()
runGenExperimentsOnMath()