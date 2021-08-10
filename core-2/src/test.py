#!/bin/python

import math
from SearchSpace import SearchSpace
from Utils import Utils
from Logger import Logger
from Core import Core
from Dataset import Dataset
from Genome import Genome
from Enums import LabelEncoding,NodeType,Loss,Metric
from Hyperparameters import Hyperparameters
from HallOfFame import HallOfFame
from StandardNeuralNetwork import StandardNeuralNetwork
from StandardGenetic import StandardGenetic
from EnhancedGenetic import EnhancedGenetic
from PopulationManager import PopulationManager

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
    ga=StandardGenetic(search_maximum,mutation_rate, sex_rate)
    population=PopulationManager(ga,limits,eggHolder,population_size,neural_genome=False,print_deltas=verbose,after_gen_callback=lambda:print('After gen'))
    population.hall_of_fame=elite_min
    population.naturalSelection(max_gens,verbose)
    print('Expected: (x: 512, y: 404.2319) = -959.6407')
    for individual in elite_min.notables{
        print(str(individual))
    }
    Utils.printDict(elite_min.best,'Elite')

    runned_after_gen=False
    def afterGen(){
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
    population_size=100
    max_gens=100
    mutation_rate=0.1
    sex_rate=0.7
    search_maximum=True
    max_notables=5
    elite_max=HallOfFame(max_notables, search_maximum)
    ga=StandardGenetic(search_maximum,mutation_rate, sex_rate)
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
    max_age=10
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=False
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGenetic(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,limits,eggHolder,population_start_size_enh,neural_genome=False,print_deltas=verbose_population_details,after_gen_callback=lambda:print('After gen') if add_callback_after_gen else None)
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
    population_size=100
    max_gens=100
    max_age=10
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=True
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGenetic(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,limits,easom,population_start_size_enh,neural_genome=False,print_deltas=verbose_population_details,after_gen_callback=afterGen if add_callback_after_gen else None)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)

    print('Expected: (x: 3.141592, y: 3.141592) = 1')
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')
}

def testStdVsEnhGenetic(){
    def eggHolder(genome){
        # https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
        x=genome.dna[0]
        y=genome.dna[1]
        return -(y+47)*math.sin(math.sqrt(abs(y+(x/2)+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
    }

    Core.FREE_MEMORY_MANUALLY=False

    verbose=False
    print('Standard vs Enhanced:')
    tests=50
    limits=SearchSpace()
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='x')
    limits.add(-512,512,SearchSpace.Type.FLOAT,name='y')
    population_start_size_enh=300
    population_start_size_std=780
    max_gens=100
    max_age=10
    max_children=4
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    search_maximum=False
    max_notables=5
    results={'standard':[],'enhanced':[]}
    for x in range(tests){
        print('Test {} of {}'.format(x+1,tests))
        std_elite=HallOfFame(max_notables, search_maximum)
        std_ga=StandardGenetic(search_maximum,mutation_rate, sex_rate)
        std_population=PopulationManager(std_ga,limits,eggHolder,population_start_size_std)
        std_population.hall_of_fame=std_elite
        std_population.naturalSelection(max_gens)
        std_result=std_elite.best
        results['standard'].append(std_result)
        if Core.FREE_MEMORY_MANUALLY==True{
            del std_elite
            del std_population
        }

        enh_elite=HallOfFame(max_notables, search_maximum)
        en_ga=EnhancedGenetic(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
        enh_population=PopulationManager(en_ga,limits,eggHolder,population_start_size_enh)
        enh_population.hall_of_fame=enh_elite
        enh_population.naturalSelection(max_gens)
        enh_result=enh_elite.best
        results['enhanced'].append(enh_result)
        if Core.FREE_MEMORY_MANUALLY==True{
            del enh_elite
            del enh_population
        }
    }
    std_mean=[0.0,0.0]
    for std_result in results['standard']{
        std_mean[0]+=std_result['generation']
        std_mean[1]+=std_result['output']
        print('Standard Best ({}): {}'.format(std_result['generation'],std_result['output']))
    }
    std_mean[0]/=tests
    std_mean[1]/=tests
    print()
    enh_mean=[0,0]
    for enh_result in results['enhanced']{
        enh_mean[0]+=enh_result['generation']
        enh_mean[1]+=enh_result['output']
        print('Enhanced Best ({}): {}'.format(enh_result['generation'],enh_result['output']))
    }
    enh_mean[0]/=tests
    enh_mean[1]/=tests
    print()
    Utils.printDict(results,'Results')
    print()
    print('Standard Mean ({}): {} | Enhanced Mean ({}): {}'.format(std_mean[0],std_mean[1],enh_mean[0],enh_mean[1]))

    Core.FREE_MEMORY_MANUALLY=True
}


def testNNIntLabel(){
    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    features,labels=Dataset.filterDataset(features,labels,'Iris-setosa')
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,LabelEncoding.INCREMENTAL)
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
    label_type=LabelEncoding.INCREMENTAL
    node_types=[NodeType.TANH,NodeType.SOFTMAX]
    batch_size=5
    alpha=0.01
    shuffle=True
    adam=True
    patience_epochs=0
    max_epochs=100
    loss=Loss.BINARY_CROSSENTROPY
    monitor_metric=Metric.F1
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, adam, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric)

    nn=StandardNeuralNetwork(hyperparameters,dataset_name='iris',verbose=True)
    nn.buildModel(input_size)
    nn.train(train[0],train[1],val[0],val[1])
    history=nn.history
    Utils.printDict(history,'History')
    preds=nn.predict(test[0])
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=nn.eval(test[0],test[1])
    nn.clearCache()
    del nn
    Utils.printDict(eval_res,'Eval')

    for i in range(len(preds)){
        if (Dataset.labelToVanilla(preds[i])==test[1][i]){
            print('OK:',Dataset.translateLabelFromOutput(preds[i],label_map,label_map_2))
        }else{
            print('Fail:','Exptected:{} But was: {}'.format(Dataset.translateLabelFromOutput(test[1][i],label_map,label_map_2),Dataset.translateLabelFromOutput(preds[i],label_map,label_map_2)))
        }
    }
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}


def testNNBinLabel_KFolds(){
    features,labels=Dataset.readLabeledCsvDataset(Utils.getResource(Dataset.getDataset('iris.data')))
    labels,label_map=Dataset.enumfyDatasetLabels(labels)
    labels,label_map_2=Dataset.encodeDatasetLabels(labels,LabelEncoding.BINARY)
    features,scale=Dataset.normalizeDatasetFeatures(features)
    features,labels=Dataset.shuffleDataset(features,labels)
    train,test=Dataset.splitDataset(features,labels,.7)

    input_size=len(train[0][0])
    output_size=len(train[1][0])
    layers=2
    dropouts=0
    bias=True
    layer_sizes=[5,output_size]
    label_type=LabelEncoding.BINARY
    node_types=[NodeType.TANH,NodeType.SOFTMAX]
    batch_size=5
    alpha=0.01
    shuffle=True
    adam=True
    patience_epochs=15
    max_epochs=100
    loss=Loss.CATEGORICAL_CROSSENTROPY
    monitor_metric=Metric.F1
    hyperparameters=Hyperparameters(batch_size, alpha, shuffle, adam, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric)

    nn=StandardNeuralNetwork(hyperparameters,dataset_name='iris',verbose=True)
    nn.buildModel(input_size)
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
    preds=nn.predict(test[0])
    print('Predicted[0]:',Dataset.translateLabelFromOutput(preds[0],label_map,label_map_2))
    eval_res=nn.eval(test[0],test[1])
    del nn
    Utils.printDict(eval_res,'Eval')

    total=0
    correct=0
    wrong=0
    for i in range(len(preds)){
        total+=1
        if (Dataset.labelToVanilla(preds[i])==test[1][i]){
            correct+=1
            print('OK:',Dataset.translateLabelFromOutput(preds[i],label_map,label_map_2))
        }else{
            wrong+=1
            print('Fail:','Exptected:{} But was: {}'.format(Dataset.translateLabelFromOutput(test[1][i],label_map,label_map_2),Dataset.translateLabelFromOutput(preds[i],label_map,label_map_2)))
        }
    }
    print('Total:',total,'-','Correct:',correct,'-','Wrong:',wrong,'-','%:','{:.2f}'.format(correct*100/float(total)))
    Utils.printDict(Dataset.statisticalAnalysis(preds,test[1]),'Statistical Analysis')
}

def testGeneticallyTunedNN(){
    search_space=SearchSpace()
    search_space.add(1,4,SearchSpace.Type.INT,'layers')
    search_space.add(5,15,SearchSpace.Type.INT,'batch_size')
    search_space.add(0.0001,0.1,SearchSpace.Type.FLOAT,'alpha')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'shuffle')
    search_space.add(15,30,SearchSpace.Type.INT,'patience_epochs')
    search_space.add(20,150,SearchSpace.Type.INT,'max_epochs')
    search_space.add(Loss.BINARY_CROSSENTROPY,Loss.BINARY_CROSSENTROPY,SearchSpace.Type.INT,'loss')
    search_space.add(LabelEncoding.BINARY,LabelEncoding.BINARY,SearchSpace.Type.INT,'label_type')
    search_space.add(False,True,SearchSpace.Type.BOOLEAN,'adam')
    search_space.add(Metric.RAW_LOSS,Metric.RAW_LOSS,SearchSpace.Type.INT,'monitor_metric')
    search_space.add(True,True,SearchSpace.Type.BOOLEAN,'model_checkpoint')
    search_space.add(2,8,SearchSpace.Type.INT,'layer_sizes')
    search_space.add(Utils.getEnumBorder(NodeType,False),Utils.getEnumBorder(NodeType,True),SearchSpace.Type.INT,'node_types')
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
    search_maximum=False

    def train_callback(genome){
        nonlocal train,search_maximum
        kfolds=5
        train_features=train[0]
        train_labels=train[1]
        train_labels,_=Dataset.encodeDatasetLabels(train_labels,genome.getHyperparametersEncoder())
        input_size=len(train_features[0])
        output_size=len(train_labels[0])
        hyperparameters=genome.toHyperparameters(output_size,NodeType.SOFTMAX)
        nn=StandardNeuralNetwork(hyperparameters,dataset_name='iris_{}'.format(genome.id),verbose=False)
        nn.buildModel(input_size)
        nn.trainKFolds(train_features,train_labels,kfolds)
        output=nn.getMetricMean(hyperparameters.monitor_metric.toKerasName(),True)
        if output!=output{ #Not a Number
            Core.LOGGER.warn('Not a number metric mean')
            output=float('-inf') if search_maximum else float('inf')
        }
        genome.setWeights(nn.getWeights())
        del nn
        return output
    }

    verbose_natural_selection=True
    verbose_population_details=True
    population_start_size_enh=10
    max_gens=10
    max_age=2
    max_children=3
    mutation_rate=0.1
    recycle_rate=0.13
    sex_rate=0.7
    max_notables=5
    enh_elite=HallOfFame(max_notables, search_maximum)
    en_ga=EnhancedGenetic(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
    enh_population=PopulationManager(en_ga,search_space,train_callback,population_start_size_enh,neural_genome=True,print_deltas=verbose_population_details)
    enh_population.hall_of_fame=enh_elite
    enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
    
    for individual in enh_elite.notables{
        print(str(individual))
    }
    Utils.printDict(enh_elite.best,'Elite')
}

# testStdGenetic()
# testEnhGenetic()
# testStdVsEnhGenetic()
# testNNIntLabel()
# testNNBinLabel_KFolds()
testGeneticallyTunedNN()