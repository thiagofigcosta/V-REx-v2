#!/bin/python

import math
from SearchSpace import SearchSpace
from Utils import Utils
from Logger import Logger
from Core import Core
from HallOfFame import HallOfFame
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
    std_mean=(0.0,0.0)
    for std_result in results['standard']{
        std_mean[0]+=std_result['generation']
        std_mean[1]+=std_result['output']
        print('Standard Best ({}): {}'.format(std_result['generation'],std_result['output']))
    }
    std_mean[0]/=tests
    std_mean[1]/=tests

    enh_mean=(0,0)
    for enh_result in results['enhanced']{
        enh_mean[0]+=enh_result['generation']
        enh_mean[1]+=enh_result['output']
        print('Enhanced Best ({}): {}'.format(enh_result['generation'],enh_result['output']))
    }
    enh_mean[0]/=tests
    enh_mean[1]/=tests
    print('Standard Mean ({}): {} | Enhanced Mean ({}): {}'.format(std_mean[0],std_mean[1],enh_mean[0],enh_mean[1]))

    # Utils.printDict(elite_min.best,'Elite')

    Core.FREE_MEMORY_MANUALLY=True
}


def testNNIntLabel(){
    pass
}

# testStdGenetic()
# testEnhGenetic()
#testStdVsEnhGenetic()
testNNIntLabel()