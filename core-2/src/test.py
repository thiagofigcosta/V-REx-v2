#!/bin/python

import math
from SearchSpace import SearchSpace
from Utils import Utils
from Logger import Logger
from Core import Core
from HallOfFame import HallOfFame
from StandardGenetic import StandardGenetic
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

testStdGenetic()