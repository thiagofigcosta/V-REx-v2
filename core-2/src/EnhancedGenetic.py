#!/bin/python
# -*- coding: utf-8 -*-

from GeneticAlgorithm import GeneticAlgorithm

class EnhancedGenetic(GeneticAlgorithm){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    WILL_OF_D_PERCENT=0.07
    RECYCLE_THRESHOLD_PERCENT=0.03
    CUTOFF_POPULATION_LIMIT=1.3

    def __init__(self, looking_highest_fitness, max_children, max_age, mutation_rate, sex_rate, recycle_rate){
        super().__init__(looking_highest_fitness)
        self.max_population=None
        self.index_age=None
        self.index_max_age=None
        self.index_max_children=None
        self.max_age=max_age
        self.max_children=max_age
        self.mutation_rate=mutation_rate
        self.sex_rate=sex_rate
        self.recycle_rate=recycle_rate
        self.current_population_size=0
    }

    def select(self, individuals){
        raise Exception('Not implemented yet!')
    }

    def fit(self, individuals){
        raise Exception('Not implemented yet!')
    }

    def sex(self, father, mother){
        raise Exception('Not implemented yet!')
    }

    def mutate(self, individuals){
        raise Exception('Not implemented yet!')
    }

    def enrichSpace(self, space){
        raise Exception('Not implemented yet!')
    }

    def copy(self){
        raise Exception('Not implemented yet!')
    }

    def age(self, individual, cur_population_size){
        raise Exception('Not implemented yet!')
    }

    def mutateIndividual(self, individual, force=False){
        raise Exception('Not implemented yet!')
    }

    def isRelative(self, father, mother){
        raise Exception('Not implemented yet!')
    }

    def randomize(self){
        raise Exception('Not implemented yet!')
    }

    def getLifeLeft(self,individual){
        raise Exception('Not implemented yet!')
    }
    
    def recycleBadIndividuals(self, individuals){
        raise Exception('Not implemented yet!')
    }

    def calcBirthRate(self,cur_population_size){
        raise Exception('Not implemented yet!')
    }

}