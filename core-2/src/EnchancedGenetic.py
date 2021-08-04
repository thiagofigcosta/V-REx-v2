#!/bin/python
# -*- coding: utf-8 -*-

from GeneticAlgorithm import GeneticAlgorithm

class EnchancedGenetic(GeneticAlgorithm){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    WILL_OF_D_PERCENT=0.07
    RECYCLE_THRESHOLD_PERCENT=0.03
    CUTOFF_POPULATION_LIMIT=1.3

    def __init__(self, max_children, max_age, mutation_rate, sex_rate, recycle_rate){
        self.looking_highest_fitness=None
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
        self.current_population_size=None
    }

    def select(self, currentGen){
        pass
    }

    def fit(self, currentGen){
        pass
    }

    def sex(self, father, mother){
        pass
    }

    def mutate(self, individuals){
        pass
    }

    def enrichSpace(self, space){
        pass
    }

    def copy(self){
        pass
    }

    def age(self, individual, cur_population_size){
        pass
    }

    def mutateIndividual(self, individual, force=False){
        pass
    }

    def isRelative(self, father, mother){
        pass
    }

    def randomize(self){
        pass
    }

    def getLifeLeft(self,individual){
        pass
    }
    
    def recycleBadIndividuals(self, individuals){
        pass
    }

    def calcBirthRate(self,cur_population_size){
        pass
    }

}