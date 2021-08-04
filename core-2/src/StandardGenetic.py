#!/bin/python
# -*- coding: utf-8 -*-

from GeneticAlgorithm import GeneticAlgorithm
from Enums import StdGeneticRankType

class StandardGenetic(GeneticAlgorithm){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    def __init__(self, mutation_rate, sex_rate, rankType=StdGeneticRankType.RELATIVE){
        self.looking_highest_fitness=None
        self.mutation_rate=mutation_rate
        self.sex_rate=sex_rate
        self.rank_type=rankType
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

    def genRandomFactor(self){
        pass
    }
}