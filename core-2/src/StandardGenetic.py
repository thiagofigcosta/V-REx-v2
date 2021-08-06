#!/bin/python
# -*- coding: utf-8 -*-

from GeneticAlgorithm import GeneticAlgorithm
from Enums import StdGeneticRankType

class StandardGenetic(GeneticAlgorithm){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    def __init__(self, looking_highest_fitness, mutation_rate, sex_rate, rankType=StdGeneticRankType.RELATIVE){
        super().__init__(looking_highest_fitness)
        self.mutation_rate=mutation_rate
        self.sex_rate=sex_rate
        self.rank_type=rankType
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

    def mutateIndividual(self, individual, force=False){
        raise Exception('Not implemented yet!')
    }

    def enrichSpace(self, space){
        raise Exception('Not implemented yet!')
    }

    def copy(self){
        raise Exception('Not implemented yet!')
    }

    def randomize(self){
        raise Exception('Not implemented yet!')
    }
}