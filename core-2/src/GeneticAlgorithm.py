#!/bin/python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class GeneticAlgorithm(ABC){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    def __init__(self,looking_highest_fitness){
        self.looking_highest_fitness=looking_highest_fitness
    }

    @abstractmethod
    def select(self, individuals){
        pass
    }

    @abstractmethod
    def fit(self, individuals){
        pass
    }

    @abstractmethod
    def sex(self, father, mother){
        pass
    }

    @abstractmethod
    def mutate(self, individuals){
        pass
    }

    @abstractmethod
    def mutateIndividual(self, individual, force=False){
        pass
    }

    @abstractmethod
    def enrichSpace(self, space){
        pass
    }

    @abstractmethod
    def randomize(self){
        pass
    }

    @abstractmethod
    def copy(self){
        pass
    }

}
