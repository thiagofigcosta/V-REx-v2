#!/bin/python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from Utils import Utils

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

    @staticmethod
    def geneShare(gene_share,gene_a,gene_b){
        if type(gene_a) in (float,int){
            new_gene_a=gene_share*gene_a+(1-gene_share)*gene_b
            new_gene_b=(1-gene_share)*gene_a+gene_share*gene_b
            return new_gene_a,new_gene_b
        }elif type(gene_a) in (bool){
            if gene_a==gene_b {
                return gene_a
            }else{
                bool_result=Utils.random()>.5
                return bool_result, bool_result
            }
        }else{
            raise Exception('Dont know how to gene share {}'.format(type(gene_a)))
        }
    }

    @staticmethod
    def geneMutation(radiation,gene){
        if type(gene) in (float,int){
            return gene*radiation
        }elif type(gene) in (bool){
            if (Utils.random()>.3){
                return gene
            }else{
                return not gene
            }
        }else{
            raise Exception('Dont know how to gene share {}'.format(type(gene)))
        }
    }
}
