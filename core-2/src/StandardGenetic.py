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
        # roulette wheel
        individuals.sort()
        min_fitness=individuals[0].fitness
        offset=0
        fitness_sum=0
        if min_fitness<0 {
            offset=abs(min_fitness)
        }
        for individual in individuals{
            fitness_sum+=individual.fitness+offset
        }
        next_gen=[]
        for i in range(int(len(individuals)/2)){
            parents=[]
            for c in range(2){
                roulette_number=Utils.randomFloat(0,fitness_sum)
                current_roulette=0
                for individual in individuals {
                    current_roulette+=individual.fitness+offset
                    if current_roulette>=roulette_number {
                        parents.append(individual)
                        break
                    }
                }
            }
            children=self.sex(parents[0],parents[1])
            next_gen+=children
        }
        for individual in individuals {
            del individual
        }
        individuals=None
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