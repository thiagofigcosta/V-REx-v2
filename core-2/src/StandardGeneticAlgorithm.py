#!/bin/python
# -*- coding: utf-8 -*-

from GeneticAlgorithm import GeneticAlgorithm
from Enums import GeneticRankType
from Utils import Utils

class StandardGeneticAlgorithm(GeneticAlgorithm){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    def __init__(self, looking_highest_fitness, mutation_rate, sex_rate, rank_type=GeneticRankType.RELATIVE){
        super().__init__(looking_highest_fitness)
        self.mutation_rate=mutation_rate
        self.sex_rate=sex_rate
        self.rank_type=rank_type
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
            potential_parents=[]
            for c in range(2){
                roulette_number=Utils.randomFloat(0,fitness_sum)
                current_roulette=0
                for individual in individuals {
                    current_roulette+=individual.fitness+offset
                    if current_roulette>=roulette_number {
                        potential_parents.append(individual)
                        break
                    }
                }
            }
            children=self.sex(potential_parents[0],potential_parents[1])
            next_gen+=children
        }
        for individual in individuals {
            if Utils.LazyCore.freeMemManually(){
                del individual
            }
        }
        individuals.clear()
        return next_gen
    }

    def fit(self, individuals){
        signal=1
        if not self.looking_highest_fitness{
            signal=-1
        }
        for individual in individuals{
            individual.fitness=individual.output*signal
        }
        if self.rank_type==GeneticRankType.RELATIVE{
            individuals.sort()
            for i in range(len(individuals)){
                individuals[i].fitness=100.0/float(len(individuals)-i+2)
            }
        }elif self.rank_type==GeneticRankType.INCREMENTAL{
            individuals.sort()
            for i in range(len(individuals)){
                individuals[i].fitness=i+1
            }
        }
        return individuals
    }

    def sex(self, father, mother){
        family=[]
        if Utils.random()<self.sex_rate{
            amount_of_children=2
            children=[[] for _ in range(amount_of_children)]
            for i in range(len(father.dna)){
                children_genes=GeneticAlgorithm.geneShare(Utils.random(),father.dna[i],mother.dna[i])
                children[0].append(children_genes[0])
                children[1].append(children_genes[1])
            }
            for i in range(len(children)){
                children[i]=mother.makeChild(children[i])
            }
            family+=children
        }else{
            family.append(father.copy())
            family.append(mother.copy())
        }
        return family
    }

    def mutate(self, individuals){
        for individual in individuals{
            self.mutateIndividual(individual,force=False)
        }
        return individuals
    }

    def mutateIndividual(self, individual, force=False){
        for i in range(len(individual.dna)){
            if force or Utils.random()<self.mutation_rate{
                individual.dna[i]=GeneticAlgorithm.geneMutation(self.randomize(),individual.dna[i])
            }
        }
        individual.fixlimits()
    }

    def enrichSpace(self, space){
        return space
    }

    def randomize(self){
        r=Utils.randomFloat(0,0.1)
        if (Utils.random()>0.5){
            r=-(1+r)
        }else{
            r=(1+r)
        }
        return r
    }
}