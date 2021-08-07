#!/bin/python
# -*- coding: utf-8 -*-

from GeneticAlgorithm import GeneticAlgorithm
from Enums import StdGeneticRankType
from Utils import Utils

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
        individuals=[]
        return next_gen
    }

    def fit(self, individuals){
        signal=1
        if not self.looking_highest_fitness{
            signal=-1
        }
        for individual in individuals{
            if self.rank_type in (StdGeneticRankType.ABSOLUTE,StdGeneticRankType.RELATIVE){
                individual.fitness=individual.output*signal
            }
        }
        if self.rank_type==StdGeneticRankType.RELATIVE{
            individuals.sort()
            for i in range(len(individuals)){
                individuals[i].fitness=100.0/float(len(individuals)-i+2)
            }
        }
        return individuals
    }

    def sex(self, father, mother){
        if Utils.random()<self.sex_rate{
            amount_of_children=2
            children=[[] for _ in range(amount_of_children)]
            for i in range(len(father.dna)){
                gene_share=Utils.random()
                children[0].append(gene_share*father.dna[i]+(1-gene_share)*mother.dna[i])
                children[1].append((1-gene_share)*father.dna[i]+gene_share*mother.dna[i])
            }
            for i in range(len(children)){
                children[i]=mother.makeChild(children[i])
            }
        }else{
            children=[]
            children.append(father.copy())
            children.append(mother.copy())
        }
        return children
    }

    def mutate(self, individuals){
        for individual in individuals{
            self.mutateIndividual(individual,False)
        }
        return individuals
    }

    def mutateIndividual(self, individual, force=False){
        for i in range(len(individual.dna)){
            if force or Utils.random()<self.mutation_rate{
                individual.dna[i]*=self.randomize()
            }
        }
        individual.fixlimits()
    }

    def enrichSpace(self, space){
        return space
    }

    def randomize(self){
        r=Utils.random()
        if (r<=0.3){
            r=Utils.randomFloat(0,0.06)
        }elif (r<=0.8){
            r=Utils.randomFloat(0,0.11)
        }elif (r<=0.9){
            r=Utils.randomFloat(0.09,0.16)
        }elif (r<=0.97){
            r=Utils.randomFloat(0.15,0.23)
        }else{
            r=Utils.randomFloat(0.333,0.666)
        }
        if (Utils.random()>0.5){
            r=-(1+r);
        }else{
            r=(1+r);
        }
        return r
    }
}