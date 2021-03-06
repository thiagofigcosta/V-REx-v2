#!/bin/python
# -*- coding: utf-8 -*-

import math
from GeneticAlgorithm import GeneticAlgorithm
from Enums import GeneticRankType
from SearchSpace import SearchSpace
from Utils import Utils

class EnhancedGeneticAlgorithm(GeneticAlgorithm){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    DEFAULT_RESET_MT_DNA_PERCENTAGE=35.89759047777904 # 10 
    DEFAULT_WILL_OF_D_PERCENT=0.11134565126126  # 0.07 
    DEFAULT_RECYCLE_THRESHOLD_PERCENT=0.19034005265624 # 0.05

    def __init__(self, looking_highest_fitness, max_children, max_age, mutation_rate, sex_rate, recycle_rate, rank_type=GeneticRankType.INCREMENTAL, resetMtDnaPercentage=DEFAULT_RESET_MT_DNA_PERCENTAGE,willOfDPercentage=DEFAULT_WILL_OF_D_PERCENT,recycleRateThreshold=DEFAULT_RECYCLE_THRESHOLD_PERCENT){
        super().__init__(looking_highest_fitness)
        self.max_population=None
        self.index_max_age=None
        self.index_max_children=None
        self.max_age=max_age
        self.max_children=max_age
        self.mutation_rate=mutation_rate
        self.sex_rate=sex_rate
        self.recycle_rate=recycle_rate
        self.rank_type=rank_type
        self.current_population_size=0
        self.resetMtDnaPercentage=resetMtDnaPercentage
        self.willOfDPercentage=willOfDPercentage
        self.recycleRateThreshold=recycleRateThreshold
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
        all_selected_pairs=[]
        selected_beings_set=set()
        for i in range(int(len(individuals)/2)){
            potential_parents=[]
            backup_individual=None
            for c in range(2){
                roulette_number=Utils.randomFloat(0,fitness_sum)
                current_roulette=0
                for individual in individuals {
                    current_roulette+=individual.fitness+offset
                    if current_roulette>=roulette_number {
                        first_selected=len(potential_parents)<1
                        relatives=False
                        if not first_selected {
                            relatives=self.isRelative(potential_parents[0],individual)
                            self.stats['total_attempts']+=1
                            if relatives{
                                self.stats['relatives_attempts']+=1
                            }
                        }
                        if ( first_selected or not relatives){
                            potential_parents.append(individual)
                            break
                        }elif backup_individual is None{
                            backup_individual=individual
                        }
                    }
                }
            }
            if (len(potential_parents)!=2){
                potential_parents.append(backup_individual)
            }
            selected_beings_set.add(potential_parents[0])
            selected_beings_set.add(potential_parents[1])
            all_selected_pairs.append(potential_parents)
        }
        non_selected_beings_set=set(individuals)-selected_beings_set
        for useless_being in non_selected_beings_set{
            if Utils.LazyCore.freeMemManually(){
                del useless_being
            }
        }
        non_selected_beings_set.clear()
        next_gen=[]
        self.current_population_size=len(individuals)
        for potential_parents in all_selected_pairs{
            children=self.sex(potential_parents[0],potential_parents[1])
            next_gen+=children
            self.current_population_size+=len(children)-2
        }
        for useful_being in selected_beings_set{
            if Utils.LazyCore.freeMemManually(){
                del useful_being
            }
        }
        selected_beings_set.clear()
        individuals.clear()
        self.current_population_size=len(next_gen)
        return next_gen
    }

    def fit(self, individuals){
        signal=1
        if not self.looking_highest_fitness{
            signal=-1
        }
        fit_iteration=0
        recycled=False
        while (fit_iteration==0 or recycled){
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
            if fit_iteration<1 {
                recycled,individuals=self.recycleBadIndividuals(individuals)
            }else{
                recycled=False
                max_allowed_population=self.getMaxAllowedPopulation()
                to_cut_off=int(len(individuals)-max_allowed_population)
                if to_cut_off > 0{
                    if self.rank_type==GeneticRankType.RELATIVE{
                        for e in range(len(individuals)-1,len(individuals)-to_cut_off,-1){
                            individual=individuals[e]
                            if Utils.LazyCore.freeMemManually(){ 
                                del individual
                            }
                        }
                        individuals=individuals[:-to_cut_off]
                    }else{
                        for e in range(to_cut_off){
                            individual=individuals[e]
                            if Utils.LazyCore.freeMemManually(){
                                del individual
                            }
                        }
                        individuals=individuals[to_cut_off:]
                    }
                } 
            }
            fit_iteration+=1
        }
        self.current_population_size=len(individuals)
        return individuals
    }

    def sex(self, father, mother){
        family=[]
        if Utils.random()<self.sex_rate{
            amount_of_children=GeneticAlgorithm.geneShare(Utils.random(),father.dna[self.index_max_children],mother.dna[self.index_max_children])[0]
            amount_of_children=max(1,amount_of_children)
            amount_of_children=min(self.max_children,amount_of_children)
            amount_of_children=self.calcBirthRate(amount_of_children)[1]
            children=[[] for _ in range(amount_of_children)]
            for c in range(amount_of_children){
                for i in range(len(father.dna)){
                    heritage_mother=Utils.random()>0.5
                    children_genes=GeneticAlgorithm.geneShare(Utils.random(),father.dna[i],mother.dna[i])
                    if heritage_mother{
                        children[c].append(children_genes[1])
                    }else{
                        children[c].append(children_genes[0])
                    }
                }
            }
            for i in range(len(children)){
                children[i]=mother.makeChild(children[i])
            }
            family+=children
        }
        family.append(father.copy())
        family.append(mother.copy())
        return family
    }

    def mutate(self, individuals){
        for individual in individuals{
            self.mutateIndividual(individual,force=False)
        }
        individuals=self.age(individuals)
        return individuals
    }

    def enrichSpace(self, space){
        self.index_max_age=space.add(self.max_age/2,self.max_age*abs(self.randomize()),SearchSpace.Type.INT,name='Max age',unique=True)
        self.index_max_children=space.add(2,self.max_children*abs(self.randomize()),SearchSpace.Type.INT,name='Max children',unique=True)
        return space
    }

    def age(self, individuals){
        cemetery=[]
        for i,individual in enumerate(individuals){
            individual.age+=1
            if self.getLifeLeft(individual)<0 {
                if (individual.fitness<=(1-self.willOfDPercentage)*self.current_population_size and self.rank_type!=GeneticRankType.RELATIVE) or (individual.fitness/100>=self.willOfDPercentage and self.rank_type==GeneticRankType.RELATIVE){
                    if Utils.LazyCore.freeMemManually(){
                        del individual # dead
                    }
                    cemetery.append(i)
                }else{
                    individual.resetMtDna() # keeping alive extremely good ones
                }
            }
        }
        for corpse in sorted(cemetery, reverse=True){
            del individuals[corpse] # remove from list
        }
        self.current_population_size=len(individuals)
        return individuals
    }

    def mutateIndividual(self, individual, force=False){
        for i in range(len(individual.dna)){
            if (force or Utils.random()<self.mutation_rate) and i not in (self.index_max_age,self.index_max_children){
                individual.dna[i]=GeneticAlgorithm.geneMutation(self.randomize(),individual.dna[i])
            }
        }
        individual.fixlimits()
    }

    def isRelative(self, father, mother){
        return father.mt_dna==mother.mt_dna
    }

    def randomize(self){
        r=Utils.random()
        if (r<=0.42){
            r=Utils.randomFloat(0,0.0777)
        }elif (r<=0.6){
            r=Utils.randomFloat(0,0.12)
        }elif (r<=0.7){
            r=Utils.randomFloat(0.03,0.15)
        }elif (r<=0.8){
            r=Utils.randomFloat(0.03,0.18)
        }elif (r<=0.85){
            r=Utils.randomFloat(0.06,0.22)
        }elif (r<=0.9){
            r=Utils.randomFloat(0.09,0.24)
        }elif (r<=0.97){
            r=Utils.randomFloat(0.1,0.33)
        }else{
            r=Utils.randomFloat(0.1,0.666)
        }
        if (Utils.random()>0.5){
            r=-(1+r)
        }else{
            r=(1+r)
        }
        return r
    }

    def getLifeLeft(self,individual){
        return individual.dna[self.index_max_age]-individual.age
    }
    
    def recycleBadIndividuals(self, individuals){
        recycled=False
        if self.rank_type==GeneticRankType.RELATIVE{
            custom_range=range(len(individuals)-1,-1,-1)
        }else{
            custom_range=range(0,len(individuals),1)
        }
        for i in custom_range {
            individual=individuals[i]
            if (individual.fitness<self.recycleRateThreshold*self.current_population_size and self.rank_type!=GeneticRankType.RELATIVE) or (individual.fitness/100>self.recycleRateThreshold and self.rank_type==GeneticRankType.RELATIVE){
                if Utils.LazyCore.freeMemManually(){
                    del individual
                }
                idx_of_amazing_individual=int(self.willOfDPercentage*len(individuals)*Utils.random())
                if self.rank_type!=GeneticRankType.RELATIVE{
                    idx_of_amazing_individual=(len(individuals)-1)-idx_of_amazing_individual
                }
                amazing_individual=individuals[idx_of_amazing_individual]
                individuals[i]=amazing_individual.makeChild(amazing_individual.dna) # exploit
                individuals[i].age=-1 # aging will occur
                self.mutateIndividual(individuals[i],force=True) # explore
                individuals[i].evaluate()
                recycled=True
            }else{
                break
            }
        }
        return recycled, individuals
    }

    def calcBirthRate(self,amount_of_children=0){
        max_pop=self.getMaxAllowedPopulation()
        growth_rate=2.2
        birth_rate=growth_rate*((max_pop-self.current_population_size)/max_pop) # Logistic Population Growth: I = rN ( K - N / K)
        new_amount_of_children=math.ceil(amount_of_children*birth_rate)
        # print('birth_rate:',birth_rate,' amount_of_children:',amount_of_children,' new_amount_of_children:',new_amount_of_children)
        return birth_rate, new_amount_of_children
    }

    def getMaxAllowedPopulation(self){
        return self.max_population
    }

    def startGen(self,individuals=None,**kwargs){
        self.stats={}
        self.stats['relatives_attempts']=0
        self.stats['total_attempts']=0
        self.stats['mt_dnas_count']=0
        self.stats['individuals']=0
    }

    def finishGen(self,individuals=None,**kwargs){
        if individuals is not None{
            verbose=kwargs.get('verbose')
            mt_dnas={}
            for individual in individuals{
                mt_dna=individual.mt_dna
                if mt_dna not in mt_dnas{
                    mt_dnas[mt_dna]=0
                }
                mt_dnas[mt_dna]+=1
            }
            self.stats['mt_dnas_count']=len(mt_dnas)
            self.stats['individuals']=len(individuals)
            self.stats['non_relative_chance']=self.stats['mt_dnas_count']/float(self.stats['individuals'])*100
            if (self.stats['non_relative_chance']<self.resetMtDnaPercentage){
                if verbose{
                    Utils.LazyCore.info('\tReseting mtDNA...')
                }
                for individual in individuals{
                    individual.resetMtDna()
                }
            }
        }
    }

}