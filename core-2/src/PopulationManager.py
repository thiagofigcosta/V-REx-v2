#!/bin/python
# -*- coding: utf-8 -*-

import time
from Core import Core
from StandardGenetic import StandardGenetic
from EnhancedGenetic import EnhancedGenetic


class PopulationManager(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    PRINT_REL_FREQUENCY=10
    MT_DNA_VALIDITY=15
    
    def __init__(self,genetic_algorithm,search_space,eval_callback,population_start_size,looking_highest_fitness,neural_genome=False,print_deltas=False,after_gen_callback=None){
        self.genetic_algorithm=genetic_algorithm
        self.looking_highest_fitness=looking_highest_fitness
        self.genetic_algorithm.looking_highest_fitness=looking_highest_fitness
        if type(self.genetic_algorithm) is StandardGenetic {
            self.space=self.genetic_algorithm.enrichSpace(space)
        }elif type(self.genetic_algorithm) is EnhancedGenetic{
            self.space=self.genetic_algorithm.enrichSpace(space)
            self.genetic_algorithm.max_population=population_start_size*2
        }else{
            raise Exception('Unknown genetic algorithm of type {}'.format(type(self.genetic_algorithm)))
        }
        self.population=[]
        for i in range(population_start_size){
            self.population.append(Genome(search_space,eval_callback,neural_genome))
        }
        self.print_deltas=print_deltas
        self.hall_of_fame=None
        self.after_gen_callback=after_gen_callback
    }

    def __del__(self){
        self.genetic_algorithm=None
        self.looking_highest_fitness=None
        self.space=None
        for individual in population{
            del individual
        }
        self.population=None
        self.print_deltas=None
        self.hall_of_fame=None
        self.after_gen_callback=None
    }

    def naturalSelection(self, gens, verbose=False){
        for g in range(1,gen+1){
            t1=time.time()
            if self.looking_highest_fitness{
                best_out=float('-inf')
            }else{
                best_out=float('inf')
            }
            if verbose{
                Core.LOGGER.info('\tEvaluating individuals...')
            }
            for p,individual in enumerate(self.population){
                individual.evaluate()
                output=individual.output
                if self.looking_highest_fitness{
                    if output>best_out {
                        best_out=output
                    }
                }else{
                    if output<best_out{
                        best_out=output
                    }
                }
                if verbose{
                    percent=(p+1)/float(len(population))*100.0
                    if int(percent)%PopulationManager.PRINT_REL_FREQUENCY==0 {
                        Core.LOGGER.info('\t\tprogress: {:2.f}%'.format(percent))
                    }
                }
            }
            if verbose{
                Core.LOGGER.info('\tEvaluated individuals...OK')
                Core.LOGGER.info('\tCalculating fitness...')
            }
            self.genetic_algorithm.fit(self.population)
            if verbose{
                Core.LOGGER.info('\tCalculated fitness...OK')
            }
            if self.hall_of_fame is not None {
                if verbose{
                    Core.LOGGER.info('\tSetting hall of fame...')
                }
                self.hall_of_fame.update(self.population,g)
                if verbose{
                    Core.LOGGER.info('\tSetted hall of fame...OK')
                }
            }
            if g<gens{
                if verbose{
                    Core.LOGGER.info('\tSelecting and breeding individuals...')
                }
                self.genetic_algorithm.select(self.population)
                if verbose{
                    Core.LOGGER.info('\tSelected and breed individuals...OK')
                    enhanced_str=' and aging'
                    if type(self.genetic_algorithm) is not EnhancedGenetic{
                        enhanced_str=''
                    }
                    Core.LOGGER.info('\tMutating{} individuals...'.format(enhanced_str))
                }
                self.genetic_algorithm.select(self.population)
                if g%PopulationManager.MT_DNA_VALIDITY==0{
                    for individual in self.population{
                        individual.resetMtDna()
                    }
                }
                if verbose{
                    enhanced_str=' and aged'
                    if type(self.genetic_algorithm) is not EnhancedGenetic{
                        enhanced_str=''
                    }
                    Core.LOGGER.info('\tMutated{} individuals...OK'.format(enhanced_str))
                }
            }else{
                self.population.sort()
            }
            t2=time.time()
            delta=t2-t1
            if self.after_gen_callback is not None {
                self.after_gen_callback()
            }
        }
        if self.print_deltas {
            Core.LOGGER.info('Generation {} of {}, size: {} takes: {}'.format(g,gens,len(self.population),Utils.timestampByExtensive(delta)))
        }
    }

}
