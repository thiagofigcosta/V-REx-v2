#!/bin/python
# -*- coding: utf-8 -*-

import os
from Genome import Genome
from Utils import Utils
from StandardGeneticAlgorithm import StandardGeneticAlgorithm
from EnhancedGeneticAlgorithm import EnhancedGeneticAlgorithm
from functools import partial
import ray
import multiprocessing
import pathos
import pathos.pools as pp
import psutil

# from memory_profiler import profile

class PopulationManager(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    PRINT_REL_FREQUENCY=10
    SIMULTANEOUS_EVALUATIONS=1 # parallelism, 0 = infinite
    PROGRESS_WHEN_PARALLEL=True # only for ray and multiprocessing
    RAY_ON=False # uses ray instead of multiprocessing or pathos
    CPU_AFFINITY=False # assign each process for a single cpu when using multiprocessing
    USE_POOL=True # use pathos when ray is not on
    USE_PATHOS=True # use pathos pool instead of vanilla multiprocessing pool
    USE_PATHOS_POOL_THREADING=True # use thread instead of process
    USE_PATHOS_POOL_VANILLA_PROCESS=False
    
    def __init__(self,genetic_algorithm,search_space,eval_callback,population_start_size,neural_genome=False,print_deltas=False,after_gen_callback=None,force_sequential=False){
        self.genetic_algorithm=genetic_algorithm
        search_space=search_space.copy() # prevent changes
        self.space=self.genetic_algorithm.enrichSpace(search_space)
        has_age=False
        if type(self.genetic_algorithm) is EnhancedGeneticAlgorithm{
            self.genetic_algorithm.max_population=population_start_size*2
            has_age=True
        }
        self.population=[]
        for i in range(population_start_size){
            self.population.append(Genome(search_space,eval_callback,neural_genome,has_age=has_age))
        }
        self.print_deltas=print_deltas
        self.hall_of_fame=None
        self.after_gen_callback=after_gen_callback
        self.last_run_population_sizes=[]
        self.force_sequential=force_sequential
        if not self.force_sequential{
            if PopulationManager.RAY_ON{
                if PopulationManager.SIMULTANEOUS_EVALUATIONS!=1 and not ray.is_initialized(){
                    if PopulationManager.SIMULTANEOUS_EVALUATIONS == 0 {
                        ray.init()
                    }else{
                        ray.init(num_cpus = PopulationManager.SIMULTANEOUS_EVALUATIONS)
                    }
                }
            }else{
                if PopulationManager.SIMULTANEOUS_EVALUATIONS==0 {
                    PopulationManager.SIMULTANEOUS_EVALUATIONS=os.cpu_count() # same as multiprocessing.cpu_count()
                }
                if not PopulationManager.USE_POOL {
                    self.multiprocessing_manager=multiprocessing.Manager()
                }
            }
        }
    }

    def __del__(self){
        self.genetic_algorithm=None
        self.space=None
        for individual in self.population{
            if Utils.LazyCore.freeMemManually(){
                del individual
            }
        }
        self.population=None
        self.print_deltas=None
        self.hall_of_fame=None
        self.after_gen_callback=None
        if not self.force_sequential{
            if PopulationManager.RAY_ON{
                if ray.is_initialized() {
                    ray.shutdown()
                }
            }
        }
    }

    @staticmethod
    @ray.remote
    def _evaluateIndividualRay(individual,g){
        return PopulationManager._evaluateIndividual(individual,g)
    }

    @staticmethod
    def _evaluateIndividual(individual,g,ret_val=None){
        individual.evaluate()
        individual.gen=g
        if ret_val is None{
            return individual.output
        }else{
            ret_val.value=individual.output
            return
        }
    }

    @staticmethod
    def _evaluateIndividualMulti(individuals,g,out,cpu_id=None){
        if cpu_id is not None and PopulationManager.CPU_AFFINITY{
            try{
                proc=psutil.Process()
                # print('Affinity before:',proc.cpu_affinity())
                proc.cpu_affinity([cpu_id])
                # print('Affinity after:',proc.cpu_affinity())
            }except{
                pass
            }
        }
        for individual in individuals{
            out.append(PopulationManager._evaluateIndividual(individual,g))
        }
    }

    # @profile
    def naturalSelection(self, gens, verbose=False, verbose_generations=None){
        mean_delta=0.0
        self.last_run_population_sizes=[]
        self.last_run_stats=[]
        if PopulationManager.SIMULTANEOUS_EVALUATIONS!=1 and not self.force_sequential {
            Utils.LazyCore.info('Using multiprocessing({})!'.format(PopulationManager.SIMULTANEOUS_EVALUATIONS))
        }
        for g in range(1,gens+1){
            t1=Utils.now()
            self.genetic_algorithm.startGen()
            if self.genetic_algorithm.looking_highest_fitness{
                best_out=float('-inf')
            }else{
                best_out=float('inf')
            }
            if verbose{
                Utils.LazyCore.info('\tEvaluating individuals...')
            }
            outputs=[]
            if PopulationManager.SIMULTANEOUS_EVALUATIONS!=1 and not self.force_sequential {
                if PopulationManager.RAY_ON{
                    if verbose and PopulationManager.PROGRESS_WHEN_PARALLEL{
                        current_run=0
                        last_print=0
                        parallel_tasks=[PopulationManager._evaluateIndividualRay.remote(individual,g) for individual in self.population]
                        while len(parallel_tasks){
                            max_returns=max(int(len(self.population)/PopulationManager.PRINT_REL_FREQUENCY),1)
                            num_returns=max_returns if len(parallel_tasks) >= max_returns else len(parallel_tasks)
                            ready,not_ready=ray.wait(parallel_tasks,num_returns=num_returns)
                            if verbose {
                                current_run+=len(ready)
                                percent=current_run/float(len(self.population))*100.0
                                if percent>=(last_print+1)*PopulationManager.PRINT_REL_FREQUENCY {
                                    last_print=int(int(percent)/PopulationManager.PRINT_REL_FREQUENCY)
                                    Utils.LazyCore.info('\t\tprogress: {:2.2f}%'.format(percent))
                                }
                            }
                            if len(ready)>0{
                                outputs+=ray.get(ready)
                            }
                            parallel_tasks=not_ready
                        }
                    }else{
                        if verbose{
                            Utils.LazyCore.info('\t\tProgress track is disabled due to parallelism!')
                        }
                        outputs=ray.get([PopulationManager._evaluateIndividualRay.remote(individual,g) for individual in self.population])
                    }
                }else{
                    if PopulationManager.USE_POOL {
                        if verbose{
                            Utils.LazyCore.info('\t\tProgress track is disabled due to parallelism!')
                        }
                        maxtasksperchild=None # default is None, otherwise is the amount of iterations per process before creating a new one
                        if PopulationManager.USE_PATHOS { # pathos uses dill as serializer which is way better than pickle
                            if PopulationManager.USE_PATHOS_POOL_THREADING{ 
                                with pp.ThreadPool(PopulationManager.SIMULTANEOUS_EVALUATIONS,maxtasksperchild=maxtasksperchild) as pool{
                                    outputs=pool.map(PopulationManager._evaluateIndividual, self.population, [g]*len(self.population))
                                }
                            }else{
                                if PopulationManager.USE_PATHOS_POOL_VANILLA_PROCESS{
                                    self.multiprocessing_pool=pp._ProcessPool(PopulationManager.SIMULTANEOUS_EVALUATIONS,maxtasksperchild=maxtasksperchild) # same as multiprocessing but using dill
                                    _evaluateIndividualPartial=partial(PopulationManager._evaluateIndividual, g=g)
                                    outputs=self.multiprocessing_pool.map(_evaluateIndividualPartial, self.population)
                                    self.multiprocessing_pool.close()
                                    self.multiprocessing_pool.join() 
                                    self.multiprocessing_pool.terminate() 
                                }else{
                                    # self.multiprocessing_pool=pp.ProcessPool(PopulationManager.SIMULTANEOUS_EVALUATIONS,maxtasksperchild=maxtasksperchild) 
                                    # outputs=self.multiprocessing_pool.map(PopulationManager._evaluateIndividual, self.population, [g]*len(self.population))
                                    # self.multiprocessing_pool.close()
                                    # self.multiprocessing_pool.join() 
                                    # pathos.helpers.shutdown()
                                    # self.multiprocessing_pool.clear()
                                    with pp.ProcessPool(PopulationManager.SIMULTANEOUS_EVALUATIONS,maxtasksperchild=maxtasksperchild) as pool{
                                        outputs=pool.map(PopulationManager._evaluateIndividual, self.population, [g]*len(self.population))
                                    }
                                }
                            }
                       }else{
                            self.multiprocessing_pool=multiprocessing.Pool(processes=PopulationManager.SIMULTANEOUS_EVALUATIONS,maxtasksperchild=maxtasksperchild) # raises can't pickle exception
                            _evaluateIndividualPartial=partial(PopulationManager._evaluateIndividual, g=g)
                            outputs=self.multiprocessing_pool.map(_evaluateIndividualPartial, self.population)
                            self.multiprocessing_pool.close()
                            self.multiprocessing_pool.join() 
                        }
                    }else{
                        t_id=0
                        parallel_tasks=[]
                        ret_vals=[]
                        new_processes=PopulationManager.SIMULTANEOUS_EVALUATIONS-1
                        parallel_args=[[] for _ in range(new_processes)]
                        for individual in self.population{
                            parallel_args[t_id].append(individual)
                            if len(parallel_args[t_id])>=int(len(self.population)/new_processes) and t_id+1<new_processes{
                                t_id+=1
                            }
                        }
                        for t_id in range(new_processes){
                            ret_val=self.multiprocessing_manager.list()
                            p=multiprocessing.Process(target=PopulationManager._evaluateIndividualMulti,args=(parallel_args[t_id],g,ret_val,t_id,))
                            parallel_tasks.append(p)
                            ret_vals.append(ret_val)
                            p.start()
                        }
                        t_id=0
                        t_complete=0
                        for task in parallel_tasks{
                            task.join()
                            task.terminate() # do I need this?
                            if verbose{
                                t_complete+=len(parallel_args[t_id])
                                percent=t_complete/float(len(self.population))*100.0
                                Utils.LazyCore.info('\t\tprogress: {:2.2f}% *log time is not accurate'.format(percent))
                                t_id+=1
                            }
                        }
                        for ret_val in ret_vals{
                            outputs+=ret_val
                        }
                    }
                }
                for pop_id,output in enumerate(outputs){
                    self.population[pop_id].output=output
                    self.population[pop_id].gen=g
                }
            }else{
                last_print=1
                for p,individual in enumerate(self.population){
                    individual.evaluate()
                    individual.gen=g
                    outputs.append(individual.output)
                    if verbose{
                        percent=(p+1)/float(len(self.population))*100.0
                        if  percent>=last_print*PopulationManager.PRINT_REL_FREQUENCY {
                            last_print+=1
                            Utils.LazyCore.info('\t\tprogress: {:2.2f}%'.format(percent))
                        }
                    }
                }
            }
            if self.genetic_algorithm.looking_highest_fitness{
                best_out=max(outputs)
            }else{
                best_out=min(outputs)
            }
            if verbose{
                Utils.LazyCore.info('\tEvaluated individuals...OK')
                Utils.LazyCore.info('\tCalculating fitness...')
            }
            self.population=self.genetic_algorithm.fit(self.population)
            if verbose{
                Utils.LazyCore.info('\tCalculated fitness...OK')
            }
            if self.hall_of_fame is not None {
                if verbose{
                    Utils.LazyCore.info('\tSetting hall of fame...')
                }
                self.hall_of_fame.update(self.population,g)
                if verbose{
                    Utils.LazyCore.info('\tSetted hall of fame...OK')
                }
            }
            if g<gens{
                if verbose{
                    Utils.LazyCore.info('\tSelecting and breeding individuals...')
                }
                self.population=self.genetic_algorithm.select(self.population)
                if verbose{
                    Utils.LazyCore.info('\tSelected and breed individuals...OK')
                    enhanced_str=' and aging'
                    if type(self.genetic_algorithm) is not EnhancedGeneticAlgorithm{
                        enhanced_str=''
                    }
                    Utils.LazyCore.info('\tMutating{} individuals...'.format(enhanced_str))
                }
                self.population=self.genetic_algorithm.mutate(self.population)
                if verbose{
                    enhanced_str=' and aged'
                    if type(self.genetic_algorithm) is not EnhancedGeneticAlgorithm{
                        enhanced_str=''
                    }
                    Utils.LazyCore.info('\tMutated{} individuals...OK'.format(enhanced_str))
                }
            }else{
                self.population.sort()
            }
            self.genetic_algorithm.finishGen(self.population,verbose=verbose)
            t2=Utils.now()
            delta=t2-t1
            mean_delta+=delta
            self.last_run_population_sizes.append(len(self.population))
            self.last_run_stats.append(self.genetic_algorithm.stats)
            if self.after_gen_callback is not None {
                args_list=[len(self.population),g,best_out,delta,self.population,self.hall_of_fame]
                self.after_gen_callback(args_list)
            }
            if len(self.population)<2{
                Utils.LazyCore.warn('Early stopping generation {} due to its small size {}, this population died!'.format(g,len(self.population)))
                break
            }
            if verbose_generations or self.print_deltas {
                Utils.LazyCore.info('Generation {} of {}, size: {} takes: {}'.format(g,gens,len(self.population),Utils.timestampByExtensive(delta)))
            }
        }
        return mean_delta/gens
    }

}
