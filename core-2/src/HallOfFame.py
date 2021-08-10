#!/bin/python
# -*- coding: utf-8 -*-

from Core import Core

class HallOfFame(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
   
    def __init__(self, max_notables, looking_highest_fitness){
        self.max_notables=max_notables
        self.looking_highest_fitness=looking_highest_fitness
        self.notables=[]
        self.best={'output':float('-inf') if looking_highest_fitness else float('inf'),'generation':-1,'genome': None}
    }

    def __del__(self){
        for notable in self.notables {
            if Core.FREE_MEMORY_MANUALLY==True{
                del notable
            }
        }
    }

    def getBestGenome(self){
        return self.notables[0]
    }

    def update(self,candidates,gen=-1){
        notables_to_select=candidates+self.notables
        notables_to_select.sort(key=lambda k: k.output, reverse=self.looking_highest_fitness)
        while len(notables_to_select)>self.max_notables {
            ordinary=notables_to_select[-1]
            if ordinary in self.notables {
                if Core.FREE_MEMORY_MANUALLY==True{
                    del ordinary
                }
            }
            notables_to_select.pop()
        }
        notables=[]
        for notable in notables_to_select{
            notables.append(notable.copy())
        }
        self.notables=notables
        if ((self.looking_highest_fitness and self.notables[0].output>self.best['output']) or (not self.looking_highest_fitness and self.notables[0].output<self.best['output'])){
            self.best['output']=self.notables[0].output
            self.best['generation']=gen
            self.best['genome']=str(self.notables[0])
        }
    }
}