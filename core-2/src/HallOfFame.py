#!/bin/python
# -*- coding: utf-8 -*-

class HallOfFame(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    CACHE_WEIGHTS=True
    CACHE_FOLDER='neural_genome_cache'
    
    def __init__(self, max_notables, looking_highest_fitness){
        self.max_notables=max_notables
        self.looking_highest_fitness=looking_highest_fitness
        self.notables=[]
        self.best=0
    }

    def update(self,candidates,gen=-1){
        raise Exception('Not implemented yet!')
    }
}