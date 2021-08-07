#!/bin/python
# -*- coding: utf-8 -*-

from SearchSpace import SearchSpace
from Core import Core
from Utils import Utils

class Genome(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    CACHE_WEIGHTS=True
    CACHE_FOLDER='neural_genome_cache'
    
    def __init__(self, search_space, eval_callback, is_neural=False, has_age=False){
        self.limits=search_space
        self.dna=[]
        for limit in search_space{
            if limit.data_type==SearchSpace.Type.INT {
                self.dna.append(Utils.randomInt(limit.min_value,limit.max_value))
            }elif limit.data_type==SearchSpace.Type.FLOAT{
                self.dna.append(Utils.randomFloat(limit.min_value,limit.max_value))
            }else{
                raise Exception('Unkown search space data type {}'.format(limit.data_type))
            }
        }
        self.eval_callback=eval_callback
        self.is_neural=is_neural
        self.mt_dna=''
        self.fitness=0
        self.output=0
        if has_age {
            self.age=0
        }else{
            self.age=None
        }
        self.id=Utils.randomUUID()
        self.resetMtDna()
        if self.is_neural {
            self._weights=None
            self.cached=False
            self.cache_file=self.genCacheFilename()
        }
    }

    def __del__(self){
        if self.is_neural and self.cached and Genome.CACHE_WEIGHTS {
            Utils.deleteFile(self.cache_file)
        }
        self._weights=None
    }

    def __lt__(self, other){
        return self.fitness < other.fitness
    }

    def __str__(self){
        return self.toString()
    }

    def makeChild(self, dna){
        mother=self
        child=mother.copy()
        child.id=Utils.randomUUID()
        child.dna=dna+[] # deep copy
        child.fitness=0
        child.output=0
        if child.age is not None{
            child.age=0
        }
        return child
    }

    def evaluate(self){
        self.output=self.eval_callback(self) 
    }

    def fixlimits(self){
        for i in range(len(self.dna)){
            self.dna[i]=self.limits[i].fixValue(self.dna[i])
            if self.limits[i].data_type==SearchSpace.Type.INT {
                self.dna[i]=int(self.dna[i])
            }elif self.limits[i].data_type==SearchSpace.Type.FLOAT{
                self.dna[i]=float(self.dna[i])
            }else{
                raise Exception('Unkown search space data type {}'.format(self.limits[i].data_type))
            }
        }
    }

    def toString(self){
        out='Output: {} Fitness: {}'.format(self.output,self.fitness)
        if self.age is not None{
            out+=' age: {}'.format(self.age)
        }
        out+=' DNA: ['
        for i in range(len(self.dna)){
            out+=' '
            if self.limits[i].name is not None{
                out+=self.limits[i].name+': '
            }
            out+=str(self.dna[i])
            if i+1<len(self.dna){
                out+=','
            }else{
                out+=' '
            }
        }
        out+=']'
        return out
    }


    def resetMtDna(self){
        self.mt_dna=Utils.randomUUID()
    }

    def hasWeights(self){
        return (self._weights is not None and len(self._weights)>0) or (self.cached and Genome.CACHE_WEIGHTS)
    }

    def forceCache(self){
        if (self.cached and Genome.CACHE_WEIGHTS){
           Utils.deleteFile(self.cache_file)
           self.cached=False 
        }
        if (not self.cached and Genome.CACHE_WEIGHTS){
           success=False
           tries=0
           max_tries=5
           while (not success and tries<max_tries){
               tries+=1
               try{
                   Utils.saveObj(self._weights,self.cache_file)
                   success=True
               }except Exception as e{
                   Core.LOGGER.exception(e)
                   if (tries==max_tries-1){
                        self.cache_file=self.genCacheFilename()
                    }
               }
           } 
           if success{
               self.cached=True
               self._weights=None
           }
        }
    }
    
    def getWeights(self){
        if (self.cached and Genome.CACHE_WEIGHTS){
            self._weights=Utils.loadObj(self.cache_file)
        }
        return self._weights
    }

    def setWeights(self,weights){
        self._weights=weights
        self.forceCache()
    }

    def clearMemoryWeightsIfCached(self){
        if (self.cached and Genome.CACHE_WEIGHTS){
           self._weights=None
        }
    }

    def clearWeights(self){
        self._weights=None
        if self.is_neural and self.cached and Genome.CACHE_WEIGHTS {
            Utils.deleteFile(self.cache_file)
            self.cached=False
        }
    }

    def genCacheFilename(self){
        filename='{}-{}.weights_cache'.format(self.id,Utils.randomUUID())
        Utils.createFolderIfNotExists(Genome.CACHE_FOLDER)
        return Utils.joinPath(Genome.CACHE_FOLDER,filename)
    }

    def copy(self){
        that=Genome([], self.eval_callback, is_neural=self.is_neural)
        that.limits=self.limits.copy()
        that.dna=self.dna+[] # deep copy
        that.mt_dna=self.mt_dna
        that.fitness=self.fitness
        that.output=self.output
        that.age=self.age
        that.id=self.id
        if self.is_neural {
            that._weights=self._weights.copy() # TODO code me
            that.cached=self.cached
            that.cache_file=self.genCacheFilename()
            if self.is_neural and self.cached and Genome.CACHE_WEIGHTS {
                Utils.copyFile(self.cache_file,that.cache_file)
            }
        }
        return that
    }
}