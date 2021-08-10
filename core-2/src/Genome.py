#!/bin/python
# -*- coding: utf-8 -*-

from SearchSpace import SearchSpace
from Core import Core
from Enums import Metric,NodeType,Loss,LabelEncoding
from Hyperparameters import Hyperparameters
from Utils import Utils

class Genome(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    CACHE_WEIGHTS=True
    CACHE_FOLDER='neural_genome_cache'
    
    def __init__(self, search_space, eval_callback, is_neural=False, has_age=False){
        self.limits=search_space
        self.dna=[]
        for limit in search_space{
            if limit.data_type in (SearchSpace.Type.INT,SearchSpace.Type.BOOLEAN) {
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
        self.gen=-1
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
        return self.fitness < other.fitness or (self.fitness == other.fitness and self.age is not None and other.age is not None and self.age < other.age)
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
            }elif self.limits[i].data_type==SearchSpace.Type.BOOLEAN{
                self.dna[i]=bool(self.dna[i])
            }else{
                raise Exception('Unkown search space data type {}'.format(self.limits[i].data_type))
            }
        }
    }

    def toString(self){
        out='Output: {} Fitness: {}'.format(self.output,self.fitness)
        if self.gen > -1{
            out+=' gen: {}'.format(self.gen)
        }
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
        that.gen=self.gen
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

    @staticmethod
    def enrichSearchSpace(search_space,enh_neural_network=False){
        if not enh_neural_network{
            # mandatory
            batch_size=search_space['batch_size']
            alpha=search_space['alpha']
            shuffle=search_space['shuffle']
            patience_epochs=search_space['patience_epochs']
            max_epochs=search_space['max_epochs']
            loss=search_space['loss']
            label_type=search_space['label_type']
            #optional
            adam=search_space['adam']
            monitor_metric=search_space['monitor_metric']
            model_checkpoint=search_space['model_checkpoint']
            # layer dependent
            layers=search_space['layers']
            layer_sizes=search_space['layer_sizes']
            node_types=search_space['node_types']
            dropouts=search_space['dropouts']
            bias=search_space['bias']

            if adam is None{
                adam=SearchSpace.Dimension(SearchSpace.Type.BOOLEAN,True,True,name='adam')
            }
            if monitor_metric is None{
                min_metric=Utils.getEnumBorder(Metric,False)
                monitor_metric=SearchSpace.Dimension(SearchSpace.Type.INT,min_metric,min_metric,name='monitor_metric')
            }
            if model_checkpoint is None{
                model_checkpoint=SearchSpace.Dimension(SearchSpace.Type.BOOLEAN,True,True,name='model_checkpoint')
            }

            enriched_search_space=SearchSpace()
            enriched_search_space.add(batch_size.min_value,batch_size.max_value,batch_size.data_type,batch_size.name)
            enriched_search_space.add(alpha.min_value,alpha.max_value,alpha.data_type,alpha.name)
            enriched_search_space.add(shuffle.min_value,shuffle.max_value,shuffle.data_type,shuffle.name)
            enriched_search_space.add(patience_epochs.min_value,patience_epochs.max_value,patience_epochs.data_type,patience_epochs.name)
            enriched_search_space.add(max_epochs.min_value,max_epochs.max_value,max_epochs.data_type,max_epochs.name)
            enriched_search_space.add(loss.min_value,loss.max_value,loss.data_type,loss.name)
            enriched_search_space.add(label_type.min_value,label_type.max_value,label_type.data_type,label_type.name)

            enriched_search_space.add(adam.min_value,adam.max_value,adam.data_type,adam.name)
            enriched_search_space.add(monitor_metric.min_value,monitor_metric.max_value,monitor_metric.data_type,monitor_metric.name)
            enriched_search_space.add(model_checkpoint.min_value,model_checkpoint.max_value,model_checkpoint.data_type,model_checkpoint.name)
            
            enriched_search_space.add(layers.min_value,layers.max_value,layers.data_type,layers.name)
            for l in range(layers.max_value){
                if l+1<layers.max_value{
                    enriched_search_space.add(layer_sizes.min_value,layer_sizes.max_value,layer_sizes.data_type,layer_sizes.name+'_{}'.format(l))
                    enriched_search_space.add(node_types.min_value,node_types.max_value,node_types.data_type,node_types.name+'_{}'.format(l))
                }else{
                    enriched_search_space.add(0,0,layer_sizes.data_type,'out_layer-size')
                    enriched_search_space.add(0,0,node_types.data_type,'out_layer-type')
                }
                enriched_search_space.add(dropouts.min_value,dropouts.max_value,dropouts.data_type,dropouts.name+'_{}'.format(l))
                enriched_search_space.add(bias.min_value,bias.max_value,bias.data_type,bias.name+'_{}'.format(l))
            }
            return enriched_search_space
        }
    }

    def toHyperparameters(self,output_size,output_layer_type,enh_neural_network=False){
        if not enh_neural_network{
            batch_size=self.dna[0]
            alpha=self.dna[1]
            shuffle=self.dna[2]
            patience_epochs=self.dna[3]
            max_epochs=self.dna[4]
            loss=Loss(self.dna[5])
            label_type=self.getHyperparametersEncoder()

            adam=self.dna[7]
            monitor_metric=Metric(self.dna[8])
            model_checkpoint=self.dna[9]

            layers=self.dna[10]
            first_layer_dependent=11
            layer_sizes=[]
            node_types=[]
            dropouts=[]
            bias=[]
            amount_of_dependent=4
            for l in range(layers){
                layer_sizes.append(self.dna[(first_layer_dependent+0)+amount_of_dependent*l])
                node_types.append(NodeType(self.dna[(first_layer_dependent+1)+amount_of_dependent*l]))
                dropouts.append(self.dna[(first_layer_dependent+2)+amount_of_dependent*l])
                bias.append(self.dna[(first_layer_dependent+3)+amount_of_dependent*l])
            }
            layer_sizes[-1]=output_size
            node_types[-1]=output_layer_type
            hyperparameters=Hyperparameters(batch_size, alpha, shuffle, adam, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss, model_checkpoint=model_checkpoint, monitor_metric=monitor_metric)
            return hyperparameters
        }
    }

    def getHyperparametersEncoder(self){
        return LabelEncoding(self.dna[6])
    }
}