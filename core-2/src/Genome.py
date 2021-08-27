#!/bin/python
# -*- coding: utf-8 -*-

from SearchSpace import SearchSpace
from Enums import Metric,NodeType,Loss,LabelEncoding,Optimizers
from Hyperparameters import Hyperparameters
from Utils import Utils

class Genome(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    COMPRESS_WEIGHTS=True
    ENCODE_B64_WEIGHTS=True
    ENCODE_B65_WEIGHTS=False
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
            }elif limit.data_type==SearchSpace.Type.BOOLEAN{
                self.dna.append(limit.max_value if Utils.random()>.5 else limit.min_value)
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
                   Utils.LazyCore.exception(e)
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
    
    def getWeights(self,raw=False){
        if (self.cached and Genome.CACHE_WEIGHTS){
            self._weights=Utils.loadObj(self.cache_file)
        }
        if raw {
            return self._weights
        }else{
            return Genome.decodeWeights(self._weights)
        }
    }

    def setWeights(self,weights,raw=False){
        if raw {
            self._weights=weights
        }else{
            self._weights=Genome.encodeWeights(weights)
        }
        self.forceCache()
    }

    @staticmethod
    def encodeWeights(weights){
        if weights is None{
            return weights
        }
        wei_str=Utils.objToJsonStr(weights,compress=Genome.COMPRESS_WEIGHTS,b64=Genome.ENCODE_B64_WEIGHTS)
        if Genome.ENCODE_B64_WEIGHTS and Genome.ENCODE_B65_WEIGHTS{
            wei_str=Utils.base64ToBase65(wei_str)
        }
        return wei_str
    }

    @staticmethod
    def decodeWeights(weights){
        if weights is None{
            return weights
        }
        wei_str=weights
        if Genome.ENCODE_B64_WEIGHTS and Genome.ENCODE_B65_WEIGHTS{
            wei_str=Utils.base65ToBase64(wei_str)
        }
        return Utils.jsonStrToObj(wei_str,compress=Genome.COMPRESS_WEIGHTS,b64=Genome.ENCODE_B64_WEIGHTS)
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
            that._weights=self._weights # string copy
            that.cached=self.cached
            that.cache_file=self.genCacheFilename()
            if self.is_neural and self.cached and Genome.CACHE_WEIGHTS {
                Utils.copyFile(self.cache_file,that.cache_file)
            }
        }
        return that
    }

    @staticmethod
    def enrichSearchSpace(search_space,multi_net_enhanced_nn=False){
        if not multi_net_enhanced_nn{
            # mandatory
            batch_size=search_space['batch_size']
            alpha=search_space['alpha']
            shuffle=search_space['shuffle']
            patience_epochs=search_space['patience_epochs']
            max_epochs=search_space['max_epochs']
            loss=search_space['loss']
            label_type=search_space['label_type']
            #optional
            # adam=search_space['adam'] # deprecated in favor of Optimizers
            optimizer=search_space['optimizer']
            monitor_metric=search_space['monitor_metric']
            model_checkpoint=search_space['model_checkpoint']
            # layer dependent
            layers=search_space['layers']
            layer_sizes=search_space['layer_sizes']
            node_types=search_space['node_types']
            dropouts=search_space['dropouts']
            bias=search_space['bias']

            if optimizer is None{
                optimizer=SearchSpace.Dimension(SearchSpace.Type.INT,Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),name='optimizer')
            }
            if monitor_metric is None{
                monitor_metric=SearchSpace.Dimension(SearchSpace.Type.INT,Metric.RAW_LOSS,Metric.RAW_LOSS,name='monitor_metric')
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

            enriched_search_space.add(optimizer.min_value,optimizer.max_value,optimizer.data_type,optimizer.name)
            enriched_search_space.add(monitor_metric.min_value,monitor_metric.max_value,monitor_metric.data_type,monitor_metric.name)
            enriched_search_space.add(model_checkpoint.min_value,model_checkpoint.max_value,model_checkpoint.data_type,model_checkpoint.name)
            
            enriched_search_space.add(layers.min_value,layers.max_value,layers.data_type,layers.name)
            for l in range(layers.max_value){
                if l+1<layers.max_value{
                    enriched_search_space.add(layer_sizes.min_value,layer_sizes.max_value,layer_sizes.data_type,layer_sizes.name[:-1]+'_{}'.format(l))
                    enriched_search_space.add(node_types.min_value,node_types.max_value,node_types.data_type,node_types.name[:-1]+'_{}'.format(l))
                }else{
                    enriched_search_space.add(0,0,layer_sizes.data_type,'out_layer-size')
                    enriched_search_space.add(0,0,node_types.data_type,'out_layer-type')
                }
                enriched_search_space.add(dropouts.min_value,dropouts.max_value,dropouts.data_type,dropouts.name[:-1]+'_{}'.format(l))
                enriched_search_space.add(bias.min_value,bias.max_value,bias.data_type,bias.name+'_{}'.format(l))
            }
            return enriched_search_space
        }else{
            networks=search_space['networks']
            if networks.min_value!=networks.max_value {
                raise Exception('The amount of networks must be static due to dataset division, networks.min_value must be equal to networks.max_value')
            }
            amount_of_networks=networks.max_value
            layers=[]
            layer_sizes=[]
            node_types=[]
            dropouts=[]

            alpha=[]
            shuffle=[]
            bias=[]
            loss=[]
            optimizer=[]

            # variables mandatory for every network
            for n in range(amount_of_networks){
                layers.append(search_space['layers'+'_{}'.format(n)])
                layer_sizes.append(search_space['layer_sizes'+'_{}'.format(n)])
                node_types.append(search_space['node_types'+'_{}'.format(n)])
                dropouts.append(search_space['dropouts'+'_{}'.format(n)])
            }
            # network variable that does not need to be specified for every single net
            for n in range(amount_of_networks){
                specific_name='alpha'+'_{}'.format(n)
                variable=search_space[specific_name]
                if variable is None{
                    variable=search_space['alpha']
                    variable.name=specific_name
                }
                alpha.append(variable)
                specific_name='shuffle'+'_{}'.format(n)
                variable=search_space[specific_name]
                if variable is None{
                    variable=search_space['shuffle']
                    variable.name=specific_name
                }
                shuffle.append(variable)
                specific_name='bias'+'_{}'.format(n)
                variable=search_space[specific_name]
                if variable is None{
                    variable=search_space['bias']
                    variable.name=specific_name
                }
                bias.append(variable)
                specific_name='loss'+'_{}'.format(n)
                variable=search_space[specific_name]
                if variable is None{
                    variable=search_space['loss']
                    variable.name=specific_name
                }
                loss.append(variable)
                specific_name='optimizer'+'_{}'.format(n)
                variable=search_space[specific_name]
                if variable is None{
                    variable=search_space['optimizer']
                    if variable is None{ # default
                        variable=SearchSpace.Dimension(SearchSpace.Type.INT,Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),name='optimizer')
                    }
                    variable.name=specific_name
                }
                optimizer.append(variable)
            }
            # static variables for every net
            # mandatory
            batch_size=search_space['batch_size']
            patience_epochs=search_space['patience_epochs']
            max_epochs=search_space['max_epochs']
            label_type=search_space['label_type']
            #optional
            monitor_metric=search_space['monitor_metric']
            model_checkpoint=search_space['model_checkpoint']
            
            if monitor_metric is None{
                monitor_metric=SearchSpace.Dimension(SearchSpace.Type.INT,Metric.RAW_LOSS,Metric.RAW_LOSS,name='monitor_metric')
            }
            if model_checkpoint is None{
                model_checkpoint=SearchSpace.Dimension(SearchSpace.Type.BOOLEAN,True,True,name='model_checkpoint')
            }

            enriched_search_space=SearchSpace()
            enriched_search_space.add(batch_size.min_value,batch_size.max_value,batch_size.data_type,batch_size.name)
            enriched_search_space.add(patience_epochs.min_value,patience_epochs.max_value,patience_epochs.data_type,patience_epochs.name)
            enriched_search_space.add(max_epochs.min_value,max_epochs.max_value,max_epochs.data_type,max_epochs.name)
            enriched_search_space.add(label_type.min_value,label_type.max_value,label_type.data_type,label_type.name)
            enriched_search_space.add(monitor_metric.min_value,monitor_metric.max_value,monitor_metric.data_type,monitor_metric.name)
            enriched_search_space.add(model_checkpoint.min_value,model_checkpoint.max_value,model_checkpoint.data_type,model_checkpoint.name)

            enriched_search_space.add(networks.min_value,networks.max_value,networks.data_type,networks.name)
            for n in range(amount_of_networks){
                enriched_search_space.add(alpha[n].min_value,alpha[n].max_value,alpha[n].data_type,alpha[n].name)
                enriched_search_space.add(shuffle[n].min_value,shuffle[n].max_value,shuffle[n].data_type,shuffle[n].name)
                enriched_search_space.add(loss[n].min_value,loss[n].max_value,loss[n].data_type,loss[n].name)
                enriched_search_space.add(optimizer[n].min_value,optimizer[n].max_value,optimizer[n].data_type,optimizer[n].name)
                enriched_search_space.add(layers[n].min_value,layers[n].max_value,layers[n].data_type,layers[n].name)
                enriched_search_space.add(layers[n].max_value,layers[n].max_value,layers[n].data_type,'max_layers'+'_{}'.format(n))
                for l in range(layers[n].max_value){
                    if l+1<layers[n].max_value or n+1<amount_of_networks{
                        enriched_search_space.add(layer_sizes[n].min_value,layer_sizes[n].max_value,layer_sizes[n].data_type,layer_sizes[n].name[:-2]+'_{}-{}'.format(n,l))
                        enriched_search_space.add(node_types[n].min_value,node_types[n].max_value,node_types[n].data_type,node_types[n].name[:-2]+'_{}-{}'.format(n,l))
                    }else{
                        enriched_search_space.add(0,0,layer_sizes[n].data_type,'out_layer-size')
                        enriched_search_space.add(0,0,node_types[n].data_type,'out_layer-type')
                    }
                    enriched_search_space.add(dropouts[n].min_value,dropouts[n].max_value,dropouts[n].data_type,dropouts[n].name[:-2]+'_{}-{}'.format(n,l))
                    enriched_search_space.add(bias[n].min_value,bias[n].max_value,bias[n].data_type,bias[n].name[:-2]+'_{}-{}'.format(n,l))
                }
            }
            return enriched_search_space
        }
    }

    def toHyperparameters(self,output_size,output_layer_type,multi_net_enhanced_nn=False){
        if not multi_net_enhanced_nn{
            batch_size=int(self.dna[0])
            alpha=float(self.dna[1])
            shuffle=bool(self.dna[2])
            patience_epochs=int(self.dna[3])
            max_epochs=int(self.dna[4])
            loss=Loss(self.dna[5])
            label_type=self.getHyperparametersEncoder(multi_net_enhanced_nn)

            optimizer=Optimizers(self.dna[7])
            monitor_metric=Metric(self.dna[8])
            model_checkpoint=bool(self.dna[9])

            layers=int(self.dna[10])
            first_layer_dependent=11
            layer_sizes=[]
            node_types=[]
            dropouts=[]
            bias=[]
            amount_of_dependent=4
            for l in range(layers){
                layer_sizes.append(int(self.dna[(first_layer_dependent+0)+amount_of_dependent*l]))
                node_types.append(NodeType(self.dna[(first_layer_dependent+1)+amount_of_dependent*l]))
                dropouts.append(float(self.dna[(first_layer_dependent+2)+amount_of_dependent*l]))
                bias.append(bool(self.dna[(first_layer_dependent+3)+amount_of_dependent*l]))
            }
            hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss, model_checkpoint=model_checkpoint, monitor_metric=monitor_metric)
            hyperparameters.setLastLayer(output_size,output_layer_type)
            return hyperparameters
        }else{
            batch_size=int(self.dna[0])
            patience_epochs=int(self.dna[1])
            max_epochs=int(self.dna[2])
            label_type=self.getHyperparametersEncoder(multi_net_enhanced_nn)
            monitor_metric=Metric(self.dna[4])
            model_checkpoint=bool(self.dna[5])

            networks=int(self.dna[6])
            last_index=7
            alpha=[]
            shuffle=[]
            loss=[]
            optimizer=[]
            layers=[]
            layer_sizes=[]
            node_types=[]
            dropouts=[]
            bias=[]
            network_parameters=10
            layer_parameters=4
            offset=0
            for n in range(networks){
                alpha.append(float(self.dna[(last_index+0)+network_parameters*n+offset]))
                shuffle.append(bool(self.dna[(last_index+1)+network_parameters*n+offset]))
                loss.append(Loss(self.dna[(last_index+2)+network_parameters*n+offset]))
                optimizer.append(Optimizers(self.dna[(last_index+3)+network_parameters*n+offset]))
                layers.append(int(self.dna[(last_index+4)+network_parameters*n+offset]))
                max_layers=int(self.dna[(last_index+5)+network_parameters*n+offset])
                layer_sizes.append([])
                node_types.append([])
                dropouts.append([])
                bias.append([])
                for l in range(layers[-1]){
                    layer_sizes[-1].append(int(self.dna[(last_index+6)+network_parameters*n+offset]))
                    node_types[-1].append(NodeType(self.dna[(last_index+7)+network_parameters*n+offset]))
                    dropouts[-1].append(float(self.dna[(last_index+8)+network_parameters*n+offset]))
                    bias[-1].append(bool(self.dna[(last_index+9)+network_parameters*n+offset]))
                    offset+=layer_parameters
                }
                offset+=(max_layers-layers[-1]-1)*layer_parameters
            }
            hyperparameters=Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss, model_checkpoint=model_checkpoint, monitor_metric=monitor_metric, amount_of_networks=networks)
            hyperparameters.setLastLayer(output_size,output_layer_type)
            return hyperparameters
        }
    }

    def getHyperparametersEncoder(self,multi_net_enhanced_nn){
        if multi_net_enhanced_nn{
            return LabelEncoding(self.dna[3])
        }else{
            return LabelEncoding(self.dna[6])
        }
    }
}