#!/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import pandas as pd
import numpy as np
from tensorflow import keras # import keras
from tensorflow.keras.models import Model, load_model # from keras.models import Model, load_model
import tensorflow.keras.backend as K # import keras.backend as K
from tensorflow.keras.utils import plot_model # from keras.utils.vis_utils import plot_model
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Utils import Utils
from Dataset import Dataset
from Enums import NodeType, Metric
from abc import ABC, abstractmethod
from pathos.helpers import mp as pmp
import multiprocessing as mp
import multiprocessing.sharedctypes

class NeuralNetwork(ABC){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

	MODELS_PATH='models'
	SAVED_PLOTS_PATH='saved_plots'
	MULTIPROCESSING=False
	MANUAL_METRICS_NAMES=['accuracy','precision','recall','f1_score']
	NO_PATIENCE_LEFT_STR='Stop Epochs - No patience left'
	USE_MANUAL_METRICS=False # manual metrics are slower
	MANUAL_METRICS_ONLY_ON_VALIDATION=True # makes manual metrics faster when enabled
	SIGMOID_CLASSES_THRESHOLD=.5
	CLIP_NORM_INSTEAD_OF_VALUE=True
	USE_LEAKY_RELU=True

    def __init__(self,hyperparameters,name='',verbose=False){
        self.hyperparameters=hyperparameters
		self.name=name
		self.verbose=verbose
		self.data=None
		self.model=None
		self.callbacks=None
		self.basename='model'
		self.checkpoint_filename=None
		self.history={}
		self.metrics={}
		Utils.createFolder(NeuralNetwork.MODELS_PATH)
		Utils.createFolder(NeuralNetwork.SAVED_PLOTS_PATH)
    }

	def __del__(self){
		self.clearCache()
	}

	def clearCache(self){
		keras.backend.clear_session()
	}

	def _loadModelPartial(self,path){
		custom_objects={}
		if not NeuralNetwork.USE_MANUAL_METRICS{
			custom_metrics=self._metricsFactory()
			for i in range(len(custom_metrics)-1){
				custom_objects[custom_metrics[i].__name__]=custom_metrics[i]
			}
		}
		return custom_objects
	}

	@staticmethod
    def createSharedNumpyArray(array){
        # 'c': ctypes.c_char,  'u': ctypes.c_wchar,
        # 'b': ctypes.c_byte,  'B': ctypes.c_ubyte,
        # 'h': ctypes.c_short, 'H': ctypes.c_ushort,
        # 'i': ctypes.c_int,   'I': ctypes.c_uint,
        # 'l': ctypes.c_long,  'L': ctypes.c_ulong,
        # 'f': ctypes.c_float, 'd': ctypes.c_double
        if type(array) is not np.ndarray{
			array=NeuralNetwork.FormatData(array)
		}
		# RawArray is not thread safe, to write on it we need to do as following:
		# arr_lock = pmp.Lock()
		# with arr_lock {
		#     np_shared_array[0]=2
		# }
		if type(array) is np.ndarray{
        	shape=array.shape
			dimensions=len(shape)
			first_el=array
			total_size=1
			for i in range(dimensions){
				total_size*=shape[i]
				first_el=first_el[0]
			}
			dtype='f'
			if type(first_el) in (int,np.int,np.int32,np.int64){
				dtype='i'
			}
			# shared_array=pmp.RawArray(dtype, total_size) # pathos
			shared_array=mp.sharedctypes.RawArray(dtype, total_size)
			np_shared_array=np.frombuffer(shared_array,dtype=dtype,count=total_size).reshape(shape)
			np.copyto(np_shared_array, array)
			return np_shared_array
		}else{ # multi net, must return a list of shared arrays because each one can have different dimensions
			shapes=[array_el.shape for array_el in array]
			dimensions=1+len(shapes[0])
			first_el=array
			for i in range(dimensions){
				first_el=first_el[0]
			}
			dtype='f'
			if type(first_el) in (int,np.int,np.int32,np.int64){
				dtype='i'
			}
			np_shared_arrays=[]
			for s,shape in enumerate(shapes){
				total_size=1
				for l_shape in shape{
					total_size*=l_shape
				}
				# shared_array=pmp.RawArray(dtype, total_size) # pathos
				shared_array=mp.sharedctypes.RawArray(dtype, total_size)
				np_shared_array=np.frombuffer(shared_array,dtype=dtype,count=total_size).reshape(shape)
				np.copyto(np_shared_array, array[s])
				np_shared_arrays.append(np_shared_array)
			}
			return np_shared_arrays
		}
    }

	@staticmethod
	def FormatData(data){
		dims=0
		first_el=data
		while type(first_el) in (list,np.ndarray){
			dims+=1
			first_el=first_el[0]
		}

		if type(data) is np.ndarray or (dims>2 and type(data[0]) is np.ndarray){
			return data
		}
		if dims==2{ # labels or single network layer
			return np.array(data)
		}
		f=[]
		for feature in data{
			f.append(np.array(feature))
		}
		return f
	}

	@staticmethod
	def FormatFeatures(features,amount_of_networks){
		if type(features) is np.ndarray or (amount_of_networks>1 and type(features[0]) is np.ndarray){
			return features
		}
		if amount_of_networks==1{
			return np.array(features)
		}
		f=[]
		for feature in features{
			f.append(np.array(feature))
		}
		return f
	}

	@staticmethod
	def FormatLabels(labels){
		if type(labels) is np.ndarray{
			return labels
		}
		return np.array(labels)
	}

	def formatFeatures(self,features){
		return NeuralNetwork.FormatFeatures(features,self.hyperparameters.amount_of_networks)
	}

	def formatLabels(self,labels){
		return NeuralNetwork.FormatLabels(labels)
	}

	def formatData(self,data){
		return NeuralNetwork.FormatData(data)
	}

	def loadModel(self,path,compileModel=True){
		objs=self._loadModelPartial(path)
		loaded_model=load_model(path,custom_objects=objs,compile=compileModel)
		return loaded_model
	}

	def restoreCheckpointWeights(self,delete_after=True){
		if self.checkpoint_filename is not None and Utils.checkIfPathExists(self.getModelPath(self.checkpoint_filename)){
			loaded_model=self.loadModel(self.getModelPath(self.checkpoint_filename),compileModel=False)	
			if self.verbose {
				Utils.LazyCore.info('Restoring model checkpoint...')
			}
			self.model.set_weights(loaded_model.get_weights())
			if delete_after{
				Utils.deleteFile(self.getModelPath(self.checkpoint_filename))
			}
		}
	}

	def getModelPath(self,filename){
		path=filename
		if Utils.appendToStrIfDoesNotEndsWith(NeuralNetwork.MODELS_PATH,Utils.FILE_SEPARATOR) not in path{
			path=Utils.joinPath(NeuralNetwork.MODELS_PATH,filename)
		}
		return path
	}

	def getPlotsPath(self,filename){
		path=filename
		if Utils.appendToStrIfDoesNotEndsWith(NeuralNetwork.SAVED_PLOTS_PATH,Utils.FILE_SEPARATOR) not in path{
			path=Utils.joinPath(NeuralNetwork.SAVED_PLOTS_PATH,filename)
		}
		return path
	}

	def saveModelSchemaToFile(self,folder=''){
		base=self.getPlotsPath(folder)
		Utils.createFolderIfNotExists(base)
		filename='{}_{}.png'.format(self.basename,self.name)
		filepath=Utils.joinPath(base,filename)
		if self.verbose{
			Utils.LazyCore.info('Saving model diagram to file: {}'.format(filepath))
		}
		try{
			plot_model(self.model,to_file=filepath,show_shapes=True,show_layer_names=True,rankdir="TB",expand_nested=False,dpi=300) # show_dtype=False,
		}except Exception as e{
			Utils.LazyCore.exception(e)
		}	
	}

	def buildModel(self,**kwargs){
		self.model,self.callbacks=self._buildModel(**kwargs)
	} 

	def getMetricsNames(self,include_manual=True){
		if include_manual and NeuralNetwork.USE_MANUAL_METRICS{
			list_of=self.model.metrics_names
			for el in NeuralNetwork.MANUAL_METRICS_NAMES{
				if el not in list_of{
					list_of.append(el)
				}
			}
			return list_of
		}else{
			return self.model.metrics_names
		}
	}

	def _metricsFactory(self){
		def auc(y_true, y_pred){
			weight = None
			N = tf.size(y_true, name="N")
			y_true = K.reshape(y_true, shape=(N,))
			y_pred = K.reshape(y_pred, shape=(N,))
			if weight is None{
				weight = tf.fill(K.shape(y_pred), 1.0)
			}
			sort_result = tf.nn.top_k(y_pred, N, sorted=False, name="sort")
			y = K.gather(y_true, sort_result.indices)
			y_hat = K.gather(y_pred, sort_result.indices)
			w = K.gather(weight, sort_result.indices)
			is_negative = K.equal(y, tf.constant(0.0))
			is_positive = K.equal(y, tf.constant(1.0))
			w_zero = tf.fill(K.shape(y_pred), 0.0)
			w_negative = tf.where(is_positive, w_zero, w, name="w_negative")
			w_positive = tf.where(is_negative, w_zero, w)
			cum_positive = K.cumsum(w_positive)
			cum_negative = K.cumsum(w_negative)
			is_diff = K.not_equal(y_hat[:-1], y_hat[1:])
			is_end = tf.concat([is_diff, tf.constant([True])], 0)
			total_positive = cum_positive[-1]
			total_negative = cum_negative[-1]
			TP = tf.concat([
				tf.constant([0.]),
				tf.boolean_mask(cum_positive, is_end),
				], 0)
			FP = tf.concat([
				tf.constant([0.]),
				tf.boolean_mask(cum_negative, is_end),
				], 0)
			FPR = FP / total_negative
			TPR = TP / total_positive
			return K.sum((FPR[1:]-FPR[:-1])*(TPR[:-1]+TPR[1:])/2)
		}

		def precision(y_true, y_pred){
			c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
			if c2 == 0{
				return 0.0
			}
			return c1 / c2
		}

		def recall(y_true, y_pred){
			c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
			if c3 == 0{
				return 0.0
			}
			return c1 / c3
		}

		def f1_score(y_true, y_pred){
			pre = precision(y_true,y_pred)
			rec = recall(y_true,y_pred)
			pre_plus_rec=pre+rec
			if pre_plus_rec == 0{
				return 0.0
			}
			f1_score = 2 * (pre * rec) / pre_plus_rec
			return f1_score
		}

		auc_original=keras.metrics.AUC(name='roc_auc-1')

		auc.__name__ = 'roc_auc-2'
		precision.__name__ = 'precision'
		recall.__name__ = 'recall'
		f1_score.__name__ = 'f1_score'

		if NeuralNetwork.USE_MANUAL_METRICS{
			return ['accuracy']
		}else{
			return [auc,precision,recall,f1_score,'accuracy']
		}
		# return [auc_original,auc,precision,recall,f1_score,'accuracy'] # TODO original auc not allowed
	}

	@abstractmethod
	def _buildModel(self,**kwargs){
		pass
	}
	
	def parseNumpyToVanillaRecursivelly(self,element){
		if type(element) == np.ndarray{
			return self.parseNumpyToVanillaRecursivelly(element.tolist())
		}elif type(element) == list{
			if len(element) == 0{
				return element
			}elif type(element[0]) in (np.float64,np.float32,np.float){
				return list(map(float,element)) # instead of map we should use recursion element by element to handle nested complex elements, but.... the way it is, is enough for now...
			}elif type(element[0]) in (np.int64,np.int32,np.int){
				return list(map(int,element))
			}else{
				raise Exception('Unhandled type {}'.format(type(element[0])))
			}
		}else{
			raise Exception('Unhandled type {}'.format(type(element)))
		}
	}


	def parseHistoryToVanilla(self){
		new_hist = {}
		for key in list(self.history.history.keys()){
			new_hist[key]=self.parseNumpyToVanillaRecursivelly(self.history.history[key])
		}
		self.history=new_hist
	}

	def train(self,features,labels,val_features=None,val_labels=None){
		val_data=None
		verbose=0
		if val_features is not None and val_labels is not None{
			val_data=(val_features,val_labels)
		}
		if self.verbose{
			verbose=2
		}
		self.history=self.model.fit(
			x=self.formatData(features),
			y=self.formatData(labels),
			batch_size=self.hyperparameters.batch_size,
			epochs=self.hyperparameters.max_epochs,
			verbose=verbose,
			callbacks=self.callbacks,
			validation_data=val_data,
			shuffle=self.hyperparameters.shuffle,
			workers=1,
			use_multiprocessing=NeuralNetwork.MULTIPROCESSING
		)
		self.parseHistoryToVanilla()
	}

	def _trainEpoch(self,e,features_epoch,labels_epoch,val_features=None,val_labels=None,best_val=None,epochs_wo_improvement=None){
		if (self.hyperparameters.shuffle){
			if self.hyperparameters.amount_of_networks==1{
				features_epoch,labels_epoch=Dataset.shuffleDataset(features_epoch,labels_epoch)
				if val_labels is not None{
					val_features,val_labels=Dataset.shuffleDataset(val_features,val_labels)
				}
			}else{
				features_epoch,labels_epoch=Dataset.shuffleFeatureGroupedDataset(features_epoch,labels_epoch)
				if val_labels is not None{
					val_features,val_labels=Dataset.shuffleFeatureGroupedDataset(val_features,val_labels)
				}
			}
		}
		batch_size=self.hyperparameters.batch_size
		if batch_size==0{
			batch_size=len(labels_epoch)
		}
		amount_of_batches=int(len(labels_epoch)/batch_size)
		if self.verbose{
			Utils.LazyCore.info('Epoch {} of {}'.format(e+1,self.hyperparameters.max_epochs))
		}
		epoch_metrics=None
		for b in range(amount_of_batches){
			labels_batch=labels_epoch[b*0:(b+1)*batch_size]
			if self.hyperparameters.amount_of_networks==1{
				features_batch=features_epoch[b*0:(b+1)*batch_size]
			}else{
				features_batch=[]
				for feature in features_epoch{
					features_batch.append(feature[b*0:(b+1)*batch_size])
				}
			}
			batch_metrics=self.model.train_on_batch(
				self.formatData(features_batch),
				self.formatData(labels_batch),
				reset_metrics=True
			)
			if epoch_metrics is None{
				epoch_metrics=[[] for _ in range(len(self.getMetricsNames()))]
			}
			for m,metric in enumerate(batch_metrics){
				epoch_metrics[m].append(metric)
			}
			if NeuralNetwork.USE_MANUAL_METRICS and not NeuralNetwork.MANUAL_METRICS_ONLY_ON_VALIDATION{
				classes=self.predict(features_batch,get_classes=True,get_confidence=False)
				manual_stats=Dataset.statisticalAnalysis(classes,labels_batch)
				for k,v in manual_stats.items(){
					if k in epoch_metrics{
						epoch_metrics[k].append(v)
					}
				}
			}
		}
		if self.history is None{
			all_metrics=self.getMetricsNames()
			if val_labels is not None{
				all_metrics+=['val_'+el for el in all_metrics]
			}
			self.history={}
			for metric in all_metrics{
				self.history[metric]=[]
			}
		}
		epoch_metrics=[Utils.mean(metric) for metric in epoch_metrics]
		epoch_metrics=self.fillMetricsNames(epoch_metrics)
		for k,v in epoch_metrics.items(){
			if len(v)>0{
				v=float(v[0])
				epoch_metrics[k]=v
				self.history[k].append(v)
			}
		}
		if val_labels is not None{
			val_metrics=self.fillMetricsNames(self.model.test_on_batch(
				self.formatData(val_features),self.formatData(val_labels), reset_metrics=True))
			for k,v in val_metrics.items(){
				if len(v)>0{
					v=float(v[0])
					val_metrics[k]=v
					self.history['val_'+k].append(v)
				}
			}
			if NeuralNetwork.USE_MANUAL_METRICS{
				classes=self.predict(val_features,get_classes=True,get_confidence=False)
				manual_stats=Dataset.statisticalAnalysis(classes,val_labels)
				for k,v in manual_stats.items(){
					if k in self.history{
						val_metrics[k]=v
						self.history['val_'+k].append(v)
					}
				}
			}
		}
		if self.verbose{
			Utils.LazyCore.logDict(epoch_metrics,'Epoch metrics',inline=True)
			if val_labels is not None{
				Utils.LazyCore.logDict(val_metrics,'Validation metrics',inline=True)
			}
		}
		if val_labels is not None and epochs_wo_improvement is not None{
			if best_val is not None{
				save_checkpoint=False
				search_maximum=self.hyperparameters.monitor_metric.isMaxMetric(self.hyperparameters.loss)
				if (search_maximum and best_val<=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]) or (not search_maximum and best_val>=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]) {
					epochs_wo_improvement=0
					if self.verbose{
						Utils.LazyCore.info('val_{} improved from {:.5f} to {:.5f}'.format(self.hyperparameters.monitor_metric.toKerasName(),best_val,val_metrics[self.hyperparameters.monitor_metric.toKerasName()]))
					}
					best_val=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]
					save_checkpoint=True
				}else{
					if self.verbose{
						Utils.LazyCore.info('val_{} did not improve from {:.5f}'.format(self.hyperparameters.monitor_metric.toKerasName(),best_val))
					}
					epochs_wo_improvement+=1
				}
			}else{
				best_val=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]
				save_checkpoint=True
			}
			if self.hyperparameters.model_checkpoint and save_checkpoint{
				if self.verbose{
					Utils.LazyCore.info('saving checkpoint on {}, epoch {}'.format(self.checkpoint_filename,e+1))
				}
				max_tries=3
				cur_try=0
				done=False
				error_e=None
				while cur_try<max_tries and done==False{
					try{ # need for parallelism
						Utils.deleteFile(self.getModelPath(self.checkpoint_filename),True)
						self.model.save(self.getModelPath(self.checkpoint_filename))
						done=True
					}except Exception as exception_e{
						cur_try+=1
						error_e=exception_e
					}
				}
				if not done{
					Utils.LazyCore.warn('Failed to save checkpoint on epoch {} for {}. Exception: {}.'.format(e,self.name,error_e))
				}
			}
			if self.verbose{
				Utils.LazyCore.info()
			}
			if self.hyperparameters.patience_epochs>0 and epochs_wo_improvement>=self.hyperparameters.patience_epochs{
				if self.verbose {
					Utils.LazyCore.info('Early stopping...')
				}
				best_val=NeuralNetwork.NO_PATIENCE_LEFT_STR
			}
		}
		if epochs_wo_improvement is None {
			return best_val
		}else{
			return best_val,epochs_wo_improvement
		}
	}


	def trainKFolds(self,features,labels,folds){
		self.history=None
		fold_size=int(len(labels)/folds)
		best_val=None
		epochs_wo_improvement=0	
		for e in range(self.hyperparameters.max_epochs){
			val_fold=int((folds-1)*Utils.random())
			if folds>1{
				labels_epoch=labels[0:val_fold*fold_size]
				val_labels=labels[val_fold*fold_size:(val_fold+1)*fold_size]
				if (val_fold!=(folds-1)){
					to_join=labels[(val_fold+1)*fold_size:(folds-val_fold-1)*fold_size]
					if type(labels_epoch) is list {
						labels_epoch+=to_join
					}else{
						labels_epoch=np.concatenate((labels_epoch,to_join))
					}
				}
				if self.hyperparameters.amount_of_networks==1{
					features_epoch=features[0:val_fold*fold_size]
					val_features=features[val_fold*fold_size:(val_fold+1)*fold_size]
					if (val_fold!=(folds-1)){
						to_join=features[(val_fold+1)*fold_size:(folds-val_fold-1)*fold_size]
						if type(features_epoch) is list {
							features_epoch+=to_join
						}else{
							features_epoch=np.concatenate((features_epoch,to_join))
						}
					}
				}else{
					features_epoch=[]
					val_features=[]
					for feature in features{
						features_epoch.append(feature[0:val_fold*fold_size])
						val_features.append(feature[val_fold*fold_size:(val_fold+1)*fold_size])
						if (val_fold!=(folds-1)){
							to_join=feature[(val_fold+1)*fold_size:(folds-val_fold-1)*fold_size]
							if type(features_epoch[-1]) is list {
								features_epoch[-1]+=to_join
							}else{
								features_epoch[-1]=np.concatenate((features_epoch[-1],to_join))
							}
						}
					}
				}
			}else{
				features_epoch=features
				labels_epoch=labels
				val_features=None
				val_labels=None
			}
			best_val,epochs_wo_improvement=self._trainEpoch(e,features_epoch,labels_epoch,val_features,val_labels,best_val,epochs_wo_improvement)
			if best_val==NeuralNetwork.NO_PATIENCE_LEFT_STR{
				break
			}
		}
	}

	def trainNoValidation(self,features,labels){
		self.history=None
		for e in range(self.hyperparameters.max_epochs){
			self._trainEpoch(e,features,labels)
		}
	}

	def trainCustomValidation(self,features,labels,features_val,labels_val){
		self.history=None
		best_val=None
		epochs_wo_improvement=0	
		for e in range(self.hyperparameters.max_epochs){
			best_val,epochs_wo_improvement=self._trainEpoch(e,features,labels,features_val,labels_val,best_val,epochs_wo_improvement)
			if best_val==NeuralNetwork.NO_PATIENCE_LEFT_STR{
				break
			}
		}
	}

	def trainRollingForecast(self,features,labels,min_size_percentage=.5){
		self.history=None
		fixed_train_pos=int(len(labels)*min_size_percentage)
		window_size=int((len(labels)-fixed_train_pos)/self.hyperparameters.max_epochs)
		if (window_size<self.hyperparameters.batch_size){
			window_size=int(self.hyperparameters.batch_size)
		}
		best_val=None
		epochs_wo_improvement=0	
		for e in range(self.hyperparameters.max_epochs){
			start_val=int(fixed_train_pos+e*window_size)
			while start_val+window_size >= len(labels){
				start_val-=window_size
			}
			labels_epoch=labels[0:start_val]
			val_labels=labels[start_val:start_val+window_size]
			if self.hyperparameters.amount_of_networks==1{
				features_epoch=features[0:start_val]
				val_features=features[start_val:start_val+window_size]
			}else{
				features_epoch=[]
				val_features=[]
				for feature in features{
					features_epoch.append(feature[0:start_val])
					val_features.append(feature[start_val:start_val+window_size])
				}
			}
			best_val,epochs_wo_improvement=self._trainEpoch(e,features_epoch,labels_epoch,val_features,val_labels,best_val,epochs_wo_improvement)
			if best_val==NeuralNetwork.NO_PATIENCE_LEFT_STR{
				break
			}
		}
	}

	def predict(self,features,get_classes=True,get_confidence=False){
		pred_res=self.model.predict(self.formatData(features))
		classes=[]
		confidence=[]
		if get_classes or get_confidence {
			for row in pred_res{
				confidence.append(row.tolist())
				row_class=[]
				if self.hyperparameters.node_types[-1]!=NodeType.SOFTMAX{
					for val in row{
						if float(val)>=NeuralNetwork.SIGMOID_CLASSES_THRESHOLD{
							row_class.append(1)
						}else{
							row_class.append(0)
						}
					}
				}else{
					max_idx=0
					max_val=float('-inf')
					for i,val in enumerate(row){
						row_class.append(0)
						if max_val<val{
							max_val=val
							max_idx=i
						}
					}
					row_class[max_idx]=1
				}
				classes.append(row_class)
			}
		}
		if get_classes and get_confidence{
			return classes, confidence
		}elif get_classes{
			return classes
		}elif get_confidence{
			return confidence
		}
		return preds
	}	

	def fillMetricsNames(self,metrics){
		output={}
		for i,metric in enumerate(self.getMetricsNames()){
			if metric not in output{
				output[metric]=[]
			}
			if i<len(metrics){
				metric_val=metrics[i]
				if metric_val is not None{
					output[metric].append(metric_val)
				}
			}
		}
		return output
	}

	def eval(self,features,labels){
		metrics=self.model.evaluate(
			x=self.formatData(features),
			y=self.formatData(labels),
			verbose=0,
			workers=1,
			use_multiprocessing=NeuralNetwork.MULTIPROCESSING
		)
		return self.fillMetricsNames(metrics)
	}

	def _resetWeights(self,model){
		for layer in model.layers{
			if isinstance(layer, tf.keras.Model){
				self._resetWeights(layer)
			}else{
				if hasattr(layer, 'cell'){
					init_container = layer.cell
				}else{
					init_container = layer
				}
				for key, initializer in init_container.__dict__.items(){
					if 'initializer' in key{
						if key == 'recurrent_initializer'{
							var = getattr(init_container, 'recurrent_kernel')
						}else{
							var = getattr(init_container, key.replace('_initializer', ''))
						}
						if var is not None{
							var.assign(initializer(var.shape, var.dtype)) #use the initializer
						}
					}
				}
			}
		}
	}

	def resetWeights(self){
		self._resetWeights(self.model)
	}

	def getWeights(self){
		weights=self.model.get_weights()
		amount_of_networks=self.hyperparameters.amount_of_networks
		amount_of_layers=self.hyperparameters.layers
		boosted_weights={}
		idx=0
		if amount_of_networks == 1 {
			for i in range(amount_of_layers){
				boosted_weights['L_{}'.format(i)]=weights[idx]
				idx+=1
				bias=None
				if self.hyperparameters.bias[i]{
					bias=weights[idx]
					idx+=1
				}
				boosted_weights['B_{}'.format(i)]=bias
			}
		}else{
			for j in range(amount_of_networks){
				for i in range(amount_of_layers[j]){
					boosted_weights['L_{}-{}'.format(j,i)]=weights[idx]
					idx+=1
					bias=None
					if self.hyperparameters.bias[j][i]{
						bias=weights[idx]
						idx+=1
					}
					boosted_weights['B_{}-{}'.format(j,i)]=bias
				}
			}
		}
		if (idx!=len(weights)){
			Utils.LazyCore.warn('Casted {} weights of {}, check the getWeights function'.format(idx,len(weights)))
		}
		return boosted_weights
	}

	def setWeights(self,boosted_weights){
		if boosted_weights is None{
			return
		}
		amount_of_networks=self.hyperparameters.amount_of_networks
		amount_of_layers=self.hyperparameters.layers
		cur_weights=self.getWeights()
		boosted_weights=self.mergeWeights(cur_weights,boosted_weights)
		weights=[]
		if amount_of_networks==1{
			for i in range(amount_of_layers){
				name='L_{}'.format(i)
				if name in boosted_weights{
					weights.append(self.shrinkWeights(boosted_weights[name],cur_weights[name]))
				}else{
					weights.append(cur_weights[name])
				}
				if self.hyperparameters.bias[i]{
					name='B_{}'.format(i)
					if name in boosted_weights{
						weights.append(self.shrinkWeights(boosted_weights[name],cur_weights[name]))
					}else{
						weights.append(cur_weights[name])
					}
				}
			}
		}else{
			for j in range(amount_of_networks){
				for i in range(amount_of_layers[j]){
					name='L_{}-{}'.format(j,i)
					if name in boosted_weights{
						weights.append(self.shrinkWeights(boosted_weights[name],cur_weights[name]))
					}else{
						weights.append(cur_weights[name])
					}
					if self.hyperparameters.bias[j][i]{
						name='B_{}-{}'.format(j,i)
						if name in boosted_weights{
							weights.append(self.shrinkWeights(boosted_weights[name],cur_weights[name]))
						}else{
							weights.append(cur_weights[name])
						}
					}
				}
			}
		}
		self.model.set_weights(weights)
	}

	def getMetricMean(self,metric_name,validation=False){
		limited_to_one=metric_name in ('f1_score','recall','accuracy','precision')
		if validation{
			metric_name='val_'+metric_name
		}
		mean=Utils.mean(self.history[metric_name])
		if limited_to_one and mean > 1 {
			Utils.LazyCore.warn('mean: '+str(mean))
			Utils.LazyCore.warn('metric_name: '+metric_name)
			Utils.printDict(self.history,'history')
		}
		return mean
	}

	def getMetric(self,metric_name,validation=False){
		if validation{
			metric_name='val_'+metric_name
		}
		return self.history[metric_name]
	}

	def mergeWeights(self,weights_old,weights_new=[None]){
		if weights_new==[None]{
			weights_new=self.getWeights()
		}
		if weights_old is None and weights_new is None{
			return None
		}elif weights_old is None and weights_new is not None{
			return weights_new
		}elif weights_old is not None and weights_new is None{
			return weights_old
		}
		weights_m={}
		for k,v in weights_old.items(){
			if v is not None and k not in weights_new{
				weights_m[k]=v
			}
		}
		for k,v in weights_new.items(){
			if v is not None{
				if k in weights_old and weights_old[k] is not None{
					weights_m[k]=self.fillWeights(v,weights_old[k])
				}else {
					weights_m[k]=v
				}
			}
		}
		return weights_m
	}

	def shrinkWeights(self,weights,weights_with_format){
		shape=list(weights.shape)
		desired_shape=list(weights_with_format.shape)
		if shape==desired_shape{
			return weights
		}
		if len(desired_shape)==1{
			desired_shape.append(None)
		}
		desired_shape_1=1 if desired_shape[1] is None else desired_shape[1]
		new_weights=[]
		for i in range(desired_shape[0]){
			array=[]
			for j in range(desired_shape_1){
				value=None
				if i < shape[0] and ((desired_shape[1] is None and j==0) or j < len(weights[i])){
					if desired_shape[1] is not None{
						value=weights[i][j]
					}else{
						value=weights[i]
					}
				}else{
					if desired_shape[1] is not None{
						value=weights_with_format[i][j]
					}else{
						value=weights_with_format[i]
					}
				}
				array.append(value)
			}
			if desired_shape[1] is not None{
				new_weights.append(np.array(array,dtype=weights_with_format.dtype))
			}else{
				new_weights.append(array[0])
			}
		}
		new_weights=np.array(new_weights,dtype=weights_with_format.dtype)
		new_weights_shape=list(new_weights.shape)
		if len(new_weights_shape)==1{
			new_weights_shape.append(None)
		}
		if new_weights_shape!=desired_shape{
			Utils.LazyCore.warn('Error on shrinkWeights, trying to format shape {} into {}, result {}'.format(shape,desired_shape,new_weights_shape))
		}
		return new_weights
	}

	def fillWeights(self,weights_a,weights_b){
		shape_a=list(weights_a.shape)
		shape_b=list(weights_b.shape)
		if shape_a==shape_b{
			return weights_a
		}
		desired_shape=[0,1]
		if shape_a[0] > shape_b[0] {
			desired_shape[0]=shape_a[0]
		}else{
			desired_shape[0]=shape_b[0]
		}
		if len(shape_a) > 1 and len(shape_b) > 1{
			if shape_a[1] > shape_b[1] {
				desired_shape[1]=shape_a[1]
			}else{
				desired_shape[1]=shape_b[1]
			}
		}elif len(shape_a) > 1{
			desired_shape[1]=shape_a[1]
			shape_b.append(None)
		}elif len(shape_b) > 1{
			desired_shape[1]=shape_b[1]
			shape_a.append(None)
		}else{
			shape_a.append(None)
			shape_b.append(None)
		}
		new_weights=[]
		for i in range(desired_shape[0]){
			array=[]
			for j in range(desired_shape[1]){
				value=None
				if i < shape_a[0] and ((shape_a[1] is None and j==0) or j < len(weights_a[i])){
					if shape_a[1] is not None{
						value=weights_a[i][j]
					}else{
						value=weights_a[i]
					}
				}elif i < shape_b[0] and ((shape_b[1] is None and j==0) or j < len(weights_b[i])){
					if shape_b[1] is not None{
						value=weights_b[i][j]
					}else{
						value=weights_b[i]
					}
				}else{
					value=Utils.randomFloat(-1,1)
				}
				array.append(value)
			}
			if shape_a[1] is not None and shape_b[1] is not None{
				new_weights.append(np.array(array,dtype=weights_b.dtype))
			}else{
				new_weights.append(array[0])
			}
		}
		new_weights=np.array(new_weights,dtype=weights_b.dtype)
		new_weights_shape=list(new_weights.shape)
		if len(new_weights_shape)==1{
			new_weights_shape.append(1)
		}
		if new_weights_shape!=desired_shape{
			Utils.LazyCore.warn('Error on fillWeights, trying to format biggest shape between {} and {}, result {} - expected {}'.format(shape_a,shape_b,new_weights_shape,desired_shape))
		}
		return new_weights
	}
}
