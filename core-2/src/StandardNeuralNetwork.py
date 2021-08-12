#!/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import pandas as pd
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import Adam, SGD
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Utils import Utils
from Dataset import Dataset
from Enums import NodeType 
from Core import Core 

class StandardNeuralNetwork(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

	MODELS_PATH='models'
	MULTIPROCESSING=False
	NO_PATIENCE_LEFT_STR='Stop Epochs - No patience left'

    def __init__(self,hyperparameters,dataset_name='',verbose=False){
        self.hyperparameters=hyperparameters
		self.dataset_name=dataset_name
		self.verbose=verbose
		self.data=None
		self.model=None
		self.callbacks=None
		self.metrics_names=None
		self.basename='model'
		self.checkpoint_filename=None
		self.history=[]
		self.metrics={}
		Utils.createFolder(StandardNeuralNetwork.MODELS_PATH)
    }

	def __del__(self){
		self.clearCache()
	}

	def clearCache(self){
		keras.backend.clear_session()
	}

	def getModelPath(self,filename){
		path=filename
		if Utils.appendToStrIfDoesNotEndsWith(StandardNeuralNetwork.MODELS_PATH,Utils.FILE_SEPARATOR) not in path{
			path=Utils.joinPath(StandardNeuralNetwork.MODELS_PATH,filename)
		}
		return path
	}

	def buildModel(self,input_size){
		self.model,self.callbacks=self._buildModel(input_size)
		self.metrics_names=self.model.metrics_names
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
		return [auc,precision,recall,f1_score,'accuracy']
		# return [auc_original,auc,precision,recall,f1_score,'accuracy'] # TODO original auc not allowed
	}

	def _buildModel(self,input_size){
		batch_size=self.hyperparameters.batch_size
		batch_size=None # already using batch size on train function
		inputs=Input(shape=(input_size,),batch_size=batch_size,dtype=np.float32,name='In')
		for l in range(self.hyperparameters.layers){
			if l==0{
				last_layer=inputs
			}else{
				last_layer=layer
			}
			layer=Dense(self.hyperparameters.layer_sizes[l], name='L{}'.format(l),activation=self.hyperparameters.node_types[l].toKerasName(), use_bias=self.hyperparameters.bias[l], kernel_initializer='glorot_uniform', bias_initializer='zeros')(last_layer)
			if self.hyperparameters.dropouts[l]>0{
				layer=Dropout(self.hyperparameters.dropouts[l], name='D{}'.format(l))(layer)
			}
		}
		outputs=layer
		model=Model(inputs=inputs, outputs=outputs,name=self.dataset_name)
		if self.hyperparameters.adam==True{
			opt=Adam(learning_rate=self.hyperparameters.alpha)
		}else{
			opt=SGD(learning_rate=self.hyperparameters.alpha)
		}
		model.compile(loss=self.hyperparameters.loss.toKerasName(),optimizer=opt,metrics=self._metricsFactory())
		if self.verbose{
			model_summary_lines=[]
			model.summary(print_fn=lambda x: model_summary_lines.append(x))
			model_summary_str='\n'.join(model_summary_lines)+'\n'
			Core.LOGGER.multiline(model_summary_str)
		}
		callbacks=[]
		if self.hyperparameters.patience_epochs>0{
			early_stopping=EarlyStopping(monitor='val_'+self.hyperparameters.monitor_metric.toKerasName(), mode='min', patience=self.hyperparameters.patience_epochs,verbose=1)
			callbacks.append(early_stopping)
		}
		if self.hyperparameters.model_checkpoint{
			checkpoint_filename=self.basename+'_cp.h5'
			self.checkpoint_filename=checkpoint_filename
			checkpoint_filepath=self.getModelPath(self.checkpoint_filename)
			checkpoint=ModelCheckpoint(checkpoint_filepath, monitor='val_'+self.hyperparameters.monitor_metric.toKerasName(), verbose=1 if self.verbose else 0, save_best_only=True, mode='auto')
			callbacks.append(checkpoint)
		}
		self._resetWeights(model)
		return model,callbacks
	}

	def parseHistoryToVanilla(self){
		new_hist = {}
		for key in list(self.history.history.keys()){
			new_hist[key]=self.history.history[key]
			if type(self.history.history[key]) == np.ndarray{
				new_hist[key] = self.history.history[key].tolist()
			}elif type(self.history.history[key]) == list{
				if  type(self.history.history[key][0]) == np.float64{
					new_hist[key] = list(map(float, self.history.history[key]))
				}
			}
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
			x=features,
			y=labels,
			batch_size=self.hyperparameters.batch_size,
			epochs=self.hyperparameters.max_epochs,
			verbose=verbose,
			callbacks=self.callbacks,
			validation_data=val_data,
			shuffle=self.hyperparameters.shuffle,
			workers=1,
			use_multiprocessing=StandardNeuralNetwork.MULTIPROCESSING
		)
		self.parseHistoryToVanilla()
	}

	def _trainEpoch(self,e,features_epoch,labels_epoch,val_features=None,val_labels=None,best_val=None,epochs_wo_improvement=None){
		if (self.hyperparameters.shuffle){
			features_epoch,labels_epoch=Dataset.shuffleDataset(features_epoch,labels_epoch)
			if val_labels is not None{
				val_features,val_labels=Dataset.shuffleDataset(val_features,val_labels)
			}
		}
		batch_size=max(min(len(labels_epoch),self.hyperparameters.batch_size),1)
		batch_width=int(len(labels_epoch)/batch_size)
		if self.verbose{
			Core.LOGGER.info('Epoch {} of {}'.format(e+1,self.hyperparameters.max_epochs))
		}
		epoch_metrics=None
		for b in range(batch_size){
			features_batch=features_epoch[b*0:(b+1)*batch_width]
			labels_batch=labels_epoch[b*0:(b+1)*batch_width]
			batch_metrics=self.model.train_on_batch(
				np.array(features_batch),
				np.array(labels_batch),
				reset_metrics=True
			)
			if epoch_metrics is None{
				epoch_metrics=[[] for _ in range(len(self.model.metrics_names))]
			}
			for m,metric in enumerate(batch_metrics){
				epoch_metrics[m].append(metric)
			}
		}
		if self.history is None{
			all_metrics=self.model.metrics_names
			if val_labels is not None{
				all_metrics+=['val_'+el for el in all_metrics]
			}
			self.history=dict.fromkeys(all_metrics,[])
		}
		epoch_metrics=[Utils.mean(metric) for metric in epoch_metrics]
		epoch_metrics=self.fillMetricsNames(epoch_metrics)
		for k,v in epoch_metrics.items(){
			v=float(v[0])
			epoch_metrics[k]=v
			self.history[k].append(v)
		}
		if val_labels is not None{
			val_metrics=self.fillMetricsNames(self.model.test_on_batch(
				np.array(val_features), np.array(val_labels), reset_metrics=True))
			for k,v in val_metrics.items(){
				v=float(v[0])
				val_metrics[k]=v
				self.history['val_'+k].append(v)
			}
		}
		if self.verbose{
			Core.LOGGER.logDict(epoch_metrics,'Epoch metrics',inline=True)
			if val_labels is not None{
				Core.LOGGER.logDict(val_metrics,'Validation metrics',inline=True)
			}
		}
		if val_labels is not None and epochs_wo_improvement is not None{
			if best_val is not None{
				if best_val<=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]{
					if self.verbose{
						Core.LOGGER.info('val_{} did not improve from {}'.format(self.hyperparameters.monitor_metric.toKerasName(),best_val))
					}
					epochs_wo_improvement+=1
				}else{
					epochs_wo_improvement=0
					best_val=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]
					if self.verbose{
						Core.LOGGER.info('val_{} improved to {}'.format(self.hyperparameters.monitor_metric.toKerasName(),best_val))
					}
					if self.hyperparameters.model_checkpoint{
						if self.verbose{
							Core.LOGGER.info('saving checkpoint on {}, epoch {}'.format(self.checkpoint_filename,e+1))
						}
						self.model.save(self.getModelPath(self.checkpoint_filename))
					}
				}
			}else{
				best_val=val_metrics[self.hyperparameters.monitor_metric.toKerasName()]
			}
			if self.verbose{
				Core.LOGGER.info()
			}
			if self.hyperparameters.patience_epochs>0 and epochs_wo_improvement>=self.hyperparameters.patience_epochs{
				if self.verbose {
					Core.LOGGER.info('Early stopping...')
				}
				best_val=StandardNeuralNetwork.NO_PATIENCE_LEFT_STR
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
				features_epoch=features[0:val_fold*fold_size]
				labels_epoch=labels[0:val_fold*fold_size]
				val_features=features[val_fold*fold_size:(val_fold+1)*fold_size]
				val_labels=labels[val_fold*fold_size:(val_fold+1)*fold_size]
				if (val_fold!=(folds-1)){
					features_epoch+=features[(val_fold+1)*fold_size:(folds-val_fold-1)*fold_size]
					labels_epoch+=labels[(val_fold+1)*fold_size:(folds-val_fold-1)*fold_size]
				}
			}else{
				features_epoch=features
				labels_epoch=labels
				val_features=None
				val_labels=None
			}
			best_val,epochs_wo_improvement=self._trainEpoch(e,features_epoch,labels_epoch,val_features,val_labels,best_val,epochs_wo_improvement)
			if best_val==StandardNeuralNetwork.NO_PATIENCE_LEFT_STR{
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
			if best_val==StandardNeuralNetwork.NO_PATIENCE_LEFT_STR{
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
			while start_val >= len(labels){
				start_val-=window_size
			}
			features_epoch=features[0:start_val]
			labels_epoch=labels[0:start_val]
			val_features=features[start_val:start_val+window_size]
			val_labels=labels[start_val:start_val+window_size]
			best_val,epochs_wo_improvement=self._trainEpoch(e,features_epoch,labels_epoch,val_features,val_labels,best_val,epochs_wo_improvement)
			if best_val==StandardNeuralNetwork.NO_PATIENCE_LEFT_STR{
				break
			}
		}
	}

	def predict(self,features,get_classes=True,get_confidence=False,threshold=.5){
		pred_res=self.model.predict(features)
		classes=[]
		confidence=[]
		if get_classes or get_confidence {
			for row in pred_res{
				confidence.append(row.tolist())
				row_class=[]
				if self.hyperparameters.node_types[-1]!=NodeType.SOFTMAX{
					for val in row{
						if float(val)>=threshold{
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
		self.metrics_names=self.model.metrics_names
		output={}
		for i,metric in enumerate(self.metrics_names){
			if metric not in output{
				output[metric]=[]
			}
			output[metric].append(metrics[i])
		}
		return output
	}

	def eval(self,features,labels){
		metrics=self.model.evaluate(
			x=features,
			y=labels,
			verbose=0,
			workers=1,
			use_multiprocessing=StandardNeuralNetwork.MULTIPROCESSING
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
		amount_of_layers=self.hyperparameters.layers
		boosted_weights={}
		idx=0
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
		if (idx!=len(weights)){
			Core.LOGGER.warn('Casted {} weights of {}, check the getWeights function'.format(idx,len(weights)))
		}
		return boosted_weights
	}

	def setWeights(self,boosted_weights){
		if boosted_weights is None{
			return
		}
		amount_of_layers=self.hyperparameters.layers
		cur_weights=self.getWeights()
		boosted_weights=self.mergeWeights(cur_weights,boosted_weights)
		weights=[]
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
		self.model.set_weights(weights)
	}

	def getMetricMean(self,metric_name,Validation=False){
		if Validation{
			metric_name='val_'+metric_name
		}
		return Utils.mean(self.history[metric_name])
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
			Core.LOGGER.warn('Error on shrinkWeights, trying to format shape {} into {}, result {}'.format(shape,desired_shape,new_weights_shape))
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
			Core.LOGGER.warn('Error on fillWeights, trying to format biggest shape between {} and {}, result {} - expected {}'.format(shape_a,shape_b,new_weights_shape,desired_shape))
		}
		return new_weights
	}

}