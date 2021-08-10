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
		return [auc_original,auc,precision,recall,f1_score,'accuracy']
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
		model.compile(loss=self.hyperparameters.loss,optimizer=opt,metrics=self._metricsFactory())
		if self.verbose{
			print(model.summary())
		}
		callbacks=[]
		if self.hyperparameters.patience_epochs>0{
			early_stopping=EarlyStopping(monitor='val_'+self.hyperparameters.monitor_metric, mode='min', patience=self.hyperparameters.patience_epochs,verbose=1)
			callbacks.append(early_stopping)
		}
		if self.hyperparameters.model_checkpoint{
			checkpoint_filename=self.basename+'_cp.h5'
			self.checkpoint_filename=checkpoint_filename
			checkpoint_filepath=self.getModelPath(self.checkpoint_filename)
			checkpoint=ModelCheckpoint(checkpoint_filepath, monitor='val_'+self.hyperparameters.monitor_metric, verbose=1, save_best_only=True, mode='auto')
			callbacks.append(checkpoint)
		}
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
			print('Epoch {} of {}'.format(e+1,self.hyperparameters.max_epochs))
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
		epoch_metrics=[float(sum(metric)/len(metric)) for metric in epoch_metrics]
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
			Utils.printDict(epoch_metrics,'Epoch metrics',inline=True)
			if val_labels is not None{
				Utils.printDict(val_metrics,'Validation metrics',inline=True)
			}
		}
		if val_labels is not None and epochs_wo_improvement is not None{
			if best_val is not None{
				if best_val<=val_metrics[self.hyperparameters.monitor_metric]{
					print('val_{} did not improve from {}'.format(self.hyperparameters.monitor_metric,best_val))
					epochs_wo_improvement+=1
				}else{
					epochs_wo_improvement=0
					best_val=val_metrics[self.hyperparameters.monitor_metric]
					print('val_{} improved to {}'.format(self.hyperparameters.monitor_metric,best_val))
					if self.hyperparameters.model_checkpoint{
						print('saving checkpoint on {}, epoch {}'.format(self.checkpoint_filename,e+1))
						self.model.save(self.getModelPath(self.checkpoint_filename))
					}
				}
			}else{
				best_val=val_metrics[self.hyperparameters.monitor_metric]
			}
			print()
			if self.hyperparameters.patience_epochs>0 and epochs_wo_improvement>=self.hyperparameters.patience_epochs{
				print('Early stopping...')
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

	def predict(self,features){
		return self.model.predict(features)
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

	def getWeights(self){
		return self.model.get_weights()
	}

	def setWeights(self,weights){
		self.model.set_weights(weights)
	}

}