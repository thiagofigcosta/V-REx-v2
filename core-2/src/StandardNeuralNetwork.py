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

class StandardNeuralNetwork(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

	MODELS_PATH='models'
	MULTIPROCESSING=False

    def __init__(self,hyperparameters,dataset_name='',verbose=False){
        self.hyperparameters=hyperparameters
		self.dataset_name=dataset_name
		self.verbose=verbose
		self.data=None
		self.model=None
		self.callbacks=None
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

	def buildModel(self,input_size){
		self.model,self.callbacks=self._buildModel(input_size)
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
		return [auc_original,auc,precision,recall,f1_score]
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
		model.compile(loss='binary_crossentropy',optimizer=opt,metrics=self._metricsFactory()+['accuracy'])
		if self.verbose{
			print(model.summary())
		}
		callbacks=[]
		if self.hyperparameters.patience_epochs>0{
			early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=self.hyperparameters.patience_epochs,verbose=1)
			callbacks.append(early_stopping)
		}
		if self.hyperparameters.model_checkpoint{
			checkpoint_filename=self.basename+'_cp.h5'
			self.checkpoint_filename=checkpoint_filename
			checkpoint_filepath=Utils.joinPath(StandardNeuralNetwork.MODELS_PATH,checkpoint_filename)
			checkpoint=ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
			callbacks.append(checkpoint)
		}
		return model,callbacks
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

}