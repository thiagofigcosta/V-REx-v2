#!/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import pandas as pd
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Utils import Utils

class StandardNeuralNetwork(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

	MODELS_PATH='models'

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
		Utils.createFolder(NeuralNetwork.MODELS_PATH)
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

	def _buildModel(self,input_size){
		inputs=Input(shape=(input_size,),batch_size=self.hyperparameters.batch_size,dtype=float32,name='In')
		for l in range(self.hyperparameters.layers):
			if l==0:
				last_layer=inputs
			else:
				last_layer=layer
			layer=Dense(self.hyperparameters.layer_sizes[l], name='L{}'.format(l) ,activation=self.hyperparameters.node_types[l].toKerasName(), use_bias=self.hyperparameters.bias[l], kernel_initializer='glorot_uniform', bias_initializer='zeros')(last_layer)
			if self.hyperparameters.dropouts[l]>0:
				layer=Dropout(self.hyperparameters.dropouts[l], name='D{}'.format(l))(layer)
		outputs=layer
		model=Model(inputs=inputs, outputs=outputs,name=self.dataset_name)
		if self.hyperparameters.adam==True{
			opt=Adam(learning_rate=self.hyperparameters.alpha)
		}else{
			opt=SGD(learning_rate=self.hyperparameters.alpha)
		}
		model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['auc','precision','recall','truepositives','truenegatives','falsepositives','falsenegatives'])
		if self.verbose:
			print(model.summary())
		callbacks=[]
		if self.hyperparameters.patient_epochs>0{
			early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=self.hyperparameters.patience_epochs,verbose=1)
			callbacks.append(early_stopping)
		}
		if self.hyperparameters.model_checkpoint{
			checkpoint_filename=self.basename+'_cp.h5'
			self.checkpoint_filename=checkpoint_filename
			checkpoint_filepath=Utils.joinPath(NeuralNetwork.MODELS_PATH,checkpoint_filename)
			checkpoint=ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
			callbacks.append(checkpoint)
		}
		return model,callbacks
	}
}