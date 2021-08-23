#!/bin/python

from NeuralNetwork import NeuralNetwork
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input # from keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model # from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from Enums import Optimizers,NodeType
from Utils import Utils
import numpy as np

class StandardNeuralNetwork(NeuralNetwork){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    def __init__(self,hyperparameters,name='',verbose=False){
        super().__init__(hyperparameters,name,verbose)
    }

	def _buildModel(self,**kwargs){
		input_size=kwargs.get('input_size')
		batch_size=self.hyperparameters.batch_size
		batch_size=None # already using batch size on train function
		inputs=Input(shape=(input_size,),batch_size=batch_size,dtype=np.float32,name='In')
		for l in range(self.hyperparameters.layers){
			if l==0{
				last_layer=inputs
			}else{
				last_layer=layer
			}
			if self.hyperparameters.node_types[l]!=NodeType.RELU or not NeuralNetwork.USE_LEAKY_RELU{
				activation=self.hyperparameters.node_types[l].toKerasName()
				advanced_activation=False
			}else{
				activation=NodeType.LINEAR.toKerasName()
				advanced_activation=True
			}
			layer=Dense(self.hyperparameters.layer_sizes[l], name='L{}'.format(l),activation=activation, use_bias=self.hyperparameters.bias[l], kernel_initializer='glorot_uniform', bias_initializer='zeros')(last_layer)
			if advanced_activation{
				layer=LeakyReLU(alpha=0.1, name='LeakyRelu{}'.format(l))(layer)
			}
			if self.hyperparameters.dropouts[l]>0{
				layer=Dropout(self.hyperparameters.dropouts[l], name='D{}'.format(l))(layer)
			}
		}
		outputs=layer
		model=Model(inputs=inputs, outputs=outputs,name=self.name)
		clip_dict={}
		if NeuralNetwork.CLIP_NORM_INSTEAD_OF_VALUE{
			clip_dict['clipnorm']=1.0
		}else{
			clip_dict['clipvalue']=0.5
		}
		if self.hyperparameters.optimizer==Optimizers.SGD{
			opt=SGD(learning_rate=self.hyperparameters.alpha, **clip_dict)
		}elif self.hyperparameters.optimizer==Optimizers.ADAM{
			opt=Adam(learning_rate=self.hyperparameters.alpha, **clip_dict)
		}elif self.hyperparameters.optimizer==Optimizers.RMSPROP{
			opt=RMSprop(learning_rate=self.hyperparameters.alpha, **clip_dict)
		}else{
			raise Exception('Unknown optimizer {}'.format(self.hyperparameters.optimizer))
		}
		model.compile(loss=self.hyperparameters.loss.toKerasName(),optimizer=opt,metrics=self._metricsFactory())
		if self.verbose{
			model_summary_lines=[]
			model.summary(print_fn=lambda x: model_summary_lines.append(x))
			model_summary_str='\n'.join(model_summary_lines)+'\n'
			Utils.LazyCore.multiline(model_summary_str)
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
}
