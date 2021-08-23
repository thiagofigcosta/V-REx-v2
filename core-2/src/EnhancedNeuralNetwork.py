#!/bin/python

from NeuralNetwork import NeuralNetwork
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input # from keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model # from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import unpack_x_y_sample_weight
import tensorflow as tf
from Enums import Optimizers
from Utils import Utils
import numpy as np

class EnhancedNeuralNetwork(NeuralNetwork){
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
			layer=Dense(self.hyperparameters.layer_sizes[l], name='L{}'.format(l),activation=self.hyperparameters.node_types[l].toKerasName(), use_bias=self.hyperparameters.bias[l], kernel_initializer='glorot_uniform', bias_initializer='zeros')(last_layer)
			if self.hyperparameters.dropouts[l]>0{
				layer=Dropout(self.hyperparameters.dropouts[l], name='D{}'.format(l))(layer)
			}
		}
		outputs=layer
		model=EnhancedNeuralNetwork.EnhancedModel(inputs=inputs, outputs=outputs,name=self.name)
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

	class EnhancedModel(Model){
		# 'Just':'to fix vscode coloring':'when using pytho{\}'

		def __init__(self,*args, **kwargs){
    		super().__init__(*args, **kwargs)
		}

		# def train_step(self, data){
		# 	x, y, sample_weight = unpack_x_y_sample_weight(data) # TODO should I just do `x, y = data` ?
		# 	# Run forward pass.
		# 	with tf.GradientTape() as tape{
		# 		y_pred = self(x, training=True)
		# 		loss = self.compiled_loss(
		# 			y, y_pred, sample_weight, regularization_losses=self.losses)

		# 		print('y_pred: {}, y: {}, loss: {}'.format(y_pred,y,loss))
		# 	}
		# 	if self.loss and y is None{
		# 		raise TypeError('Target data is missing. Your model has `loss`: {}, and therefore expects target data to be passed in `fit()`.'.format(self.loss))
		# 	}
		# 	# Run backwards pass.
		# 	self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
		# 	self.compiled_metrics.update_state(y, y_pred, sample_weight)
		# 	# Collect metrics to return
		# 	return_metrics = {}
		# 	for metric in self.metrics{
		# 		result = metric.result()
		# 		if isinstance(result, dict){
		# 			return_metrics.update(result)
		# 		}else{
		# 			return_metrics[metric.name] = result
		# 		}
		# 	}
		# 	return return_metrics
		# }

		def train_step(self, data){
			# Unpack the data. Its structure depends on your model and
			# on what you pass to `fit()`.
			x, y = data

			with tf.GradientTape() as tape{
				y_pred = self(x, training=True)  # Forward pass
				# Compute the loss value
				# (the loss function is configured in `compile()`)
				loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
			}

			print('1 | y_pred: {}, y: {}, loss: {}'.format(y_pred,y,loss))

			# Compute gradients
			trainable_vars = self.trainable_variables
			gradients = tape.gradient(loss, trainable_vars)
			# Update weights
			self.optimizer.apply_gradients(zip(gradients, trainable_vars))
			# Update metrics (includes the metric that tracks the loss)
			self.compiled_metrics.update_state(y, y_pred)
			# Return a dict mapping metric names to current value

			return_metrics = {}
			for m in self.metrics {
				return_metrics[m.name]=m.result()
			}
			return return_metrics
		}
	}
}
