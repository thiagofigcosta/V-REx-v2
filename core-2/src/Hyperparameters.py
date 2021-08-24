#!/bin/python

from Enums import Metric,NodeType,Optimizers,LabelEncoding,Loss

class Hyperparameters(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    def __init__(self, batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss, model_checkpoint=True, monitor_metric=Metric.RAW_LOSS,amount_of_networks=1){
        self.batch_size=batch_size
		self.alpha=alpha
		self.shuffle=shuffle
		self.optimizer=optimizer
		self.label_type=label_type
		self.model_checkpoint=model_checkpoint
		self.patience_epochs=patience_epochs
		self.max_epochs=max_epochs
		self.layers=layers
		self.layer_sizes=layer_sizes
		self.node_types=node_types
		self.dropouts=dropouts
		self.bias=bias
		self.loss=loss
		self.monitor_metric=monitor_metric
		self.amount_of_networks=amount_of_networks

		if type(self.amount_of_networks) is not int {
			raise Exception('amount_of_networks must be integer')
		}

		if self.amount_of_networks == 1 {
			if type(self.batch_size) is not int {
				raise Exception('batch_size must be integer')
			}
			if type(self.alpha) not in (float,int) {
				raise Exception('alpha must be float')
			}
			if type(self.shuffle) is not bool {
				raise Exception('shuffle must be bool')
			}
			if type(self.optimizer) is not Optimizers {
				raise Exception('optimizer must be Optimizers')
			}
			if type(self.label_type) is not LabelEncoding {
				raise Exception('label_type must be LabelEncoding')
			}
			if type(self.model_checkpoint) is not bool {
				raise Exception('model_checkpoint must be bool')
			}
			if type(self.patience_epochs) is not int {
				raise Exception('patience_epochs must be integer')
			}
			if type(self.max_epochs) is not int {
				raise Exception('max_epochs must be integer')
			}
			if type(self.layers) is not int {
				raise Exception('layers must be integer')
			}
			if type(self.layer_sizes[0]) is not int {
				raise Exception('layer_sizes must contain integers')
			}
			if type(self.node_types[0]) is not NodeType {
				raise Exception('node_types must contain NodeTypes')
			}
			if type(self.loss) is not Loss {
				raise Exception('loss must be Loss')
			}
			if type(self.monitor_metric) is not Metric {
				raise Exception('monitor_metric must be Metric')
			}

			if type(self.dropouts) is not list {
				self.dropouts=[self.dropouts]*self.layers
			}
			if type(self.bias) is not list {
				self.bias=[self.bias]*self.layers
			}

			if type(self.bias[0]) is not bool {
				raise Exception('bias must contain bools')
			}
			if type(self.dropouts[0]) not in (float,int) {
				raise Exception('dropouts must contain floats')
			}

			if self.layers != len(self.layer_sizes){
				raise Exception('len(layer sizes) different from amount of layers')
			}
			if self.layers != len(self.node_types){
				raise Exception('len(node types) different from amount of layers')
			}
			if self.layers != len(self.dropouts) {
				raise Exception('len(dropouts) different from amount of layers')
			}
			if self.layers != len(self.bias) {
				raise Exception('len(bias) different from amount of layers')
			}

			self.setLastLayer(self.layer_sizes[-1],self.node_types[-1])
		}else{
			if len(self.layers)!=self.amount_of_networks{
				raise Exception('(layers) different from amount_of_networks')
			}
			if len(self.dropouts)!=self.amount_of_networks{
				raise Exception('(dropouts) different from amount_of_networks')
			}
			if len(self.layer_sizes)!=self.amount_of_networks{
				raise Exception('(layer_sizes) different from amount_of_networks')
			}
			if len(self.node_types)!=self.amount_of_networks{
				raise Exception('(node_types) different from amount_of_networks')
			}
			if len(self.alpha)!=self.amount_of_networks{
				raise Exception('(alpha) different from amount_of_networks')
			}
			if len(self.shuffle)!=self.amount_of_networks{
				raise Exception('(shuffle) different from amount_of_networks')
			}
			if len(self.loss)!=self.amount_of_networks{
				raise Exception('(loss) different from amount_of_networks')
			}
			for i in range(len(self.dropouts)){
				if type(self.dropouts[i]) is not list {
					self.dropouts[i]=[self.dropouts[i]]*self.layers[i]
				} 
			}
			for i in range(len(self.bias)){
				if type(self.bias[i]) is not list {
					self.bias[i]=[self.bias[i]]*self.layers[i]
				} 
			}

			for i in range(len(self.layer_sizes)){
				if len(self.layer_sizes[i]) != self.layers[i]{
					raise Exception('len(layer_sizes[{}]) different from amount of layers'.format(i))
				} 
			}
			for i in range(len(self.node_types)){
				if len(self.node_types[i]) != self.layers[i]{
					raise Exception('len(node_types[{}]) different from amount of layers'.format(i))
				} 
			}
			for i in range(len(self.dropouts)){
				if len(self.dropouts[i]) != self.layers[i]{
					raise Exception('len(dropouts[{}]) different from amount of layers'.format(i))
				} 
			}
			for i in range(len(self.bias)){
				if len(self.bias[i]) != self.layers[i]{
					raise Exception('len(bias[{}]) different from amount of layers'.format(i))
				} 
			}
		}
    }

	def __str__(self){
		str_out='Hyperparameters: {\n'
		str_out+='\t{}: {}'.format('amount_of_networks',self.amount_of_networks)
		str_out+='\t{}: {}'.format('batch_size',self.batch_size)
		str_out+='\t{}: {}'.format('alpha',self.alpha)
		str_out+='\t{}: {}'.format('shuffle',self.shuffle)
		str_out+='\t{}: {}'.format('optimizer',self.optimizer)
		str_out+='\t{}: {}'.format('label_type',self.label_type)
		str_out+='\t{}: {}'.format('model_checkpoint',self.model_checkpoint)
		str_out+='\t{}: {}'.format('max_epochs',self.max_epochs)
		str_out+='\t{}: {}'.format('patience_epochs',self.patience_epochs)
		str_out+='\t{}: {}'.format('loss',self.loss)
		str_out+='\t{}: {}'.format('monitor_metric',self.monitor_metric)
		str_out+='\t{}: {}'.format('layers',self.layers)
		str_out+='\t{}: {}'.format('layer_sizes',self.layer_sizes)
		str_out+='\t{}: {}'.format('node_types',self.node_types)
		str_out+='\t{}: {}'.format('dropouts',self.dropouts)
		str_out+='\t{}: {}'.format('bias',self.bias)
		str_out+='}'
		return str_out
	}

	def setLastLayer(self,output_size,out_node_type){
		if out_node_type==NodeType.SOFTMAX and output_size==1{
			out_node_type=NodeType.SIGMOID # keras returns 1 and 0 for 1 sized softmax
		}
		self.layer_sizes[-1]=output_size
		self.node_types[-1]=out_node_type
	}
}