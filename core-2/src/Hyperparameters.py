#!/bin/python

class Hyperparameters(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    def __init__(self, batch_size, alpha, shuffle, adam, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, model_checkpoint=True){
        self.batch_size=batch_size
		self.alpha=alpha
		self.shuffle=shuffle
		self.adam=adam
		self.label_type=label_type
		self.model_checkpoint=model_checkpoint
		self.patience_epochs=patience_epochs
		self.max_epochs=max_epochs
		self.layers=layers
		self.layer_sizes=layer_sizes
		self.node_types=node_types
		self.dropouts=dropouts
		self.bias=bias

		if type(self.dropouts) is not list {
			self.dropouts=[self.dropouts]*self.layers
		}
		if type(self.bias) is not list {
			self.bias=[self.bias]*self.layers
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
    }
}