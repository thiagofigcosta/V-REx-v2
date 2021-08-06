#!/bin/python

class Hyperparameters(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    def __init__(self, batch_size, alpha, shuffle, adam, label_type, layers, layer_sizes, node_types){
        self.batch_size=batch_size
		self.alpha=alpha
		self.shuffle=shuffle
		self.adam=adam
		self.label_type=label_type
		self.layers=layers
		self.layer_sizes=layer_sizes
		self.node_types=node_types
    }
}