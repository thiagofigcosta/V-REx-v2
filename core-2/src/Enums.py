#!/bin/python
# -*- coding: utf-8 -*-

from enum import Enum
 

class CrossValidation(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    NONE = 0
    ROLLING_FORECASTING_ORIGIN = 1
    KFOLDS = 2
    TWENTY_PERCENT = 3
}

class Metric(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    RAW_LOSS = 0
    F1 = 1
    RECALL = 2
    ACCURACY = 3
    PRECISION = 4
}

class LabelEncoding(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    INT_CLASS = 0
    NEURON_BY_NEURON = 1
    NEURON_BY_N_LOG_LOSS = 2
}

class GeneticAlgorithmType(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    ENHANCED = 0
    STANDARD = 1
}

class NodeType(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    RELU = 0
    SOFTMAX = 1
    SIGMOID = 2
    TANH = 3
    SOFTPLUS = 4
    SOFTSIGN = 5
    SELU = 6
    ELU = 7
    EXPONENTIAL = 8

    def toKerasName(self){
        if self == NodeType.RELU{
            return 'relu'
        }elif self == NodeType.SOFTMAX{
            return 'softmax'
        }elif self == NodeType.SIGMOID{
            return 'sigmoid'
        }elif self == NodeType.TANH{
            return 'tanh'
        }elif self == NodeType.SOFTPLUS{
            return 'softplus'
        }elif self == NodeType.SOFTSIGN{
            return 'softsign'
        }elif self == NodeType.SELU{
            return 'selu'
        }elif self == NodeType.ELU{
            return 'elu'
        }elif self == NodeType.EXPONENTIAL{
            return 'exponential'
        }
        return None
    }
}

class GeneticRankType(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    RELATIVE = 0
    ABSOLUTE = 1
    INCREMENTAL = 1
}