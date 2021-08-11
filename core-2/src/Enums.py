#!/bin/python
# -*- coding: utf-8 -*-

from enum import Enum
 

class CrossValidation(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    NONE = 0
    ROLLING_FORECASTING_ORIGIN = 1
    KFOLDS = 2
    FIXED_PERCENT = 3
}

class Metric(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    RAW_LOSS = 0
    F1 = 1
    RECALL = 2
    ACCURACY = 3
    PRECISION = 4

    def toKerasName(self){
        if self == Metric.RAW_LOSS{
            return 'loss'
        }elif self == Metric.F1{
            return 'f1_score'
        }elif self == Metric.RECALL{
            return 'recall'
        }elif self == Metric.ACCURACY{
            return 'accuracy'
        }elif self == Metric.PRECISION{
            return 'precision'
        }
        return None
    }
}

class LabelEncoding(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    INCREMENTAL = 0
    BINARY = 1
    # yes, I will jump number 2
    BINARY_PLUS_ONE = 3
    SPARSE = 4
    DISTINCT_SPARSE = 5
    DISTINCT_SPARSE_PLUS_ONE = 6
    INCREMENTAL_PLUS_ONE = 7
    EXPONENTIAL = 8
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
    LINEAR = 9

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
        }elif self == NodeType.LINEAR{
            return 'linear'
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


class Loss(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    BINARY_CROSSENTROPY = 0
    CATEGORICAL_CROSSENTROPY = 1
    MEAN_SQUARED_ERROR = 2
    MEAN_ABSOLUTE_ERROR = 3

    def toKerasName(self){
        if self == Loss.BINARY_CROSSENTROPY{
            return 'binary_crossentropy'
        }elif self == Loss.CATEGORICAL_CROSSENTROPY{
            return 'categorical_crossentropy'
        }elif self == Loss.MEAN_SQUARED_ERROR{
            return 'mean_squared_error'
        }elif self == Loss.MEAN_ABSOLUTE_ERROR{
            return 'mean_absolute_error'
        }
        return None
    }
}