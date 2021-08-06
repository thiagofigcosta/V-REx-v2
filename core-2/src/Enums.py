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

class GeneticAlgorithm(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    ENHANCED = 0
    STANDARD = 1
}

class NodeType(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    RELU = 0
    SOFTMAX = 1
    SIGMOID = 2
}

class StdGeneticRankType(Enum){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    RELATIVE = 0
    ABSOLUTE = 1
}