#!/bin/python

import math
import numpy as np
from Utils import Utils
from Enums import LabelEncoding

class Dataset(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    DATASET_PATH='.'

    @staticmethod
    def getDataset(path){
        return Utils.joinPath(Dataset.DATASET_PATH,path)
    }

    @staticmethod
    def readLabeledCsvDataset(path,separator=','){
        labels=[]
        features=[]
        with open(path) as f{
            for line in f{
                if line {
                    splitted=line.split(separator)
                    features.append([float(el) for el in splitted[0:-1]])
                    labels.append(splitted[-1].strip())
                }
            }
        }
        return features, labels
    }

    @staticmethod
    def enumfyDatasetLabels(labels){
        label_set=set()
        for el in labels{
            label_set.add(el)
        }
        label_equivalence={}
        for i,label in enumerate(label_set){
            label_equivalence[label]=i
            label_equivalence[repr(i)]=label
        }
        labels=[label_equivalence[el] for el in labels]
        return labels, label_equivalence
    }

    @staticmethod
    def encodeDatasetLabels(labels,enc,ext_max=0){
        max_value=float('-inf')
        if ext_max>0{
            max_value=ext_max
        }
        label_set=set()
        for label in labels{
            label_set.add(label)
            max_value=max(max_value,label)
        }
        output_neurons=0
        if enc in (LabelEncoding.BINARY,LabelEncoding.BINARY_PLUS_ONE){
            output_neurons=int(math.ceil(math.log2(max_value+1)))
        }elif enc in (LabelEncoding.SPARSE,LabelEncoding.DISTINCT_SPARSE,LabelEncoding.DISTINCT_SPARSE_PLUS_ONE){
            output_neurons=int(max_value+1)
        }elif enc in (LabelEncoding.INCREMENTAL,LabelEncoding.INCREMENTAL_PLUS_ONE,LabelEncoding.EXPONENTIAL){
            output_neurons=1
        }else{
            raise Exception('Unknown LabelEnconding {}'.format(enc))
        }
        label_equivalence={}
        for label in label_set{
            new_label=[]
            for i in range(output_neurons){
                if enc == LabelEncoding.BINARY{
                    new_label.append((label >> i) & 1)
                }elif enc == LabelEncoding.SPARSE{
                    new_label.append(1 if i==label else 0)
                }elif enc == LabelEncoding.DISTINCT_SPARSE{
                    new_label.append(label if i==label else 0)
                }elif enc == LabelEncoding.DISTINCT_SPARSE_PLUS_ONE{
                    new_label.append((label+1) if i==label else 0)
                }elif enc == LabelEncoding.INCREMENTAL{
                    new_label.append(label)
                }elif enc == LabelEncoding.INCREMENTAL_PLUS_ONE{
                    new_label.append(label+1)
                }elif enc == LabelEncoding.EXPONENTIAL{
                    new_label.append(math.pow(2,label+1))
                }elif enc == LabelEncoding.BINARY_PLUS_ONE{
                    new_label.append(((label+1) >> i) & 1)
                }else{
                    raise Exception('Unknown LabelEnconding {}'.format(enc))
                }
            }
            label_equivalence[label]=new_label
            label_equivalence[repr(new_label)]=label
        }
        labels=[label_equivalence[el] for el in labels]
        return labels, label_equivalence
    }

    @staticmethod
    def translateLabelFromOutput(label,first_equivalence,second_equivalence=None){
        label=Dataset.labelToVanilla(label)
        if second_equivalence is None{
            if repr(label) not in first_equivalence{
                return [0]*len(first_equivalence.values()[0])
            }
            return first_equivalence[repr(label)]
        }else{
            if repr(label) not in second_equivalence{
                return 'Abscent'
            }
            return first_equivalence[repr(second_equivalence[repr(label)])]
        }
    }

    @staticmethod
    def labelToVanilla(label){
        if type(label) is not list{
            label=[int(i) for i in label]
        }
        return label
    }

    @staticmethod
    def normalizeDatasetFeatures(features){
        values_min=[float('inf') for _ in range(len(features[0]))]
        values_max=[float('-inf') for _ in range(len(features[0]))]
        for feature in features{
            for i,f_component in enumerate(feature){
                values_min[i]=min(values_min[i],f_component)
                values_max[i]=max(values_max[i],f_component)
            }
        }
        scale=[]
        for i in range(len(values_min)){
            scale.append((values_min[i],(values_max[i]-values_min[i])))
        }
        features_normalized=[]
        for feature in features{
            feature_normalized=[]
            for i,f_component in enumerate(feature){
                feature_normalized.append((f_component-scale[i][0])/scale[i][1])
            }
            features_normalized.append(feature_normalized)
        }
        return features_normalized,scale
    }

    @staticmethod
    def shuffleDataset(features,labels){
        indexes=list(range(len(features)))
        indexes=Utils.shuffle(indexes)
        random_features=[]
        random_labels=[]
        for idx in indexes{
            random_features.append(features[idx])
            random_labels.append(labels[idx])
        }
        return random_features,random_labels
    }

    @staticmethod
    def splitDataset(features,labels,percentage){
        firstSize=int(len(features)*percentage)
        return (features[:firstSize],labels[:firstSize]),(features[firstSize:],labels[firstSize:])
    }

    @staticmethod
    def statisticalAnalysis(predictions,corrects){
        size = len(predictions)
        if(len(corrects)<size){
            size=len(corrects)
        }
        total=0.0
        hits=0.0
        true_negative=0
        true_positive=0
        false_negative=0
        false_positive=0
        wrong=0
        pos_count=0
        for i in range(size){
            cur_pred=Dataset.labelToVanilla(predictions[i])
            cur_correct=corrects[i]
            equal=True
            positive=True
            for j in range(len(cur_pred)){
                if(cur_pred[j]!=cur_correct[j]){
                    equal=False
                }
                if(cur_correct[j]==0){
                    positive=False
                }
            }
            total+=1
            if(equal){
                hits+=1
                if(positive){
                    true_positive+=1
                    pos_count+=1
                }else{
                    true_negative+=1
                }
            }else{
                wrong+=1
                if(positive){
                    false_negative+=1
                    pos_count+=1
                }else{
                    false_positive+=1
                }
            }
        }
        stats={}
        stats['accuracy']=float(hits)/float(total)
        if (len(corrects[0])==1 and pos_count>0){
            stats['precision']=float(true_positive)/float((true_positive+false_positive))
            stats['recall']=float(true_positive)/float((true_positive+false_negative))
            stats['f1']=2*(stats['precision']*stats['recall'])/(stats['precision']+stats['recall'])
        }
        return stats
    }

    @staticmethod
    def filterDataset(features,labels,only){
        features_out=[]
        labels_out=[]
        value=None
        for i in range(len(features)){
            if labels[i]==only{
                features_out.append(features[i])
                labels_out.append(labels[i])
                value=labels[i]
            }
        }
        if type(value) is str{
            abscent_value='Abscent'
        }elif type(value) in (int,float){
            if 0==value{
                abscent_value=1
            }else{
                abscent_value=0
            }
        }else{
            abscent_value=[0]*len(value)
            if abscent_value==value{
                abscent_value=[1]*len(value)
            }
        }
        # abscent_value=None
        for i in range(len(features)){
            if labels[i]!=only{
                if abscent_value is None{
                    abscent_value=labels[i]
                }
                features_out.append(features[i])
                labels_out.append(abscent_value)
            }
        }
        return features_out,labels_out
    }

    @staticmethod
    def balanceDataset(features,labels){
        features_pos=[]
        labels_pos=[]
        features_neg=[]
        labels_neg=[]
        for i in range(len(features)){
            if int(labels[i][0])==0{
                features_neg.append(features[i])
                labels_neg.append(labels[i])
            }else{
                features_pos.append(features[i])
                labels_pos.append(labels[i])
            }
        }
        if (len(labels_pos)>len(labels_neg)){
            features_pos=features_pos[:len(labels_neg)]
            labels_pos=labels_pos[:len(labels_neg)]
            features_pos+=features_neg
            labels_pos+=labels_neg
            return Dataset.shuffleDataset(features_pos,labels_pos)
        } elif (len(labels_pos)<len(labels_neg)){
            features_neg=features_neg[:len(labels_pos)]
            labels_neg=labels_neg[:len(labels_pos)]
            features_neg+=features_pos
            labels_neg+=labels_pos
            return Dataset.shuffleDataset(features_neg,labels_neg)
        }else{
            return features,labels
        }
    }
}