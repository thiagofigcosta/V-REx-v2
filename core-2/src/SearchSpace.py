#!/bin/python
# -*- coding: utf-8 -*-
 
from enum import Enum

class SearchSpace(object){
    
    class Type(Enum){
        INT=0
        FLOAT=1
    }

    class Dimension(object){
        # 'Just':'to fix vscode coloring':'when using pytho{\}'

        def __init__(self,min_value,max_value,data_type){
            self.data_type=data_type
            self.min_value=min_value
            self.max_value=max_value
            if self.data_type==SearchSpace.Type.INT{
                self.min_value=int(self.min_value)
                self.max_value=int(self.max_value)
            }elif self.data_type==SearchSpace.Type.FLOAT{
                self.min_value=float(self.min_value)
                self.max_value=float(self.max_value)
            }
        }
    }

    def __init__(self){
        self.search_space={}
    }

    def add(self,name,min_value,max_value,data_type){
        self.search_space[name]=SearchSpace.Dimension(min_value,max_value,data_type)
    }
    
}
