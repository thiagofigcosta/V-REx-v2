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

        def __init__(self,min_value,max_value,data_type,name=''){
            self.name=name
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

        def fixValue(self,value){
            if value > self.max_value {
                return self.max_value
            }
            if value < self.min_value {
                return self.min_value
            }
            return value
        }

        def copy(self){
            that=SearchSpace.Dimension(self.min_value,self.max_value,self.data_type,self.name)
            return that
        }
    }

    def __init__(self){
        self.search_space=[]
    }

    def __len__(self){
        return len(self.search_space)
    }

    def __getitem__(self, i){
        return self.search_space[i]
    }

    def __iter__(self){
       return SearchSpaceIterator(self)
    }

    def add(self,min_value,max_value,data_type,name=''){
        self.search_space.append(SearchSpace.Dimension(min_value,max_value,data_type,name))
    }

    def get(self,i){
        return self.search_space[i]
    }
    
    def copy(self){
        that=SearchSpace()
        for dimension in self.search_space {
            that.search_space.append(dimension.copy())
        } 
        return that
    }
}

class SearchSpaceIterator{
   def __init__(self,search_space){
       self._search_space=search_space
       self._index=0
   }

   def __next__(self){
       if self._index < len(self._search_space){
           result=self._search_space.get(self._index)
           self._index+=1
           return result
       }
       raise StopIteration
   }
}
