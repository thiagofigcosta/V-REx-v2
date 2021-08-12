#!/bin/python
# -*- coding: utf-8 -*-
 
from enum import Enum
from Utils import Utils

class SearchSpace(object){
    
    class Type(Enum){
        INT=0
        FLOAT=1
        BOOLEAN=2
    }

    class Dimension(object){
        # 'Just':'to fix vscode coloring':'when using pytho{\}'

        def __init__(self,data_type,min_value=0,max_value=1,name=''){
            if isinstance(min_value,Enum){
                min_value=min_value.value
            }
            if isinstance(max_value,Enum){
                max_value=max_value.value
            }
            if type(min_value)!=bool and (min_value > max_value){
                raise Exception('Incorrect limits, swap min and max')
            }
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
            }elif self.data_type==SearchSpace.Type.BOOLEAN{
                self.min_value=False if self.min_value in (0,False) else True
                self.max_value=False if self.max_value in (0,False) else True
            }else{
                raise Exception('Unhandled data type')
            }
        }

        def fixValue(self,value){
            if (self.data_type==SearchSpace.Type.BOOLEAN and type(value) is bool and value in (self.min_value,self.max_value)){
                return value
            }elif (self.data_type==SearchSpace.Type.BOOLEAN) {
                if Utils.random() >.5{
                    return self.min_value
                }else{
                    return self.max_value
                }
            }else{
                if value > self.max_value or value < self.min_value {
                    value_abs=abs(value)
                    if value_abs <= self.max_value and value_abs >= self.min_value {
                        return value_abs
                    }
                    if value > self.max_value {
                        return self.max_value
                    }
                    if value < self.min_value {
                        return self.min_value
                    }
                }
            }
            return value
        }

        def copy(self){
            that=SearchSpace.Dimension(self.data_type,self.min_value,self.max_value,self.name)
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
        if type(i) is int {
            return self.search_space[i]
        }elif type(i) is str{
            for el in self.search_space{
                if el.name==i{
                    return el
                }
            }
        }
        return None
    }

    def __iter__(self){
       return SearchSpaceIterator(self)
    }

    def __str__(self){
       str_out='Search Space: { \n'
       for dim in self.search_space{
           str_out+='\t{}: [ min: {}, max: {} | {} ] \n'.format(dim.name,dim.min_value,dim.max_value,dim.data_type)
       }
       str_out+='}'
       return str_out
    }

    def add(self,min_value,max_value,data_type,name=''){
        self.search_space.append(SearchSpace.Dimension(data_type,min_value,max_value,name))
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
