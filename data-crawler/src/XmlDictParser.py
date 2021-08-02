#!/bin/python
# -*- coding: utf-8 -*-

from xml.etree import cElementTree as ElementTree
import re 
from lxml import etree
import xmlschema
from Utils import Utils

class XmlDictParser(dict){ 

    def __init__(self, parent_element,filter=False){
        if parent_element.items(){
            self.updateShim( dict(parent_element.items()) )
        }
        for element in parent_element{
            if len(element){
                aDict = XmlDictParser(element)
                self.updateShim({element.tag: aDict})
            }elif element.items(){ 
                elementattrib= element.items()
                if element.text{           
                    elementattrib.append((element.tag,element.text ))
                }    
                self.updateShim({element.tag: dict(elementattrib)})
            }else{
                self.updateShim({element.tag: element.text})
            }
        }
        if filter{
            self.filterKeys()
        }
    }

    @classmethod
    def fromFile(cls,path,filter=False){
        tree = ElementTree.parse(path)
        root = tree.getroot()
        return cls(root,filter)
    }

    @classmethod
    def fromFileWithSchema2(cls,xml_path,schema_path,filter=False){
         with open(schema_path) as f{
            xmlschema_doc = etree.parse(f)
        }
        xmlschema = etree.XMLSchema(xmlschema_doc)
        tree = etree.parse(xml_path)
        root = tree.getroot()
        return cls(root,filter)
    }

    # TODO cannot parse fields because cannot handle mixed="true" data from parent
    @staticmethod
    def fromFileWithSchema(xml_path,schema_path){
        schema = xmlschema.XMLSchema(schema_path)
        return schema.to_dict(xml_path) 
    }

    @classmethod
    def fromString(cls,string,filter=False){
        root = ElementTree.XML(string)
        return cls(root,filter)
    }

    def filterKeys(self){
        self=XmlDictParser.staticFilterKeys(self)
    }

    @staticmethod
    def staticFilterKeys(dictionary){
        to_replace=[]
        for k,v in dictionary.items(){
            if type(v) in (dict,XmlDictParser,list){
                if type(v) is list {
                    for i in range(len(v)){
                        if type(v[i]) in (dict,XmlDictParser){
                            v[i]=XmlDictParser.staticFilterKeys(v[i])
                        }
                    }
                }else{
                    dictionary[k]=XmlDictParser.staticFilterKeys(v)
                }
            }
            if re.search(r'{.*}.+', str(k)){
                k_new=re.sub(r'{.*?}','',str(k))
                to_replace.append((k_new,k))               
            }
        }
        for pair in to_replace{
            dictionary[pair[0]]=dictionary.pop(pair[1])
        }
        return dictionary
    }

    @staticmethod
    def recursiveRemoveKey(dictionary,key_name){
        to_remove=[]
        for k,v in dictionary.items(){
            if k==key_name{
                to_remove.append(k)
            }elif type(v) in (dict,XmlDictParser,list){
                if type(v) is list {
                    for i in range(len(v)){
                        if type(v[i]) in (dict,XmlDictParser){
                            v[i]=XmlDictParser.recursiveRemoveKey(v[i],key_name)
                        }
                    }
                }else{
                    dictionary[k]=XmlDictParser.recursiveRemoveKey(v,key_name)
                }
            }
        }
        for k in to_remove{
            dictionary.pop(k)
        }
        return dictionary
    }

    @staticmethod
    def recursiveRemoveEmpty(dictionary){
        to_remove=[]
        for k,v in dictionary.items(){
            if type(v) in (dict,XmlDictParser,list){
                if type(v) is list {
                    to_remove_el_list=[]
                    for i in range(len(v)){
                        if type(v[i]) in (dict,XmlDictParser){
                            v[i]=XmlDictParser.recursiveRemoveEmpty(v[i])
                            if len(v[i])==0{
                                to_remove_el_list.append(v[i])
                            }
                        }
                    }
                    for el in to_remove_el_list{
                        dictionary[k].remove(el)
                    }
                    if len(v)==0{
                        to_remove.append(k)
                    }
                }else{
                    dictionary[k]=XmlDictParser.recursiveRemoveEmpty(v)
                    if (len(dictionary[k]))==0{
                        to_remove.append(k)
                    }
                }
            }
        }
        for k in to_remove{
            dictionary.pop(k)
        }
        return dictionary
    }

    @staticmethod
    def compressDictOnFollowingKeys(dictionary,keys,father=None){
        default_key_name='value'
        for k in list(dictionary){
            v=dictionary[k]
            if type(v) in (dict,XmlDictParser,list){
                if type(v) is list {
                    for i in range(len(v)){
                        if type(v[i]) in (dict,XmlDictParser){
                            v[i]=XmlDictParser.compressDictOnFollowingKeys(v[i],keys,v)
                        }
                    }
                }else{
                    dictionary[k]=XmlDictParser.compressDictOnFollowingKeys(v,keys,dictionary)
                }

                if type(v) in (dict,XmlDictParser){
                    compressed=[]
                    other_type=False
                    for k2 in list(v){
                        v2=v[k2]
                        if k2 in keys{
                            if type(v2) is str {
                                compressed.append(v2)
                                v.pop(k2)
                            }elif type(v2) is list and all(type(v3) is str for v3 in v2){
                                compressed=compressed+v2
                                v.pop(k2)
                            }elif type(v2) in (dict,XmlDictParser,list) and len(v2)==0 {
                                v.pop(k2)
                            }else{
                                other_type=True
                            }
                        }
                    }
                    if len(compressed)>1{
                        if not other_type and father{
                            if len(father)==1{
                                father=compressed
                            }else{
                                if type(father) is list{ 
                                    father.append(compressed)
                                }else{
                                    father[default_key_name]=compressed
                                }
                                
                            }
                        }else{
                            v[default_key_name]=compressed
                        }
                    }
                }
            }
        }
        return dictionary
    }

    @staticmethod
    def stringfyDict(dictionary){
        for k in list(dictionary){
            v=dictionary[k]
            dictionary.pop(k)
            dictionary[str(k)]=v
            if type(v) in (dict,XmlDictParser,list){
                if type(v) is list {
                    for i in range(len(v)){
                        if type(v[i]) in (dict,XmlDictParser){
                            v[i]=XmlDictParser.stringfyDict(v[i])
                        }
                    }
                }else{
                    dictionary[k]=XmlDictParser.stringfyDict(v)
                }
            }else{
                dictionary[k]=str(v)
            }
        }
        return dictionary
    }

    def updateShim (self, aDict){
        for key in aDict.keys(){   # keys() includes tag and attributes
            if key in self{
                value = self.pop(key)
                if type(value) is not list{
                    listOfDicts = []
                    listOfDicts.append(value)
                    listOfDicts.append(aDict[key])
                    self.update({key: listOfDicts})
                }else{
                    value.append(aDict[key])
                    self.update({key: value})
                }
            }else{
                dic={key:aDict[key]}
                self.update(dic)
            }
        }
    }
}
