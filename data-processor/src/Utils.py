#!/bin/python
# -*- coding: utf-8 -*-

import shutil
import re
import os
import codecs
import zipfile
from datetime import datetime
import gzip
import json
import sys
from pympler.asizeof import asizeof
import random as rd
from bson.json_util import dumps as bdumps
from bson.json_util import loads as bloads

class Utils(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    if os.name == 'nt'{
        FILE_SEPARATOR='\\'
    }else{
        FILE_SEPARATOR='/'
    }
    TMP_FOLDER=None
    LOGGER=None

    def __init__(self,tmp_folder,logger){
        Utils.LOGGER=logger
		Utils.TMP_FOLDER=tmp_folder
    }

    @staticmethod
    def createFolderIfNotExists(path){
        if not os.path.exists(path){
            os.makedirs(path, exist_ok=True)
        }
    }

    @staticmethod
    def checkIfPathExists(path){
        return os.path.exists(path)
    }

    @staticmethod
    def deletePath(path){
        if os.path.isdir(path){
            Utils.deleteFolder(path)
        }elif os.path.isfile(path){
            Utils.deleteFile(path)
        }else{
            Utils.LOGGER.fatal('File {} is a special file.'.format(path))
        }
    }

    @staticmethod
    def deleteFile(path){
        if os.path.exists(path){
            os.remove(path)
        }else{
            Utils.LOGGER.warn('The file {} does not exist.'.format(path))
        }
    }

    @staticmethod
    def deleteFolder(path){
        if os.path.exists(path){
            shutil.rmtree(path)
        }else{
            Utils.LOGGER.warn('The folder {} does not exist.'.format(path))
        }
    }

    @staticmethod
    def saveJson(path,data,pretty=True,use_bson=False){
        if use_bson{
            Utils.saveFile(path,bdumps(data))
            return
        }
        with open(path, 'w') as fp{
            json.dump(data, fp, indent=3 if pretty else None)
        }
    }

    @staticmethod
    def changeStrDateFormat(date,input_format,output_format){
        date=datetime.strptime(date, input_format)
        return date.strftime(output_format)
    }

    @staticmethod
    def loadJson(path,use_bson=False){
        if use_bson {
            data=Utils.openFile(path)
            data=bloads(data)
        }else{
            with open(path, 'r') as fp {
                data=json.load(fp)
            }
        }
        return data
    }

    @staticmethod
    def getTmpFolder(base_name,random=False){
        if random{
            destination_folder=Utils.joinPath(Utils.TMP_FOLDER,base_name+str(rd.randint(0,65535)))
        }else{
            destination_folder=Utils.joinPath(Utils.TMP_FOLDER,base_name)
        }
        Utils.createFolderIfNotExists(destination_folder)
        return destination_folder
    }

    @staticmethod
    def getTodayDate(date_format='%d/%m/%Y'){
        return datetime.now().strftime(date_format)
    }

    @staticmethod
    def getIndexOfDictList(docs,key,value){
        for i in range(len(docs)){
            if docs[i][key]==value{
                return i
            }
        }
    }

    @staticmethod
    def countCSVColumns(path,delimiter=','){
        max_columns=0
        with codecs.open(path, 'r', 'utf-8',errors='ignore') as file{
            for line in file{
                line = re.sub(r'".*?"','',line)
                count=len(line.split(delimiter))
                if count>max_columns{
                    max_columns=count
                }
            }
        }
        return max_columns
    }

    @staticmethod
    def filenameFromPath(path,get_extension=False){
        if get_extension {
            re_result=re.search(r'.+\/(.+)', path)
            return re_result.group(1) if re_result is not None else path
        }else{
            re_result=re.search(r'.+\/(.+)\..+', path)
            return re_result.group(1) if re_result is not None else path
        }
    } 

    @staticmethod
    def removeExtFromFilename(filename){
        re_result=re.search(r'(.*)\..*', filename)
        return re_result.group(1) if re_result is not None else filename
    } 

    @staticmethod
    def parentFromPath(path){
        re_result=re.search(r'(.+\/).+', path)
        return re_result.group(1) if re_result is not None else path
    } 

    @staticmethod
    def unzip(path,destination_folder=None,delete=True){
        Utils.LOGGER.info('Unziping file {}...'.format(path))
        if not destination_folder{
            destination_folder=Utils.joinPath(Utils.TMP_FOLDER,Utils.filenameFromPath(path))
        }else{
            destination_folder=Utils.joinPath(Utils.TMP_FOLDER,destination_folder)
        }
        Utils.createFolderIfNotExists(destination_folder)
        with zipfile.ZipFile(path, 'r') as zip_ref{
            zip_ref.extractall(destination_folder)
        }
        if delete{
            Utils.deleteFile(path)
        }
        Utils.LOGGER.info('Unziped file {}...OK'.format(path))
        return destination_folder
    }

    @staticmethod
    def zip(path_to_zip,compressed_output_file,base=None){
        Utils.LOGGER.info('Ziping file {}...'.format(path_to_zip))
        zipf=zipfile.ZipFile(compressed_output_file, 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(path_to_zip){
            for file in files{
                file_path=os.path.join(root, file)
                if base {
                    re_result=re.search(r'('+base+r'.*)', file_path)
                    file_path_on_zip=re_result.group(1) if re_result is not None else file_path
                    zipf.write(file_path,file_path_on_zip)
                }else{
                    zipf.write(file_path)
                }
            }
        }
        zipf.close()
        Utils.LOGGER.info('Ziped file {}...OK'.format(path_to_zip))
        return compressed_output_file
    }

    @staticmethod
    def gunzip(path,extension){
        Utils.LOGGER.info('Gunziping file {}...'.format(path))
        destination_folder=Utils.joinPath(Utils.TMP_FOLDER,Utils.filenameFromPath(path))
        Utils.createFolderIfNotExists(destination_folder)        
        destination_filename=Utils.filenameFromPath(path)+extension
        block_size=65536
        with gzip.open(path, 'rb') as s_file{
            destination_path=Utils.joinPath(destination_folder,destination_filename)
            with open(destination_path, 'wb') as d_file{
                while True{
                    block = s_file.read(block_size)
                    if not block{
                        break
                    }else{
                        d_file.write(block)
                    }
                }
            }
        }
        Utils.deleteFile(path)
        Utils.LOGGER.info('Gunziped file {}...OK'.format(path))
        return destination_folder
    }

    @staticmethod
    def openFile(path){
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as file{
            return file.read()
        }
    }

    @staticmethod
    def saveFile(path,content){
        with codecs.open(path, 'w', 'utf-8') as file{
            file.write(content)
        }
    }

    @staticmethod
    def bytesToHumanReadable(num, suffix='B'){
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']{
            if abs(num) < 1024.0{
                return "%3.1f%s%s" % (num, unit, suffix)
            }
            num /= 1024.0
        }
        return "%.1f%s%s" % (num, 'Yi', suffix)
    }

    @staticmethod
    def sizeof(obj,to_human_readable=False){
        size=asizeof(obj)
        if to_human_readable{
            size=Utils.bytesToHumanReadable(size)
        }
        return size
    }   

    @staticmethod
    def appendToStrIfDoesNotEndsWith(base,suffix){
        if not base.endswith(suffix){
            return base+suffix
        }
        return base
    }

    @staticmethod
    def joinPath(parent,child){
        parent=Utils.appendToStrIfDoesNotEndsWith(parent,Utils.FILE_SEPARATOR)
        return parent+child
    }

    @staticmethod
    def isFirstStrDateOldest(date1,date2,date_format){
        date1=datetime.strptime(date1, date_format)
        date2=datetime.strptime(date2, date_format)
        return date1<date2
    }

    @staticmethod
    def daysBetweenStrDate(date1,date2,input_format){
        date1=datetime.strptime(date1, input_format)
        date2=datetime.strptime(date2, input_format)
        delta=date2-date1
        return delta.days
    }

    @staticmethod
    def binarySearch(lis,el){ # list must be sorted
        low=0
        high=len(lis)-1
        ret=None 
        while low<=high{
            mid=(low+high)//2
            if el<lis[mid]{
                high=mid-1
            }elif el>lis[mid]{
                low=mid+1
            }else{
                ret=mid
                break
            }
        }
        return ret
    }

    @staticmethod
    def runningOnDockerContainer(){
        path = '/proc/self/cgroup'
        return (
            os.path.exists('/.dockerenv') or
            os.path.isfile(path) and any('docker' in line for line in open(path))
        )
    }
}