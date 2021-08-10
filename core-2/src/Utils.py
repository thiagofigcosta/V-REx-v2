#!/bin/python
# -*- coding: utf-8 -*-

import shutil
import re
import os
import time
import codecs
import zipfile
from datetime import datetime
import datetime as dt
import joblib
import gzip
import json
import sys
import uuid
from pympler.asizeof import asizeof
import random as rd
from bson.json_util import dumps as bdumps
from bson.json_util import loads as bloads
from numpy.random import Generator, MT19937

class Utils(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    
    if os.name == 'nt'{
        FILE_SEPARATOR='\\'
    }else{
        FILE_SEPARATOR='/'
    }
    RESOURCES_FOLDER='res'
    DATE_FORMAT='%d/%m/%Y'
    DATETIME_FORMAT='%d/%m/%Y %H:%M:%S'
	FIRST_DATE='01/01/1970'
    TMP_FOLDER=None
    LOGGER=None
    RNG=Generator(MT19937(int(time.time()*rd.random())))

    def __init__(self,tmp_folder,logger){
        Utils.LOGGER=logger
		Utils.TMP_FOLDER=tmp_folder
        Utils.createFolderIfNotExists(Utils.RESOURCES_FOLDER)
    }

    @staticmethod
    def shuffle(list_to_shuffle){
        list_to_shuffle=list_to_shuffle.copy()
        Utils.RNG.shuffle(list_to_shuffle)
        return list_to_shuffle
    }

    @staticmethod
    def random(){
        return Utils.RNG.random()
    }

    @staticmethod
    def randomInt(min_inclusive,max_inclusive,size=1){
        out=Utils.RNG.integers(low=min_inclusive, high=max_inclusive+1, size=size)
        if size == 1{
            out=out[0]
        } 
        return out
    }

    @staticmethod
    def randomFloat(min_value,max_value,size=1){
        out=[]
        for _ in range(size){
            out.append(min_value + (Utils.random() * (max_value - min_value)))
        }
        if size == 1{
            out=out[0]
        } 
        return out
    }

    @staticmethod
    def randomUUID(){
        return uuid.uuid4().hex
    }

    @staticmethod
    def createFolderIfNotExists(path){
        if not os.path.exists(path){
            os.makedirs(path)
        }
    }

    @staticmethod
    def checkIfPathExists(path){
        return os.path.exists(path)
    }

    @staticmethod
    def getResource(filename){
        path=filename
		if Utils.appendToStrIfDoesNotEndsWith(Utils.RESOURCES_FOLDER,Utils.FILE_SEPARATOR) not in path{
			path=Utils.joinPath(Utils.RESOURCES_FOLDER,filename)
		}
		return path
    }

    @staticmethod
    def deletePath(path){
        if os.path.isdir(path){
            Utils.deleteFolder(path)
        }elif os.path.isfile(path) or os.path.islink(path){
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
                return '{:3.1f}{}{}'.format(num,unit,suffix)
            }
            num /= 1024.0
        }
        return '{:.1f}{}{}'.format(num,'Yi',suffix)
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
    def runningOnDockerContainer(){
        path = '/proc/self/cgroup'
        return (
            os.path.exists('/.dockerenv') or
            os.path.isfile(path) and any('docker' in line for line in open(path))
        )
    }

    @staticmethod
	def getPythonVersion(getTuple=False){
		version=sys.version_info
		version_tuple=(version.major,version.minor,version.micro)
		if getTuple{
			return version.major,version.minor,version.micro
		}else{
			return '.'.join([str(el) for el in version_tuple])
        }
    }
	
	@staticmethod
	def getPythonExecName(){
		version=Utils.getPythonVersion(getTuple=True)
		full_name='python{}.{}'.format(version[0],version[1])
		short_name='python{}'.format(version[0])
		default_name='python'
		if shutil.which(full_name) is not None{
			return full_name
        }
		if shutil.which(short_name) is not None{
			return short_name
        }
		return default_name
    }

	@staticmethod
	def moveFile(src_path,dst_path){
		os.replace(src_path, dst_path)
    }

    @staticmethod
	def copyFile(src_path,dst_path){
		shutil.copy(src_path, dst_path)
    }

	@staticmethod
	def createFolder(path){
		if not os.path.exists(path){
			os.makedirs(path, exist_ok=True)
        }
    }


	@staticmethod
	def getFolderPathsThatMatchesPattern(folder,pattern){
		paths=[]
		if os.path.exists(folder){
			for filename in os.listdir(folder){
				if re.match(pattern,filename){
					file_path = Utils.joinPath(folder,filename)
					paths.append(file_path)
                }
            }
        }
		return paths
    }

	@staticmethod
	def deleteFolderContents(folder){
		if os.path.exists(folder){
			for filename in os.listdir(folder){
				file_path = Utils.joinPath(folder,filename)
				Utils.deletePath(file_path)
            }
        }
    }

    @staticmethod
	def timestampByExtensive(timestamp,seconds=True){
        if seconds {
            timestamp_ms=timestamp*1000
        }else{
            timestamp_ms=timestamp
        }
		timestamp_ms=int(timestamp_ms)
		D=int(timestamp_ms/1000/60/60/24)
		H=int(timestamp_ms/1000/60/60%24)
		M=int(timestamp_ms/1000/60%60)
		S=int(timestamp_ms/1000%60)
		MS=int(timestamp_ms%1000)
		out='' if timestamp_ms > 0 else 'FINISHED'
		if D > 0{
			out+='{} days '.format(D)
        }
		if D > 0 and MS == 0 and S == 0 and M == 0 and H > 0{
			out+='and '
        }
		if H > 0{
			out+='{} hours '.format(H)
        }
		if (D > 0 or H > 0) and MS == 0 and S == 0 and M > 0{
			out+='and '
        }
		if M > 0{
			out+='{} minutes '.format(M)
        }
		if (D > 0 or H > 0 or M > 0) and MS == 0 and S > 0{
			out+='and '
        }
		if S > 0{
			out+='{} seconds '.format(S)
        }
		if (D > 0 or H > 0 or M > 0 or S > 0) and MS > 0{
			out+='and '
        }
		if MS > 0{
			out+='{} milliseconds '.format(MS)
        }
		return out
    }

    @staticmethod
	def getNextNWorkDays(from_date, add_days){
		business_days_to_add = add_days
		current_date = from_date
		dates=[]
		while business_days_to_add > 0{
			current_date += dt.timedelta(days=1)
			weekday = current_date.weekday()
			if weekday >= 5{ # sunday = 6
				continue
            }
			business_days_to_add -= 1
			dates.append(current_date)
        }
		return dates
    }

	@staticmethod
	def getStrNextNWorkDays(from_date, add_days, date_format=DATE_FORMAT){
		from_date=datetime.strptime(from_date,date_format)
		dates=Utils.getNextNWorkDays(from_date,add_days)
		dates=[date.strftime(date_format) for date in dates]
		return dates
    }

    @staticmethod
	def saveObj(obj,path){
		joblib.dump(obj, path)
    }

	@staticmethod
	def loadObj(path){
		return joblib.load(path)
    }

    @staticmethod
	def printDict(dictionary,name=None,tabs=0,inline=False){
		start=''
		if name is not None{
			print('{}{}:'.format('\t'*tabs,name),end='' if inline else '\n')
			start=' | ' if inline else '\t'
        }
        first=True
		for key,value in dictionary.items(){
			print('{}{}{}: {}'.format('\t'*tabs,' ' if first and inline else start,key,value),end='' if inline else '\n')
            first=False
        }
        if inline{
            print()
        }
    }

    @staticmethod
    def getEnumBorder(enum,max_instead_of_min=False){
        if max_instead_of_min{
            return enum(list(enum._member_map_.items())[-1][1]).value
        }else{
            return enum(list(enum._member_map_.items())[0][1]).value
        }
    }
}