#!/bin/python
# -*- coding: utf-8 -*-
from Utils import Utils
import pymongo
import datetime
import socket
import os
import time
from pymongo import MongoClient
from MongoQueue import MongoQueue,MongoJob,MongoLock

class MongoDB(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    QUEUE_DB_NAME='queue'
    QUEUE_COL_CRAWLER_NAME='crawler'
    QUEUE_COL_PROCESSOR_NAME='processor'
    RAW_DATA_DB_NAME='raw_data'
    PROCESSED_DATA_DB_NAME='processed_data'
    DUMMY_FOLDER='tmp/crawler/DummyMongo/'
    QUEUE_TIMEOUT_WITHOUT_PROGRESS=1500

	def __init__(self, address, port=27017, logger=None, user=None,password=None){
        if logger is None{
            raise Exception('MongoDB logger cannot be None')
        }
        self.logger=logger
        if address is None and port is None{
            self.dummy=True
            Utils.createFolderIfNotExists(MongoDB.DUMMY_FOLDER)
        }else{
            self.dummy=False
            self.address=address
            self.port=port
            self.user=user 
            self.password=password
            self.logger.info('Preparing mongo on {}:{}...'.format(address,port))
            self.establishConnection()
            self.startMongoDatabases()
            self.logger.info('Prepared mongo on {}:{}...OK'.format(address,port))
        }
    }

    @classmethod
    def asDummy(cls,logger){
        return cls(address=None,port=None,logger=logger)
    }

    def establishConnection(self){
        client = MongoClient(self.address, self.port,username=self.user, password=self.password)
        self.client=client
    }

    def saveReferences(self,refs){
        refs=refs.copy()
        loaded=self.loadReferences()
        if not self.dummy{
            lock=self.getLock(self.getRawDB(),'References',lease=150)
            while self.checkIfCollectionIsLocked(lock=lock){
                time.sleep(1)
            }
            lock.acquire()
        }
        for k,_ in refs.items(){
            refs[k]=list(refs[k].union(loaded[k]))
            refs[k].sort()
        }
        if self.dummy{
            ref_path=Utils.joinPath(MongoDB.DUMMY_FOLDER,'references.json')
            Utils.saveJson(ref_path,refs) 
        }else{
            refs={'unique':'unique','refs':refs}
            self.insertOneOnDB(self.getRawDB(),refs,'References',index='unique',ignore_lock=True)
            lock.release()
        }
    }

    def loadReferences(self){
        if self.dummy{
            ref_path=Utils.joinPath(MongoDB.DUMMY_FOLDER,'references.json')
            found= 1 if Utils.checkIfPathExists(ref_path) else 0
        }else{
            query={'unique':'unique'}
            refs=self.findOneOnDB(self.getRawDB(),'References',query,wait_unlock=True)
            found=1 if refs else 0
        }
        if found>0{
            if self.dummy{
                refs=Utils.loadJson(ref_path)
            }else{
                refs=refs['refs']
            }
            for k,_ in refs.items(){
                refs[k]=set(refs[k])
            }
        }else{
            refs={'cve':set(),'cwe':set(),'exploit':set(),'capec':set(),'oval':set()}
        }
        return refs
    }

    def startMongoDatabases(self){
        dblist = self.client.list_database_names()
        if MongoDB.RAW_DATA_DB_NAME not in dblist{
            self.logger.warn('Database {} does not exists, creating it...'.format(MongoDB.RAW_DATA_DB_NAME))
        }
        self.raw_db = self.client[MongoDB.RAW_DATA_DB_NAME]
        if MongoDB.PROCESSED_DATA_DB_NAME not in dblist{
            self.logger.warn('Database {} does not exists, creating it...'.format(MongoDB.PROCESSED_DATA_DB_NAME))
        }
        self.processed_db = self.client[MongoDB.PROCESSED_DATA_DB_NAME]
    }

    def getDB(self,name){
        return self.client[name]
    }

    def getRawDB(self){
        return self.raw_db
    }

    def getProcessedDB(self){
        return self.processed_db
    }

    def insertOneOnDB(self,db,document,collection,index=None,verbose=True,ignore_lock=False){
        return self.insertManyOnDB(db,[document],collection,index=index,verbose=verbose,ignore_lock=ignore_lock)
    }

    def insertManyOnDB(self,db,documents,collection_str,index=None,verbose=True,ignore_lock=False){
        if self.dummy {
            path=Utils.joinPath(MongoDB.DUMMY_FOLDER,'{}.json'.format(collection_str))
            Utils.saveJson(path,documents)
            return path
        }else{
            if verbose{
                self.logger.info('Inserting on {} db on col: {} with index: {} and size: {}...'.format(db.name,collection_str,index,len(documents)))
            }
            collection=db[collection_str]
            lock=self.getLock(db,collection_str)
            MAX_TRIES=3
            SECONDS_BETWEEN_TRIES=60
            tries=0
            trying=True
            while trying{
                try{
                    if not ignore_lock{
                        lock.fetch()
                    }
                    if not lock.locked or ignore_lock{
                        if not ignore_lock{
                            lock.acquire()
                        }
                        trying=False
                        if index is not None{
                            collection.create_index([(index, pymongo.ASCENDING)], unique=True)
                        }else{
                            index='_id'
                        }
                        mod_count=0
                        for doc in documents{
                            if index in doc{
                                query={index: doc[index]}
                                result=collection.replace_one(query, doc, upsert=True) 
                                if result.modified_count > 0{
                                    mod_count+=result.modified_count
                                }
                                if result.upserted_id{
                                    mod_count+=1
                                }
                            }else{
                                result=collection.insert_one(doc)
                                if result.inserted_id{
                                    mod_count+=1
                                }
                            }
                        }
                        if verbose{
                            self.logger.info('Inserted size: {}...OK'.format(mod_count))
                        }
                        if not ignore_lock{
                            lock.release()
                        }
                    }else{
                        raise Exception('Cannot insert on collection {} of db {}, collection is locked!'.format(collection_str,db.name))
                    }
                }except Exception as e{
                    if 'Cannot insert on collection' not in str(e){
                        raise e
                    }
                    tries+=1
                    if tries <= MAX_TRIES {
                        self.logger.warn('Collection {} of db {}, collection is locked, trying again later! {} of {}'.format(collection_str,db.name,tries,MAX_TRIES))
                        time.sleep(SECONDS_BETWEEN_TRIES)
                    }else{
                        trying=False
                        raise e
                    }
                }
            }
        }
    }

    def getLock(self,db,collection_str,lease=None){
        if not lease{
            lease=MongoDB.QUEUE_TIMEOUT_WITHOUT_PROGRESS
        }
        lock=MongoLock(db[collection_str],'__MongoDB-Inserting__',lease=lease)
        lock.fetch()
        return lock
    }

    def checkIfCollectionIsLocked(self,db=None,collection_str=None,lock=None){
        if not lock{
            lock=MongoLock(db[collection_str],'__MongoDB-Inserting__',lease=MongoDB.QUEUE_TIMEOUT_WITHOUT_PROGRESS)
        }
        lock.fetch()
        return lock.locked
    }

    def checkIfListOfCollectionsExistsAndItsNotLocked(self,db,collections_to_check){
        if all(el in db.list_collection_names() for el in collections_to_check){
            for col in collections_to_check{
                if self.checkIfCollectionIsLocked(db,col) {
                    return False
                }
            }
            return True
        }else{
            return False
        }
    }

    def startQueue(self,id=0){ 
        consumer_id='processor_{}-{}'.format(socket.gethostname(),id)
        self.consumer_id=consumer_id
        self.queues={}
        collection=self.client[MongoDB.QUEUE_DB_NAME][MongoDB.QUEUE_COL_CRAWLER_NAME]
        self.queues[MongoDB.QUEUE_COL_CRAWLER_NAME]=MongoQueue(collection, consumer_id=consumer_id, timeout=MongoDB.QUEUE_TIMEOUT_WITHOUT_PROGRESS, max_attempts=3)
        collection=self.client[MongoDB.QUEUE_DB_NAME][MongoDB.QUEUE_COL_PROCESSOR_NAME]
        self.queues[MongoDB.QUEUE_COL_PROCESSOR_NAME]=MongoQueue(collection, consumer_id=consumer_id, timeout=MongoDB.QUEUE_TIMEOUT_WITHOUT_PROGRESS, max_attempts=3)
    }

    def getQueues(self){
        return self.queues
    }

    def getQueueConsumerId(self){
        return self.consumer_id
    }

    def getQueueNames(self){
        return self.client[MongoDB.QUEUE_DB_NAME].list_collection_names()
    }

    def clearQueue(self,name){
        return self.queues[name].clear()
    }

    def getAllQueueJobs(self){
        out={}
        queues=self.getQueueNames()
        for queue_name in queues{
            queue=self.client[MongoDB.QUEUE_DB_NAME][queue_name]
            jobs=queue.find({})
            parsed_jobs=[]
            for job in jobs{
                job_dict={'task':job['payload']['task']}
                if 'args' in job['payload']{
                    if job['payload']['args']{
                        job_dict['args']=job['payload']['args']
                    }
                }
                if job['locked_by'] is None{
                    status='WAITING'
                }else{
                    status='RUNNING'
                    locked_at=job['locked_at']
                    end_date=locked_at+datetime.timedelta(0,self.queues[queue_name].timeout+3)
                    if end_date<datetime.datetime.now(){
                        query={"_id": job['_id']}
                        if job['attempts'] > self.queues[queue_name].max_attempts{
                            status='FAILED ALL ATTEMPS'
                            queue.remove(query)
                        }else{
                            status='FAILED'
                        }
                        time_failed=datetime.datetime.now()-end_date
                        if datetime.timedelta(hours=2)<time_failed{
                            status='CANCELING'
                            update={"$set": {"locked_by": None, "locked_at": None},"$inc": {"attempts": 1}}
                            queue.find_and_modify(query,update=update)
                        }
                    }
                }
                job_dict['status']=status
                if status=='RUNNING'{
                    job_dict['worker']=job['locked_by']
                }
                job_dict['attemps']=job['attempts']
                if job['last_error']{
                    job_dict['error']=job['last_error']
                }
                job_dict['id']=job['_id']
                parsed_jobs.append(job_dict)
            }
            if len(parsed_jobs)>0{
                out[queue_name]=parsed_jobs
            }
        }
        return out
    }

    def insertOnQueue(self,queue,task,args=None){
        payload={'task': task}
        if args{
            payload['args']=args
        }
        self.queues[queue].put(payload)
    }

    def insertOnCrawlerQueue(self,task,args=None){
        self.insertOnQueue(MongoDB.QUEUE_COL_CRAWLER_NAME,task,args)
    }

    def insertOnProcessorQueue(self,task,args=None){
        self.insertOnQueue(MongoDB.QUEUE_COL_PROCESSOR_NAME,task,args)
    }

    def findOneOnDB(self,db,collection,query,wait_unlock=True){
        if self.dummy{
            raise Exception('Find one is not supported on DummyMode')
        }
        if wait_unlock{
            while self.checkIfCollectionIsLocked(db,collection){
                time.sleep(1)
            }
        }
        result=db[collection].find(query)
        found=result.count()
        if found>0{
            return result.next()
        }
    }

    def findOneOnDBFromIndex(self,db,collection,index_field,index_value,wait_unlock=True){
        query={index_field:index_value}
        return self.findOneOnDB(db,collection,query,wait_unlock=wait_unlock)
    }

    def getAllDbNames(self,filter_non_data=True){
        dbs=self.client.list_database_names()
        if filter_non_data{
            if 'admin' in dbs {
                dbs.remove('admin')
            }
            if 'config' in dbs {
                dbs.remove('config')
            }
            if 'local' in dbs {
                dbs.remove('local')
            }
        }
        return dbs
    }

    def dumpDB(self,db,base_path){
        dst_path=Utils.joinPath(base_path,db.name)
        zip_path=dst_path+'.zip'
        self.logger.info('Dumping database {} to file {}...'.format(db.name,zip_path))
        Utils.createFolderIfNotExists(dst_path)
        collections=db.list_collection_names()
        for col in collections{
            col_path=Utils.joinPath(dst_path,col)
            Utils.createFolderIfNotExists(col_path)
            collection=db[col]
            indexes_info=collection.index_information()
            for k,v in indexes_info.items(){
                key_name=v['key'][0][0]
                if key_name!='_id'{
                    idx_path=Utils.joinPath(col_path,key_name+'.idx')
                    Utils.saveFile(idx_path,key_name)
                }
            }
            for document in collection.find(){
                doc_path=Utils.joinPath(col_path,str(document['_id'])+'.json')
                Utils.saveJson(doc_path,document,use_bson=True)
            }
        }
        Utils.zip(dst_path,zip_path,base=db.name)
        Utils.deletePath(dst_path)
        self.logger.info('Dumped database {} to file {}...OK'.format(db.name,zip_path))
        return zip_path
    }

    def restoreDB(self,compressed_db_dump,db_name=None){
        if not db_name{
            db_name=Utils.filenameFromPath(compressed_db_dump)
        }
        self.logger.info('Restoring database {} from file {}...'.format(db_name,compressed_db_dump))
        uncompressed_root=Utils.unzip(compressed_db_dump,db_name,delete=False)
        folders_inside=os.listdir(uncompressed_root)
        if len(folders_inside)>1{
            raise Exception('Wrong compressed file format on restoreDB, please use dumpDB to generate the compressed file. File:{}'.format(compressed_db_dump))
        }
        uncompressed_path=Utils.joinPath(uncompressed_root,folders_inside[0])
        db=self.getDB(db_name)
        for col in os.listdir(uncompressed_path){
            index=None
            col_path=Utils.joinPath(uncompressed_path,col)
            for doc in os.listdir(col_path){
                if doc.endswith('.json'){
                    pass
                }elif doc.endswith('.idx'){
                    index=Utils.removeExtFromFilename(doc)
                }else{
                    raise Exception('Wrong compressed file format on restoreDB, please use dumpDB to generate the compressed file. File:{}'.format(compressed_db_dump))
                }
            }
            for doc in os.listdir(col_path){
                if doc.endswith('.json'){
                    doc_path=Utils.joinPath(col_path,doc)
                    self.insertOneOnDB(db,Utils.loadJson(doc_path,use_bson=True),col,index=index,verbose=False)
                }
            }
        }
        Utils.deletePath(uncompressed_root)
        self.logger.info('Restored database {} from file {}...OK'.format(db_name,compressed_db_dump))
    }

    def findAllOnDB(self,db,collection,wait_unlock=True,query=dict(),projection=None){
        if self.dummy{
            raise Exception('Find all is not supported on DummyMode')
        }
        if wait_unlock{
            while self.checkIfCollectionIsLocked(db,collection){
                time.sleep(1)
            }
        }
        if projection {
            return db[collection].find(query,projection)
        }else{
            return db[collection].find(query)
        }
    }
}