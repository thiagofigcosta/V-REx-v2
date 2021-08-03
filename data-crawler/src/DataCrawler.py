#!/bin/python
# -*- coding: utf-8 -*-

import urllib
import urllib.request
import pandas as pd
import os
import codecs
import re
import time
from Utils import Utils
from XmlDictParser import XmlDictParser
from MongoDB import MongoDB

class DataCrawler(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'
    SOURCES=[]
    TMP_FOLDER='tmp/crawler/'

    def __init__(self, mongo, logger){
        self.logger=logger
		self.mongo=mongo
        self.loadDatabases()
        self.references=self.mongo.loadReferences()
    }
    
    def downloadFromLink(self,link,filename,timeout=600){
        retries=3
        fake_headers={}
        fake_headers['User-Agent']='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'
        fake_headers['Accept']='text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        fake_headers['Accept-Charset']='ISO-8859-1,utf-8;q=0.7,*;q=0.3'
        fake_headers['Accept-Encoding']='none'
        fake_headers['Accept-Language']='en-US,en;q=0.8'
        fake_headers['Connection']='keep-alive'
        for i in range(retries+1){
            try{
                Utils.createFolderIfNotExists(DataCrawler.TMP_FOLDER)
                if filename.startswith(DataCrawler.TMP_FOLDER){
                    path=filename
                }else{
                    path=Utils.joinPath(DataCrawler.TMP_FOLDER,filename)
                }
                self.logger.info('Downloading {} to {}...'.format(link,filename))
                req = urllib.request.Request(link, headers=fake_headers)
                with urllib.request.urlopen(req,timeout=timeout) as response{
                    if response.code == 200{
                        self.logger.info('HTTP Response Code = 200')
                        piece_size = 4096 # 4 KiB                        
                        with open(path, 'wb') as file{          
                            while True{
                                one_piece=response.read(piece_size)
                                if not one_piece{
                                    break
                                }
                                file.write(one_piece)
                            }
                        }
                        self.logger.info('Downloaded {} to {}...OK'.format(link,filename))
                        return path
                    }else{
                        raise Exception('Response code {} - {}'.format(response.code),response.code)
                    }
                }
            } except Exception as e {
                if str(e) in ('HTTP Error 404: Not Found','HTTP Error 403: Forbidden') or i>=retries{
                    raise e
                }else{
                    self.logger.exception(e,fatal=False)
                    self.logger.error('Failed to download ({} of {}) trying again...'.format(i+1,retries))
                }
            }
        }
    }

    def loadDatabases(self){
        DataCrawler.SOURCES.append({'id':'CVE_MITRE','index':'cve','direct_download_url':'https://cve.mitre.org/data/downloads/allitems.csv.gz'})
        DataCrawler.SOURCES.append({'id':'CWE_MITRE','index':'cwe','direct_download_url':'https://cwe.mitre.org/data/xml/cwec_latest.xml.zip'})
        DataCrawler.SOURCES.append({'id':'CVE_NVD','index':'cve','direct_download_urls':['https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2020.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2019.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2018.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2017.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2016.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2015.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2014.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2013.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2012.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2011.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2010.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2009.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2008.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2007.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2006.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2005.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2004.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2003.json.zip','https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2002.json.zip']})
        DataCrawler.SOURCES.append({'id':'CAPEC_MITRE','index':'capec','direct_download_url':'http://capec.mitre.org/data/archive/capec_latest.zip'})
        DataCrawler.SOURCES.append({'id':'OVAL','index':'oval','direct_download_url':'https://oval.cisecurity.org/repository/download/5.11.2/all/oval.xml.zip'})
        DataCrawler.SOURCES.append({'id':'CVE_DETAILS','index':'cve','base_download_url':'https://www.cvedetails.com/cve/'})
        DataCrawler.SOURCES.append({'id':'EXPLOIT_DB','index':'exploit','base_download_url':'https://www.exploit-db.com/exploits/'})
        # TODO download data from: https://www.kb.cert.org/vuls/search/
        # TODO download data from: https://www.securityfocus.com/vulnerabilities
        # TODO download data from: https://www.broadcom.com/support/security-center/attacksignatures
        # TODO download data from: ????????? Rapid 7’s Metasploit, D2 Security’s Elliot Kit and Canvas Exploitation Framework, OpenVAS
    }

    @staticmethod
    def getAllDatabasesId(){
        ids=[]
        for source in DataCrawler.SOURCES{
            ids.append(source['id'])
        }
        return ids
    }

    def getAllDatabasesIds(self){
        return DataCrawler.getAllDatabasesId()
    }

    def getSourceFromId(self,id){
        for source in DataCrawler.SOURCES{
            if source['id']==id{
                return source
            }
        }
        return None
    }

    def parseDBtoDocuments(self,id,paths,update_callback=None){
        source=self.getSourceFromId(id)
        if type(paths) is str{
            path=paths
        }elif type(paths) is list{
            path='Multiple files at {}'.format(Utils.parentFromPath(paths[0]))
        }else{
            raise Exception('Unknown data type for paths on parseDBtoDocuments. Type was {}'.format(type(paths)))
        }
        self.logger.info('Parsing data {} for {}...'.format(path,id))
        documents=[]
        if id=='CVE_MITRE'{
            columns_size=Utils.countCSVColumns(path)
            df=pd.read_csv(path,header=None,engine='python',names=range(columns_size)).values
            documents.append({source['index']:'__metadata__'})
            documents[0]['CVE Version']=Utils.changeStrDateFormat(re.sub('[^0-9]','', str(df[0][0])),'%Y%m%d','%d/%m/%Y')
            documents[0]['Update Date']=Utils.changeStrDateFormat(re.sub('[^0-9]','', str(df[1][0])),'%Y%m%d','%d/%m/%Y')
            documents[0]['Inserted At']=Utils.getTodayDate()
            columns=[source['index']]
            for i in range(1,columns_size){
                columns.append(df[2][i])
            }
            df=df[10:] # remove header
            documents[0]['Data Count']=len(df)
            for cve in df{
                cve_entry={}
                for i in range(columns_size){
                    column=columns[i]
                    value=cve[i]
                    if value==value{
                        cve_entry[column]=value
                        if column==source['index']{
                            ref=value.replace('CVE-', '')
                            self.references['cve'].add(ref)
                        }
                    }
                }
                documents.append(cve_entry)
                if update_callback { update_callback() }
            }
        }elif id=='CWE_MITRE'{ # TODO Severe workarounds, it is not the ideal solution but xmlschema is not working
            xmldict=XmlDictParser.fromFile(path,filter=True)
            self.logger.info('Searching for {} schema...'.format(id))
            possible_schemas=xmldict['schemaLocation'].split(' ')
            for possible_schema in possible_schemas{
                if re.search(r'.*\.xsd$', possible_schema){
                    schema_url=possible_schema
                    break
                }
            }
            if not schema_url{
                raise Exception('Schema not found')
            }
            self.logger.info('Found {} schema...OK'.format(id))
            self.logger.info('Downloading schema for {}...'.format(id))
            schema_path=self.downloadFromLink(schema_url,'{}_schema.xsd'.format(source['id']))
            self.logger.info('Downloaded schema for {}...OK'.format(id))
            xmldict=XmlDictParser.fromFileWithSchema2(path,schema_path,filter=True)
            
            xmldict=XmlDictParser.recursiveRemoveKey(xmldict,'br')
            xmldict=XmlDictParser.recursiveRemoveKey(xmldict,'style')
            documents.append({source['index']:'__metadata__'})
            documents[0]['CWE Version']=xmldict['Version']
            documents[0]['Update Date']=Utils.changeStrDateFormat(xmldict['Date'],'%Y-%m-%d','%d/%m/%Y')
            documents[0]['Data Count']=len(xmldict['Weaknesses']['Weakness'])
            documents[0]['Inserted At']=Utils.getTodayDate()
            for cwe in xmldict['Weaknesses']['Weakness']{
                cwe_entry={}
                for k,v in cwe.items(){
                    if k=='ID'{
                        k=source['index']
                        self.references['cwe'].add(int(v))
                    }
                    cwe_entry[k]=v
                }
                documents.append(cwe_entry)
                if update_callback { update_callback() }
            }
            for cat in xmldict['Categories']['Category']{
                if 'Relationships' in cat {
                    category_data={'Category':{}}
                    for k,v in cat.items(){
                        if k != 'Relationships'{
                            category_data['Category'][k]=v
                        }
                    }
                    if type(cat['Relationships']['Has_Member']) is dict{
                        cat['Relationships']['Has_Member']=[cat['Relationships']['Has_Member']]
                    }
                    for member in cat['Relationships']['Has_Member']{
                        loc=Utils.getIndexOfDictList(documents,source['index'],member['CWE_ID'])
                        if loc{
                            documents[loc]={**documents[loc],**category_data}
                        }
                    }
                }
            }
            for view in xmldict['Views']['View']{
                if 'Members' in view {
                    view_data={'View':{}}
                    for k,v in view.items(){
                        if k != 'Members'{
                            view_data['View'][k]=v
                        }
                    }
                    if type(view['Members']['Has_Member']) is dict{
                        view['Members']['Has_Member']=[view['Members']['Has_Member']]
                    }
                    for member in view['Members']['Has_Member']{
                        loc=Utils.getIndexOfDictList(documents,source['index'],member['CWE_ID'])
                        if loc{
                            documents[loc]={**documents[loc],**view_data}
                        }
                    }
                }
            }
            for i in range(len(documents)){
                if 'References' in documents[i]{
                    if type(documents[i]['References']['Reference']) is not list{
                        documents[i]['References']['Reference']=[documents[i]['References']['Reference']]
                    }
                    refs=[]
                    for ref in documents[i]['References']['Reference']{
                        ref_if=ref['External_Reference_ID']
                        for ext_ref in xmldict['External_References']['External_Reference']{
                            if ext_ref['Reference_ID']==ref_if{
                                ref_data=ext_ref.copy()
                                ref_data.pop('Reference_ID', None)
                                refs.append(ref_data)
                                break
                            }
                        }
                    }
                    documents[i]['References']=refs
                }
            }
            #prettify
            for i in range(len(documents)){
                if 'Related_Weaknesses' in documents[i]{
                    documents[i]['Related_Weaknesses']=documents[i]['Related_Weaknesses']['Related_Weakness']
                    if type(documents[i]['Related_Weaknesses']) is not list{
                        documents[i]['Related_Weaknesses']=[documents[i]['Related_Weaknesses']]
                    }
                }
                if 'Common_Consequences' in documents[i]{
                    documents[i]['Common_Consequences']=documents[i]['Common_Consequences']['Consequence']
                    if type(documents[i]['Common_Consequences']) is not list{
                        documents[i]['Common_Consequences']=[documents[i]['Common_Consequences']]
                    }
                }
                if 'Demonstrative_Examples' in documents[i]{
                    documents[i]['Demonstrative_Examples']=documents[i]['Demonstrative_Examples']['Demonstrative_Example']
                    if type(documents[i]['Demonstrative_Examples']) is not list{
                        documents[i]['Demonstrative_Examples']=[documents[i]['Demonstrative_Examples']]
                    }
                } 
                if 'Observed_Examples' in documents[i]{
                    documents[i]['Observed_Examples']=documents[i]['Observed_Examples']['Observed_Example']
                    if type(documents[i]['Observed_Examples']) is not list{
                        documents[i]['Observed_Examples']=[documents[i]['Observed_Examples']]
                    }
                } 
                if 'Potential_Mitigations' in documents[i]{
                    documents[i]['Potential_Mitigations']=documents[i]['Potential_Mitigations']['Mitigation']
                    if type(documents[i]['Potential_Mitigations']) is not list{
                        documents[i]['Potential_Mitigations']=[documents[i]['Potential_Mitigations']]
                    }
                }  
                if 'Taxonomy_Mappings' in documents[i]{
                    documents[i]['Taxonomy_Mappings']=documents[i]['Taxonomy_Mappings']['Taxonomy_Mapping']
                    if type(documents[i]['Taxonomy_Mappings']) is not list{
                        documents[i]['Taxonomy_Mappings']=[documents[i]['Taxonomy_Mappings']]
                    }
                } 
                if 'Notes' in documents[i]{
                    documents[i]['Notes']=documents[i]['Notes']['Note']
                    if type(documents[i]['Notes']) is not list{
                        documents[i]['Notes']=[documents[i]['Notes']]
                    }
                } 
                if 'Related_Attack_Patterns' in documents[i]{
                    documents[i]['Related_Attack_Patterns']=documents[i]['Related_Attack_Patterns']['Related_Attack_Pattern']
                    if type(documents[i]['Related_Attack_Patterns']) is not list{
                        documents[i]['Related_Attack_Patterns']=[documents[i]['Related_Attack_Patterns']]
                    }
                }        
                documents[i]=XmlDictParser.compressDictOnFollowingKeys(documents[i],['p','li','ul','div','i'])
                documents[i]=XmlDictParser.recursiveRemoveEmpty(documents[i])
                if update_callback { update_callback() }
            }
            Utils.deletePath(schema_path)
        }elif id=='CVE_NVD'{
            documents.append({source['index']:'__metadata__'})
            documents[0]['CVE Version']=[]
            documents[0]['Update Date']=[]
            documents[0]['Data Count']=0
            documents[0]['Inserted At']=Utils.getTodayDate()
            for json_path in paths{
                cves_data=Utils.loadJson(json_path)
                if cves_data['CVE_data_version'] not in documents[0]['CVE Version']{
                    documents[0]['CVE Version'].append(cves_data['CVE_data_version'])
                }
                if cves_data['CVE_data_timestamp'] not in documents[0]['Update Date']{
                    documents[0]['Update Date'].append(cves_data['CVE_data_timestamp'])
                }
                documents[0]['Data Count']+=int(cves_data['CVE_data_numberOfCVEs'])
                for cve_data in cves_data['CVE_Items']{
                    cve_entry={}
                    for k,v in cve_data.items(){
                        if k=='cve'{
                           cve_entry[source['index']]=v['CVE_data_meta']['ID']
                           ref=v['CVE_data_meta']['ID'].replace('CVE-', '')
                           self.references['cve'].add(ref)
                           for k2,v2 in v.items(){
                               if k2 not in ('data_type','data_format','data_version','CVE_data_meta'){
                                    if k2=='references'{
                                        v2=v2['reference_data']
                                        for i in range(len(v2)){
                                            ref_data=v2[i]
                                            if 'tags' in ref_data and ref_data['tags']==[]{
                                                ref_data.pop('tags', None)
                                            }
                                        }
                                    }
                                    cve_entry[k2]=v2
                               }
                           }
                        }else{
                            if k=='configurations'{
                                v=v['nodes']
                            }
                            cve_entry[k]=v
                        }
                    }
                    if source['index'] in cve_entry{
                        documents.append(cve_entry)
                        if update_callback {
                            update_callback()
                        }
                    }else{
                        raise Exception('No CVE id found on source {} path {}'.format(id,json_path))
                    }
                }
            }
        }elif id=='CAPEC_MITRE'{
            xmldict=XmlDictParser.fromFileWithSchema2(paths[0],paths[1],filter=True)
            xmldict=XmlDictParser.stringfyDict(xmldict)
            xmldict=XmlDictParser.recursiveRemoveKey(xmldict,'br')
            xmldict=XmlDictParser.recursiveRemoveKey(xmldict,'style')
            documents.append({source['index']:'__metadata__'})
            documents[0]['CAPEC Version']=xmldict['Version']
            documents[0]['Update Date']=Utils.changeStrDateFormat(xmldict['Date'],'%Y-%m-%d','%d/%m/%Y')
            documents[0]['Data Count']=len(xmldict['Attack_Patterns']['Attack_Pattern'])
            documents[0]['Inserted At']=Utils.getTodayDate()
            for capec in xmldict['Attack_Patterns']['Attack_Pattern']{
                capec_entry={}
                for k,v in capec.items(){
                    if k=='ID'{
                        k=source['index']
                        self.references['capec'].add(int(v))
                    }
                    capec_entry[k]=v
                }
                documents.append(capec_entry)
                if update_callback { update_callback() }
            }
            for cat in xmldict['Categories']['Category']{
                if 'Relationships' in cat {
                    category_data={'Category':{}}
                    for k,v in cat.items(){
                        if k != 'Relationships'{
                            category_data['Category'][k]=v
                        }
                    }
                    if type(cat['Relationships']['Has_Member']) is dict{
                        cat['Relationships']['Has_Member']=[cat['Relationships']['Has_Member']]
                    }
                    for member in cat['Relationships']['Has_Member']{
                        loc=Utils.getIndexOfDictList(documents,source['index'],member['CAPEC_ID'])
                        if loc{
                            documents[loc]={**documents[loc],**category_data}
                        }
                    }
                }
            }
            for view in xmldict['Views']['View']{
                if 'Members' in view {
                    view_data={'View':{}}
                    for k,v in view.items(){
                        if k != 'Members'{
                            view_data['View'][k]=v
                        }
                    }
                    if type(view['Members']['Has_Member']) is dict{
                        view['Members']['Has_Member']=[view['Members']['Has_Member']]
                    }
                    for member in view['Members']['Has_Member']{
                        loc=Utils.getIndexOfDictList(documents,source['index'],member['CAPEC_ID'])
                        if loc{
                            documents[loc]={**documents[loc],**view_data}
                        }
                    }
                }
            }
            for i in range(len(documents)){
                if 'References' in documents[i]{
                    if type(documents[i]['References']['Reference']) is not list{
                        documents[i]['References']['Reference']=[documents[i]['References']['Reference']]
                    }
                    refs=[]
                    for ref in documents[i]['References']['Reference']{
                        ref_if=ref['External_Reference_ID']
                        for ext_ref in xmldict['External_References']['External_Reference']{
                            if ext_ref['Reference_ID']==ref_if{
                                ref_data=ext_ref.copy()
                                ref_data.pop('Reference_ID', None)
                                refs.append(ref_data)
                                break
                            }
                        }
                    }
                    documents[i]['References']=refs
                }
            }
            #prettify
            for i in range(len(documents)){
                if 'Related_Weaknesses' in documents[i]{
                    documents[i]['Related_Weaknesses']=documents[i]['Related_Weaknesses']['Related_Weakness']
                    if type(documents[i]['Related_Weaknesses']) is not list{
                        documents[i]['Related_Weaknesses']=[documents[i]['Related_Weaknesses']]
                    }
                }
                if 'Related_Attack_Patterns' in documents[i]{
                    documents[i]['Related_Attack_Patterns']=documents[i]['Related_Attack_Patterns']['Related_Attack_Pattern']
                    if type(documents[i]['Related_Attack_Patterns']) is not list{
                        documents[i]['Related_Attack_Patterns']=[documents[i]['Related_Attack_Patterns']]
                    }
                }
                if 'Execution_Flow' in documents[i]{
                    documents[i]['Execution_Flow']=documents[i]['Execution_Flow']['Attack_Step']
                    if type(documents[i]['Execution_Flow']) is not list{
                        documents[i]['Execution_Flow']=[documents[i]['Execution_Flow']]
                    }
                } 
                if 'Notes' in documents[i]{
                    documents[i]['Notes']=documents[i]['Notes']['Note']
                    if type(documents[i]['Notes']) is not list{
                        documents[i]['Notes']=[documents[i]['Notes']]
                    }
                }  
                if 'Consequences' in documents[i]{
                    documents[i]['Consequences']=documents[i]['Consequences']['Consequence']
                    if type(documents[i]['Consequences']) is not list{
                        documents[i]['Consequences']=[documents[i]['Consequences']]
                    }
                } 
                if 'Mitigations' in documents[i]{
                    documents[i]['Mitigations']=documents[i]['Mitigations']['Mitigation']
                    if type(documents[i]['Mitigations']) is not list{
                        documents[i]['Mitigations']=[documents[i]['Mitigations']]
                    }
                }   
                if 'Prerequisites' in documents[i]{
                    documents[i]['Prerequisites']=documents[i]['Prerequisites']['Prerequisite']
                    if type(documents[i]['Prerequisites']) is not list{
                        documents[i]['Prerequisites']=[documents[i]['Prerequisites']]
                    }
                }  
                if 'Skills_Required' in documents[i]{
                    documents[i]['Skills_Required']=documents[i]['Skills_Required']['Skill']
                    if type(documents[i]['Skills_Required']) is not list{
                        documents[i]['Skills_Required']=[documents[i]['Skills_Required']]
                    }
                }       
                documents[i]=XmlDictParser.compressDictOnFollowingKeys(documents[i],['p','li','ul','div','i','class'])
                documents[i]=XmlDictParser.recursiveRemoveEmpty(documents[i])
                if update_callback { update_callback() }
            }
        }elif id=='OVAL'{
            xmldict=XmlDictParser.fromFile(path,filter=True)
            documents.append({source['index']:'__metadata__'})
            documents[0]['OVAL Version']=xmldict['generator']['schema_version']
            documents[0]['Update Date']=Utils.changeStrDateFormat(xmldict['generator']['timestamp'],'%Y-%m-%dT%H:%M:%S','%d/%m/%Y')
            documents[0]['Data Count']=len(xmldict['definitions']['definition'])
            documents[0]['Inserted At']=Utils.getTodayDate()
            for definition in xmldict['definitions']['definition']{
                oval_entry={}
                add_entry=True
                for k,v in definition.items(){
                    if k=='id'{
                        k=source['index']
                        self.references['oval'].add(v)
                    }
                    if k=='metadata'{
                        for k2,v2 in v.items(){
                            oval_entry[k2]=v2
                            if k2=='oval_repository'{
                                for k3,v3 in v2['dates'].items(){
                                    if k3=='status_change'{
                                        if type(v3) is list{
                                            last_el=v3[len(v3)-1]
                                        }else{
                                            last_el=v3
                                        }
                                        if  'status_change' in last_el and last_el['status_change']=='DEPRECATED'{
                                            add_entry=False
                                        }
                                    }
                                }
                            }
                        }
                    }else{
                        oval_entry[k]=v
                    }
                }
                if add_entry{
                    documents.append(oval_entry)
                    if update_callback {
                        update_callback()
                    }
                }
            }
        }elif id=='EXPLOIT_DB'{
            patterns={}
            patterns[source['index']]=r'[.|\n]*EDB-ID:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['cve']=r'[.|\n]*CVE:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['author']=r'[.|\n]*Author:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['type']=r'[.|\n]*Type:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['platform']=r'[.|\n]*Platform:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['date']=r'[.|\n]*Date:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['vulnerable']=r'[.|\n]*Vulnerable App:[\s|\n]*<\/.*>[\s|\n]*<.*>((\n|.)*?)<\/.*>[.|\n]*'
            patterns['verified']=r'[.|\n]*EDB Verified:[\s|\n]*<\/.*>[\s|\n]*(<.*\n.*>)[\s|\n]*'
            patterns['code']=r'.*<code.*>((.|\n)*)<\/code>.*'
            documents.append({source['index']:'__metadata__'})
            documents[0]['Update Date']=Utils.getTodayDate()
            documents[0]['Data Count']=len(paths)
            documents[0]['Inserted At']=Utils.getTodayDate()
            for path in paths {
                exploit_entry={}
                raw_html=Utils.openFile(path)
                found_at_least_one=False
                for k,pattern in patterns.items(){
                    result=re.search(pattern, raw_html, re.MULTILINE)
                    if result{
                        found_at_least_one=True
                        v=result.group(1)
                        if 'mdi-close' in v{
                            v='False'
                        }elif 'mdi-check' in v{
                            v='True'
                        }
                        v=re.sub(r'<.*>','',v).strip()
                        if k=='cve'{
                            cve_ids=re.findall(r'[0-9]+\-[0-9]+', v,re.MULTILINE)
                            if type(cve_ids) is list and len(cve_ids)>0{
                                v=cve_ids[-1]
                            }else{
                                v='N/A'
                            }
                        }
                        if v and v!='N/A'{
                            exploit_entry[k]=v
                        }
                    }else{
                        self.logger.warn('Unknown \'{}\' value for file {} on {}'.format(k,path,id))
                    }
                }           
                if found_at_least_one{     
                    documents.append(exploit_entry)
                    if update_callback {
                        update_callback()
                    }
                }
            }
        }elif id=='CVE_DETAILS'{
            patterns={}
            patterns[source['index']]=r'Vulnerability Details *: *<.*>(CVE-[0-9\-]+)<\/'
            patterns['cvss score']=r'[.|\n]*CVSS Score[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['confidentiality imp.']=r'[.|\n]*Confidentiality Impact[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['integrity imp.']=r'[.|\n]*Integrity Impact[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['availability imp.']=r'[.|\n]*Availability Impact[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['complexity']=r'[.|\n]*Access Complexity[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['authentication']=r'[.|\n]*Authentication[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['gained acc.']=r'[.|\n]*Gained Access[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['vul. type']=r'[.|\n]*Vulnerability Type\(s\)[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['cwe']=r'[.|\n]*CWE ID[\s|\n]*<\/.*>[\s|\n]*((\n|.)*?)<\/.*>'
            patterns['description']=r'<div class="cvedetailssummary"(>[.|\n]*(\n|.)*?)[\s|\n]*<\/div>'
            filter_regex=r'.*>((.|\n)*)$'
            invalid_regex=r'<strong>Unknown CVE ID<\/strong>'
            table_regex=r'<table .*>(\n|.)*?<\/table>'
            patterns['prod. affected']=r'Products Affected By CVE-[0-9\-]+[\s|\n]*<\/.*>[\s|\n]*(<table .*>(.|\n)*?<\/table>)'
            patterns['versions affected']=r'Number Of Affected Versions By Product[\s|\n]*<\/.*>[\s|\n]*(<table .*>(.|\n)*?<\/table>)'
            patterns['references']=r'References For CVE-[0-9\-]*[\s|\n]*<\/.*>[\s|\n]*(<table .*>(.|\n)*?<\/table>)'
            patterns['metasploitable']=r'Metasploit Modules Related To CVE-[0-9\-]*[\s|\n]*<\/.*>[\s|\n]*<div id="metasploitmodstable">((.|\n)*?)<\/div>'

            documents.append({source['index']:'__metadata__'})
            documents[0]['Update Date']=Utils.getTodayDate()
            documents[0]['Data Count']=0
            documents[0]['Inserted At']=Utils.getTodayDate()
            for path in paths {
                cve_entry={}
                raw_html=Utils.openFile(path)
                if not re.search(invalid_regex, raw_html, re.MULTILINE){
                    found_at_least_one=False
                    for k,pattern in patterns.items(){
                        result=re.search(pattern, raw_html, re.MULTILINE)
                        if result{
                            found_at_least_one=True
                            v=result.group(1)
                            if re.search(table_regex, v, re.MULTILINE){
                                df=pd.read_html(v)[0]
                                headers=df.columns.values
                                rows=df.values
                                v=[]
                                for row in rows{
                                    row_entry={}
                                    for i in range(len(headers)){
                                        header=str(headers[i])
                                        if header != '#' and 'Unnamed' not in header{
                                            header=header.replace('&nbsp',' ').strip()
                                            value=str(row[i]).replace('&nbsp',' ').strip()
                                            if value != 'nan'{
                                                row_entry[header]=value
                                            }
                                        }
                                    }
                                    if k=='references'{
                                        keys=list(row_entry.keys())
                                        if len(keys)==1 and keys[0]=='0'{
                                            row_entry=row_entry['0']
                                        }
                                        row_entry=row_entry.split(' ')[0]
                                    }
                                    v.append(row_entry)
                                }
                                cve_entry[k]=v
                            }else{
                                result=re.search(filter_regex, v, re.MULTILINE)
                                if result{
                                    v=result.group(1).strip()
                                    if k=='description'{
                                        v=str(v).replace('<br>','\n')
                                        result2=re.search(r'((.|\n)*)<span class="datenote">((.|\n)*)<', v, re.MULTILINE)
                                        if result2{
                                            remain=result2.group(3)
                                            v=result2.group(1).strip()
                                            result3=re.search(r'Publish ?Date ?: ?([0-9\-]*)', remain, re.MULTILINE)
                                            result4=re.search(r'Last ?Update ?Date ?: ?([0-9\-]*)', remain, re.MULTILINE)
                                            if result3{
                                                cve_entry['publish date']=result3.group(1)
                                            }
                                            if result4{
                                                cve_entry['last mod date']=result4.group(1)
                                            }
                                        }
                                    }
                                    if k=='vul. type'{
                                        result2=re.search(r'<span class="(.*)">((.|\n)*)', v, re.MULTILINE)
                                        if result2{
                                            v='{} - {}'.format(result2.group(2).strip(),result2.group(1))
                                        }else{
                                            v=v.strip()
                                        }
                                    }
                                }
                                if not (k=='metasploitable' and v=='for more information)') and not (k=='cwe' and v=='CWE id is not defined for this vulnerability'){
                                    if v!='None'{
                                        cve_entry[k]=v
                                    }
                                }
                            }
                        }
                    }       
                    if found_at_least_one{
                        documents[0]['Data Count']+=1         
                        documents.append(cve_entry)
                        if update_callback {
                            update_callback()
                        }
                    }
                }
            }

        }else{
            raise Exception('Unknown id({}).'.format(id))
        }
        if type(paths) is str{
            path=paths
        }elif type(paths) is list{
            path='Multiple files at {}'.format(Utils.parentFromPath(paths[0]))
        }else{
            raise Exception('Unknown data type for paths on parseDBtoDocuments. Type was {}'.format(type(paths)))
        }
        self.logger.info('Parsed data {} for {}...OK'.format(path,id))
        return documents
    }

    def downloadRawDataAndParseFrom(self,id,update_callback=None){
        source=self.getSourceFromId(id)
        if id=='CVE_MITRE'{
            self.logger.info('Downloading CVEs from {}...'.format(id))
            path=self.downloadFromLink(source['direct_download_url'],'{}_all_items.csv.gz'.format(source['id']))
            self.logger.info('Downloaded CVEs from {}...OK'.format(id))
            destination_folder=Utils.gunzip(path,'')
            for file_str in os.listdir(destination_folder){
                if re.search(r'.*\.csv$', file_str){
                    csv_path=Utils.joinPath(destination_folder,file_str)
                }
            }
            if update_callback {
                update_callback()
            }
            documents=self.parseDBtoDocuments(id,csv_path,update_callback=update_callback)
            return documents, destination_folder
        }elif id=='CWE_MITRE'{
            self.logger.info('Downloading CWEs from {}...'.format(id))
            path=self.downloadFromLink(source['direct_download_url'],'{}_cwec_latest.xml.zip'.format(source['id']))
            self.logger.info('Downloaded CWEs from {}...OK'.format(id))
            destination_folder=Utils.unzip(path)
            for file_str in os.listdir(destination_folder){
                if re.search(r'cwe.*\.xml$', file_str){
                    xml_path=Utils.joinPath(destination_folder,file_str)
                }
            }
            if update_callback {
                update_callback()
            }
            documents=self.parseDBtoDocuments(id,xml_path,update_callback=update_callback)
            return documents, destination_folder            
        }elif id=='CVE_NVD'{
            self.logger.info('Downloading CVEs from {}...'.format(id))
            urls = source['direct_download_urls']
            paths = []
            for url in urls{
                paths.append(self.downloadFromLink(url,Utils.filenameFromPath(url,get_extension=True)))
            }
            self.logger.info('Downloaded CWEs from {}...OK'.format(id))
            for path in paths{
                 destination_folder=Utils.unzip(path,destination_folder='CVE_NVD_unziped/')
            }
            json_files=[]
            for file_str in os.listdir(destination_folder){
                if re.search(r'.*\.json$', file_str){
                    json_files.append(Utils.joinPath(destination_folder,file_str))
                }
            }
            if update_callback {
                update_callback()
            }
            documents=self.parseDBtoDocuments(id,json_files,update_callback=update_callback)
            return documents, destination_folder
        }elif id=='CAPEC_MITRE'{
            self.logger.info('Downloading CAPECs from {}...'.format(id))
            path=self.downloadFromLink(source['direct_download_url'],'{}_latest.zip'.format(source['id']))
            self.logger.info('Downloaded CAPECs from {}...OK'.format(id))
            destination_folder=Utils.unzip(path)
            for file_str in os.listdir(destination_folder){
                if re.search(r'.*\.xml$', file_str){
                    xml_path=Utils.joinPath(destination_folder,file_str)
                }
                if re.search(r'.*\.xsd$', file_str){
                    xsd_path=Utils.joinPath(destination_folder,file_str)
                }
            }
            if update_callback {
                update_callback()
            }
            documents=self.parseDBtoDocuments(id,[xml_path,xsd_path],update_callback=update_callback)
            return documents, destination_folder            
        }elif id=='OVAL'{
            self.logger.info('Downloading OVALs from {}...'.format(id))
            path=self.downloadFromLink(source['direct_download_url'],'{}_all.zip'.format(source['id']))
            self.logger.info('Downloaded OVALs from {}...OK'.format(id))
            destination_folder=Utils.unzip(path)
            for file_str in os.listdir(destination_folder){
                if re.search(r'.*\.xml$', file_str){
                    xml_path=Utils.joinPath(destination_folder,file_str)
                }
            }
            if update_callback {
                update_callback()
            }
            documents=self.parseDBtoDocuments(id,xml_path,update_callback=update_callback)
            return documents, destination_folder            
        }elif id=='EXPLOIT_DB'{
            self.logger.info('Downloading EXPLOITs from {}...'.format(id))
            destination_folder=Utils.getTmpFolder(id)
            exploit_id=1
            max_known_exploit=49915
            downloaded=[]
            for file_str in os.listdir(destination_folder){
                result=re.search(r'.*id-([0-9]+)\..*$', file_str)
                if result{
                    downloaded.append(int(result.group(1)))
                }
                if re.search(r'.*id-([0-9]+)\.DELETEME$', file_str){
                    Utils.deletePath(Utils.joinPath(destination_folder,file_str))
                }
            }
            if len(downloaded)>0{
                exploit_id=max(downloaded)+1
            }
            timeouts=0
            max_timeouts=2
            while True{
                try{
                    self.downloadFromLink(source['base_download_url']+str(exploit_id),Utils.joinPath(destination_folder,'{}_id-{}.html'.format(source['id'],exploit_id)),timeout=120)
                    timeouts=0
                    if update_callback {
                        update_callback()
                    }
                }except Exception as e {
                    if exploit_id>max_known_exploit{
                        if str(e)=='HTTP Error 404: Not Found'{
                            self.logger.info('Last exploit found, id {}'.format(exploit_id-1))
                            break
                        }elif str(e) in ('<urlopen error timed out>','<urlopen error _ssl.c:1106: The handshake operation timed out>'){
                            self.logger.exception(e,fatal=False)
                            break
                        }else{
                            raise e
                        }
                    }else{
                        if str(e)=='HTTP Error 404: Not Found'{
                            self.logger.warn('Exploit with id {} not found. 404. Keep going until: {}'.format(exploit_id,max_known_exploit))
                        }elif str(e) in ('<urlopen error timed out>','<urlopen error _ssl.c:1106: The handshake operation timed out>'){
                            if timeouts<max_timeouts{
                                timeouts+=1
                                self.logger.warn('Server probably blocked access temporally on id {}. Waiting 20 Minutes...'.format(exploit_id))
                                exploit_id-=1
                                time.sleep(1200)
                            }else{  
                                self.logger.warn('Downloaded EXPLOITs from {}...FAILED. Server probably blocked access temporally. Aborting...'.format(exploit_id))
                                self.logger.exception(e,fatal=False)
                                documents=None
                                Utils.saveFile(Utils.joinPath(destination_folder,'{}_id-{}.DELETEME'.format(source['id'],exploit_id-1)),'') # checkpoint
                                return documents, destination_folder
                            }
                        }else{
                            raise e
                        }
                    }
                }
                exploit_id+=1
            }
            self.logger.info('Downloaded EXPLOITs from {}...OK'.format(id))
            paths=[]
            for file_str in os.listdir(destination_folder){
                result=re.search(r'.*id-([0-9]+)\.html$', file_str)
                if result{
                    paths.append(Utils.joinPath(destination_folder,file_str))
                    self.references['exploit'].add(int(result.group(1)))
                }
            }
            documents=self.parseDBtoDocuments(id,paths,update_callback=update_callback)
            return documents, destination_folder 
        }elif id=='CVE_DETAILS'{
            self.logger.info('Downloading CVEs from {}...'.format(id))
            destination_folder=Utils.getTmpFolder(id)
            cves=list(self.references['cve'])
            for file_str in os.listdir(destination_folder){
                result=re.search(r'.*id-([0-9\-]+)\..*$', file_str)
                if result{
                    cves.remove(result.group(1))
                }
            }
            max_timeouts=2
            timeouts=0
            for i in range(len(cves)){
                cve=cves[i]
                cve_formatted='CVE-{}'.format(cve)
                try{
                    self.downloadFromLink(source['base_download_url']+str(cve_formatted),Utils.joinPath(destination_folder,'{}_id-{}.html'.format(source['id'],cve)),timeout=120)
                    timeouts=0
                    if update_callback {
                        update_callback()
                    }
                }except Exception as e{
                    if str(e)=='HTTP Error 404: Not Found'{
                        self.logger.warn('Exploit with id {} not found. 404.'.format(cve))
                    }elif str(e) in ('<urlopen error timed out>','<urlopen error _ssl.c:1106: The handshake operation timed out>'){
                        if timeouts<max_timeouts{
                            timeouts+=1
                            self.logger.warn('Server probably blocked access temporally on CVE-{}. Waiting 20 Minutes...'.format(cve))
                            i=-1
                            time.sleep(1200)
                        }else{
                            raise e
                        }
                    }else{
                        raise e
                    }
                }
            }
            self.logger.info('Downloaded CVEs from {}...OK'.format(id))
            paths=[]
            for file_str in os.listdir(destination_folder){
                if re.search(r'.*\.html$', file_str){
                    paths.append(Utils.joinPath(destination_folder,file_str))
                }
            }
            documents=self.parseDBtoDocuments(id,paths,update_callback=update_callback)
            return documents, destination_folder 
        }else{ 
            raise Exception('Unknown id({}).'.format(id))
        }
    }

    def downloadRawDataFromSources(self,sources=None,update_callback=None){
        DELETE_DOWNLOADED_SOURCES_AFTER_PARSE=False
        failed=[]
        for source in DataCrawler.SOURCES{
            if sources is None or source['id'] in sources{
                try{
                    documents,tmp_path=self.downloadRawDataAndParseFrom(source['id'],update_callback=update_callback)
                    if documents is not None{
                        self.mongo.insertManyOnDB(self.mongo.getRawDB(),documents,source['id'],source['index'])
                        if DELETE_DOWNLOADED_SOURCES_AFTER_PARSE{
                            Utils.deletePath(tmp_path)
                        }
                    }else{
                        failed.append(source['id'])
                    }
                } except Exception as e {
                    self.logger.exception(e,fatal=False)
                    failed.append(source['id'])
                }finally{
                    self.mongo.saveReferences(self.references)
                }
            }
        }
        if len(failed)>0{
            return failed
        }
    }

    def loopOnQueue(self){
        while True{
            job=self.mongo.getQueues()[MongoDB.QUEUE_COL_CRAWLER_NAME].next()
            if job is not None{
                payload=job.payload
                task=payload['task']
                try{
                    self.logger.info('Running job {}-{}...'.format(task,job.job_id))
                    if task=='DownloadAll'{
                        for db_id in self.getAllDatabasesIds(){
                            self.mongo.getQueues()[MongoDB.QUEUE_COL_CRAWLER_NAME].put({'task': 'Download','args':{'id':db_id}})
                        }
                    }elif task=='Download'{
                        if payload['args']['id']=='CVE_DETAILS'{
                            if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getRawDB(),['CVE_MITRE','CVE_NVD']){
                                self.logger.warn('Returning {} job to queue, because it does not have its requirements fulfilled'.format(payload['args']['id']))
                                job.put_back()
                                job=None
                                time.sleep(20)
                            }else{
                                time.sleep(10)
                            }
                        }
                        if job{
                            failed=self.downloadRawDataFromSources(sources=[payload['args']['id']],update_callback=lambda: job.progress())
                            if failed{
                                raise Exception('Failed to download from {}'.format(','.join(failed)))
                            }
                        }
                    }
                    if job{
                        job.complete()
                        self.logger.info('Runned job {}-{}...'.format(task,job.job_id))
                    }
                }except Exception as e{
                    job.error(str(e))
                    self.logger.error('Failed to run job {}-{}...'.format(task,job.job_id))      
                    self.logger.exception(e)
                }
            }
        }
    }
}




