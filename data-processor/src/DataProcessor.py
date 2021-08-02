#!/bin/python
# -*- coding: utf-8 -*-

from Utils import Utils
from FeatureGenerator import FeatureGenerator
import re
import nltk
import math 
from nltk.stem import WordNetLemmatizer
import time
from MongoDB import MongoDB

class DataProcessor(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    TMP_FOLDER='tmp/processor/'

    def __init__(self, mongo, logger){
        self.logger=logger
		self.mongo=mongo
        self.references=self.mongo.loadReferences()
        nltk.data.path.append('res')
    }

    def mergeCve(self,update_callback=None){
        raw_db=self.mongo.getRawDB()
        keys_source={'CVE_MITRE':set(),'CVE_NVD':set(),'CVE_DETAILS':set()}
        cve_collections=[k for k,_ in keys_source.items()]
        if not all(el in raw_db.list_collection_names() for el in cve_collections){
            raise Exception('Mongo does not contains every needed collection: {}'.format(' - '.join(cve_collections)))
        }
        verbose_frequency=666
        iter_count=0
        data_size=0
        self.references=self.mongo.loadReferences()
        total_iters=len(self.references['cve'])
        self.logger.info('Running \"Merge\" on CVE Data...')
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'merged_cve')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for cve_ref in self.references['cve']{
            merged_entry={}
            for col in cve_collections{
                data=self.mongo.findOneOnDBFromIndex(raw_db,col,'cve','CVE-{}'.format(cve_ref))
                if data{
                    for k,v in data.items(){
                        if k.lower()=='references'{
                            k='references-{}'.format(col)
                        }
                        if k.lower()=='description'{
                            k='description-{}'.format(col)
                        }
                        if k!= '_id'{
                            merged_entry[k]=v
                            keys_source[col].add(k)
                        }
                    }
                }
            }
            if update_callback { update_callback() }
            iter_count+=1
            if len(merged_entry)>0{
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),merged_entry,'merged_cve','cve',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(merged_entry)
            }
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.info('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Merge\" on CVE Data...OK')
    }

    def flatternAndSimplifyCve(self,update_callback=None){
        # cve - OK
        # Status - OK
        # description-CVE_MITRE - OK
        # references-CVE_MITRE - OK
        # Phase - OK
        # Votes - OK
        # problemtype - OK
        # references-CVE_NVD - OK
        # description-CVE_NVD - OK
        # configurations - OK
        # impact - OK
        # publishedDate - OK
        # lastModifiedDate - OK
        # cvss score - OK
        # confidentiality imp. - OK
        # integrity imp. - OK
        # availability imp. - OK
        # complexity - OK
        # authentication - OK
        # gained acc. - OK
        # vul. type - OK
        # publish date - OK
        # last mod date - OK
        # description-CVE_DETAILS - OK
        # prod. affected - OK
        # versions affected - OK
        # references-CVE_DETAILS - OK
        # cwe - OK
        # metasploitable - OK
        # Comments - OK
        self.logger.info('Running \"Flattern and Simplify\" on CVE Data...')
        merged_data=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'merged_cve')
        verbose_frequency=1333
        iter_count=0
        data_size=0
        total_iters=merged_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'flat_cve')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for cve in merged_data{
            # References
            cve['References']=[]
            cve['References_class']=[]
            if 'references-CVE_NVD' in cve{
                refs_mitre=cve['references-CVE_NVD']
                for ref in refs_mitre{
                    ref_url=ref['url'].strip()
                    if ref_url not in cve['References']{
                        cve['References'].append(ref_url)
                        if 'tags' in ref{
                            cve['References_class'].append(ref['tags'])
                        }else{
                            cve['References_class'].append(['URL'])
                        }
                    }else{
                        ref_idx=cve['References'].index(ref_url)
                        for tag in ref['tags']{
                            if tag.strip() not in cve['References_class'][ref_idx]{
                                cve['References_class'][ref_idx].append(tag.strip())
                            }
                        }
                    }
                }
                cve.pop('references-CVE_NVD', None)
            }
            if 'references-CVE_MITRE' in cve{
                refs_mitre=cve['references-CVE_MITRE'].split('|')
                for ref in refs_mitre{
                    ref=ref.split(':',1)
                    if len(ref)>1{
                        ref_url=ref[1].strip()
                        if ref_url not in cve['References']{
                            cve['References'].append(ref_url)
                            cve['References_class'].append([ref[0].strip()])
                        }else{
                            ref_idx=cve['References'].index(ref_url)
                            if ref[0].strip() not in cve['References_class'][ref_idx]{
                                cve['References_class'][ref_idx].append(ref[0].strip())
                            }
                        }
                    }
                }
                cve.pop('references-CVE_MITRE', None)
            }
            if 'references-CVE_DETAILS' in cve{
                refs_details=cve['references-CVE_DETAILS']
                for ref in refs_details{
                    ref_url=ref.strip()
                    if ref_url not in cve['References']{
                        cve['References'].append(ref_url)
                        cve['References_class'].append(['URL'])
                    }
                }
                cve.pop('references-CVE_DETAILS', None)
            }
            for i in range(len(cve['References_class'])){
                if len(cve['References_class'][i])>1 and 'URL' in cve['References_class'][i]{
                    cve['References_class'][i].remove('URL')
                }
            }
            if len(cve['References'])==0{
                cve.pop('References', None)
                cve.pop('References_class', None)
            }
            # References

            # Description
            cve['Description']=[]
            if 'description-CVE_MITRE' in cve{
                cve_description=cve['description-CVE_MITRE'].strip()
                if cve_description not in cve['Description']{
                    cve['Description'].append(cve_description)
                }
                cve.pop('description-CVE_MITRE', None)
            }
            if 'description-CVE_NVD' in cve{
                for desc in cve['description-CVE_NVD']['description_data']{
                    cve_description=desc['value'].strip()
                    if cve_description not in cve['Description']{
                        cve['Description'].append(cve_description)
                    }
                }
                cve.pop('description-CVE_NVD', None)
            }
            if 'description-CVE_DETAILS' in cve{
                cve_description=cve['description-CVE_DETAILS'].strip().replace('&quot;','\"')
                if cve_description not in cve['Description']{
                    cve['Description'].append(cve_description)
                }
                cve.pop('description-CVE_DETAILS', None)
            }
            if len(cve['Description'])>0{
                cve['Description']='\n'.join(cve['Description'])
                if '** RESERVED **' in cve['Description']{
                    cve['Status']='RESERVED'
                    cve.pop('Description', None)
                }
            }else{
                cve.pop('Description', None)
            }
            # Description
            
            # Phase
            if 'Phase' in cve{
                result=re.match(r'([a-zA-Z]*) \(([0-9]{8})\)', cve['Phase'])
                if result{
                    cve['Phase']=result.group(1)
                    cve['{}Date'.format(result.group(1).lower())]=Utils.changeStrDateFormat(result.group(2),'%Y%m%d','%d/%m/%Y')
                }else{
                    cve['Phase']=cve['Phase'].strip()
                }
            }
            # Phase

            if 'Votes' in cve{
                cve.pop('Votes', None)
            }

            # problemtype AND cwe
            cve['CWEs']=[]
            if 'problemtype' in cve{
                cwes=cve['problemtype']['problemtype_data']
                for cwe in cwes{
                    cwes2=cwe['description']
                    for cwe in cwes2{
                        cwe=cwe['value'].split('-')[1]
                        if cwe.isdecimal(){
                            cve['CWEs'].append(cwe)
                        }
                    }
                }
                cve.pop('problemtype', None)
            }
            if 'cwe' in cve{
                cwe=cve['cwe']
                if cwe.isdecimal() and cwe not in cve['CWEs']{
                    cve['CWEs'].append(cwe)
                }
                cve.pop('cwe', None)
            }
            if len(cve['CWEs'])==0{
                cve.pop('CWEs', None)
            }
            # problemtype AND cwe

            # configurations and prod. affected and versions affected
            cve['vendors']=set()
            cve['products']=set()
            if 'configurations' in cve{
                cve['CPEs_vulnerable']=[]
                cve['CPEs_non_vulnerable']=[]
                for conf in cve['configurations']{
                    if 'cpe_match' in conf{
                        for cpe in conf['cpe_match']{
                            if cpe['vulnerable']{
                                cve['CPEs_vulnerable'].append(cpe['cpe23Uri'])
                                uncompressed_cpe=cpe['cpe23Uri'].split(':')
                                cve['vendors'].add(uncompressed_cpe[3].lower())
                                cve['products'].add(uncompressed_cpe[4].lower())
                            }else{
                                cve['CPEs_non_vulnerable'].append(cpe['cpe23Uri'])
                            }
                        }
                    }
                    if 'children' in conf{
                        for children in conf['children']{
                            if 'cpe_match' in children{
                                for cpe in children['cpe_match']{
                                    if cpe['vulnerable']{
                                        cve['CPEs_vulnerable'].append(cpe['cpe23Uri'])
                                        uncompressed_cpe=cpe['cpe23Uri'].split(':')
                                        cve['vendors'].add(uncompressed_cpe[3].lower())
                                        cve['products'].add(uncompressed_cpe[4].lower())
                                    }else{
                                        cve['CPEs_non_vulnerable'].append(cpe['cpe23Uri'])
                                    }
                                }
                            }
                        }
                    }
                }
                cve.pop('configurations', None)
                if len(cve['CPEs_non_vulnerable'])==0{
                    cve.pop('CPEs_non_vulnerable', None)
                }
                if len(cve['CPEs_vulnerable'])==0{
                    cve.pop('CPEs_vulnerable', None)
                }
            }

            if 'versions affected' in cve {
                cve['AffectedVersionsCount']=0
                for prod in cve['versions affected']{
                    cve['vendors'].add(prod['Vendor'].lower())
                    if 'Product' in prod{
                        cve['products'].add(prod['Product'].lower())
                    }
                    cve['AffectedVersionsCount']+=int(prod['Vulnerable Versions'])
                }
                cve.pop('versions affected', None)
            }
            if 'prod. affected' in cve {
                cve.pop('prod. affected', None)
            }
            cve['vendors']=list(cve['vendors'])
            if '*' in cve['vendors']{
                cve['vendors'].remove('*')
            }
            cve['products']=list(cve['products'])
            if '*' in cve['products']{
                cve['products'].remove('*')
            }
            if len(cve['vendors'])==0{
                cve.pop('vendors', None)
            }
            if len(cve['products'])==0{
                cve.pop('products', None)
            }
            # configurations and prod. affected and versions affected

            # impact
            if 'impact' in cve{
                if 'baseMetricV2' in cve['impact']{
                    cve['CVSS_version']=cve['impact']['baseMetricV2']['cvssV2']['version']
                    cve['CVSS_score']=cve['impact']['baseMetricV2']['cvssV2']['baseScore']
                    cve['CVSS_AV']=cve['impact']['baseMetricV2']['cvssV2']['accessVector']
                    cve['CVSS_AC']=cve['impact']['baseMetricV2']['cvssV2']['accessComplexity']
                    cve['CVSS_AuPR']=cve['impact']['baseMetricV2']['cvssV2']['authentication']
                    cve['CVSS_C']=cve['impact']['baseMetricV2']['cvssV2']['confidentialityImpact']
                    cve['CVSS_I']=cve['impact']['baseMetricV2']['cvssV2']['integrityImpact']
                    cve['CVSS_A']=cve['impact']['baseMetricV2']['cvssV2']['availabilityImpact']
                    cve['CVSS_exploitabilityScore']=cve['impact']['baseMetricV2']['exploitabilityScore']
                    cve['CVSS_impactScore']=cve['impact']['baseMetricV2']['impactScore']
                }
                if 'baseMetricV3' in cve['impact']{
                    cve['CVSS_version']=cve['impact']['baseMetricV3']['cvssV3']['version']
                    cve['CVSS_score']=cve['impact']['baseMetricV3']['cvssV3']['baseScore']
                    cve['CVSS_AV']=cve['impact']['baseMetricV3']['cvssV3']['attackVector']
                    cve['CVSS_AC']=cve['impact']['baseMetricV3']['cvssV3']['attackComplexity']
                    cve['CVSS_AuPR']=cve['impact']['baseMetricV3']['cvssV3']['privilegesRequired']
                    cve['CVSS_UI']=cve['impact']['baseMetricV3']['cvssV3']['userInteraction']
                    cve['CVSS_S']=cve['impact']['baseMetricV3']['cvssV3']['scope']
                    cve['CVSS_C']=cve['impact']['baseMetricV3']['cvssV3']['confidentialityImpact']
                    cve['CVSS_I']=cve['impact']['baseMetricV3']['cvssV3']['integrityImpact']
                    cve['CVSS_A']=cve['impact']['baseMetricV3']['cvssV3']['availabilityImpact']
                    cve['CVSS_exploitabilityScore']=cve['impact']['baseMetricV3']['exploitabilityScore']
                    cve['CVSS_impactScore']=cve['impact']['baseMetricV3']['impactScore']
                }
                cve.pop('impact', None)
            }
            # impact

            if 'publishedDate' in cve{
                cve['publishedDate']=Utils.changeStrDateFormat(cve['publishedDate'].split('T')[0],'%Y-%m-%d','%d/%m/%Y')
            }

            if 'lastModifiedDate' in cve{
                cve['lastModifiedDate']=Utils.changeStrDateFormat(cve['lastModifiedDate'].split('T')[0],'%Y-%m-%d','%d/%m/%Y')
            }

            if 'cvss score' in cve{
                if 'CVSS_score' not in cve{
                    cve['CVSS_score']=cve['cvss score']
                }
                cve.pop('cvss score', None)
            }

            if 'confidentiality imp.' in cve{
                if 'CVSS_C' not in cve{
                    cve['CVSS_C']=cve['confidentiality imp.'].upper()
                }
                cve.pop('confidentiality imp.', None)
            }

            if 'integrity imp.' in cve{
                if 'CVSS_I' not in cve{
                    cve['CVSS_I']=cve['integrity imp.'].upper()
                }
                cve.pop('integrity imp.', None)
            }

            if 'availability imp.' in cve{
                if 'CVSS_A' not in cve{
                    cve['CVSS_A']=cve['availability imp.'].upper()
                }
                cve.pop('availability imp.', None)
            }

            if 'complexity' in cve{
                if 'CVSS_AC' not in cve{
                    cve['CVSS_AC']=cve['complexity'].upper()
                }
                cve.pop('complexity', None)
            }

            if 'authentication' in cve{
                if 'CVSS_AuPR' not in cve{
                    cve['CVSS_AuPR']=cve['authentication'].replace('Not required','NONE').upper()
                }
                cve.pop('authentication', None)
            }

            if 'vul. type' in cve{
                if cve['vul. type']{
                    vul=cve['vul. type'].split('-',1)
                    if len(vul)>1{
                        cve['Type']=vul[1].strip()
                    }else{
                        cve['Type']=vul[0].strip()
                    }
                }
                cve.pop('vul. type', None)
            }

            if 'publish date' in cve{
                date=Utils.changeStrDateFormat(cve['publish date'],'%Y-%m-%d','%d/%m/%Y')
                if 'publishedDate' not in cve{
                    cve['publishedDate']=date
                }else{
                    if Utils.isFirstStrDateOldest(date,cve['publishedDate'],'%d/%m/%Y'){ # oldest
                        cve['publishedDate']=date
                    }
                }
                cve.pop('publish date', None)
            }

            if 'last mod date' in cve{
                date=Utils.changeStrDateFormat(cve['last mod date'],'%Y-%m-%d','%d/%m/%Y')
                if 'lastModifiedDate' not in cve{
                    cve['lastModifiedDate']=date
                }else{
                    if Utils.isFirstStrDateOldest(cve['lastModifiedDate'],date,'%d/%m/%Y'){ # newest
                        cve['lastModifiedDate']=date
                    }
                }
                cve.pop('last mod date', None)
            }

            if 'gained acc.' in cve{
                cve.pop('gained acc.', None)
            }

            if 'metasploitable' in cve{
                modules={}
                for module in cve['metasploitable']{
                    for k,v in module.items(){
                        result=re.match(r'.*Module type\s*:\s*([A-Za-z]*).*', v, re.MULTILINE)
                        mod_type='other'
                        if result{
                            mod_type=result.group(1)
                        }
                        if mod_type not in modules{
                            modules[mod_type]=1
                        }else{
                            modules[mod_type]+=1
                        }
                    }
                }
                cve.pop('metasploitable', None)
                cve['weaponized_modules_types']=[]
                cve['weaponized_modules_count']=[]
                for k,v in modules.items(){
                    cve['weaponized_modules_types'].append(k)
                    cve['weaponized_modules_count'].append(v)
                }
            }

            if 'Comments' in cve{
                cve['Comments']=len(cve['Comments'].split('|'))
            }

            for k,v in cve.items(){
                if type(v) not in (int,str,float) and k not in ('_id','References_class'){
                    if type(v) is list{
                        for el in v{
                            if type(el) not in (int,str,float){
                                raise Exception('Non-flat field on {} inside list {}: type:{} v:{}'.format(cve['cve'],k,type(el),el))
                            }
                        }
                    }else{
                        raise Exception('Non-flat field on {}: type:{} k:{} v:{}'.format(cve['cve'],type(v),k,v))
                    }
                }
            }
            if update_callback { update_callback() }
            if cve['Status']!='RESERVED'{
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),cve,'flat_cve','cve',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(cve)
            }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Flattern and Simplify\" on CVE Data...OK')
    }

    def flatternAndSimplifyOval(self,update_callback=None){
        self.logger.info('Running \"Flattern and Simplify\" on OVAL Data...')
        oval_data=self.mongo.findAllOnDB(self.mongo.getRawDB(),'OVAL')
        verbose_frequency=1333
        iter_count=0
        data_size=0
        total_iters=oval_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'flat_oval')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for oval in oval_data{
            oval_parsed={}
            append=False
            if 'reference' in oval{
                if type(oval['reference']) is list{
                    for ref in oval['reference']{
                        if ref['source']=='CVE'{
                            append=True
                            oval_parsed['CVE']=ref['ref_id']
                            break
                        }
                    }
                }else{
                    if oval['reference']['source']=='CVE'{
                        append=True
                        oval_parsed['CVE']=oval['reference']['ref_id']
                    }
                }
            }
            if append{
                oval_parsed['oval']=oval['oval'].split(':')[-1]
                oval_parsed['type']=oval['class']
                if 'oval_repository' in oval{
                    if 'dates' in oval['oval_repository']{
                        if 'submitted' in oval['oval_repository']['dates']{
                            oval_parsed['submittedDate']=Utils.changeStrDateFormat(oval['oval_repository']['dates']['submitted']['date'].split('T')[0],'%Y-%m-%d','%d/%m/%Y')
                        }
                        if 'modified' in oval['oval_repository']['dates']{
                            if type(oval['oval_repository']['dates']['modified']) is list{
                                oval_parsed['modifiedDate']=Utils.changeStrDateFormat(oval['oval_repository']['dates']['modified'][0]['date'].split('T')[0],'%Y-%m-%d','%d/%m/%Y')
                                for entry in oval['oval_repository']['dates']['modified']{
                                    date=Utils.changeStrDateFormat(entry['date'].split('T')[0],'%Y-%m-%d','%d/%m/%Y')
                                    if Utils.isFirstStrDateOldest(oval_parsed['modifiedDate'],date,'%d/%m/%Y'){ # newest
                                        oval_parsed['modifiedDate']=date
                                    }
                                }
                            }else{
                                oval_parsed['modifiedDate']=Utils.changeStrDateFormat(oval['oval_repository']['dates']['modified']['date'].split('T')[0],'%Y-%m-%d','%d/%m/%Y')
                            }
                        }
                    }
                }
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),oval_parsed,'flat_oval','oval',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(oval_parsed)
            }
            if update_callback { update_callback() }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Flattern and Simplify\" on OVAL Data...OK')
    }

    def flatternAndSimplifyCapec(self,update_callback=None){
        self.logger.info('Running \"Flattern and Simplify\" on CAPEC Data...')
        capec_data=self.mongo.findAllOnDB(self.mongo.getRawDB(),'CAPEC_MITRE')
        verbose_frequency=1333
        iter_count=0
        data_size=0
        total_iters=capec_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'flat_capec')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for capec in capec_data{
            if capec['capec']!='__metadata__' and capec['Status']!='Deprecated'{
                if 'Description' in capec and type(capec['Description']) is dict{
                    desc=""
                    for k,v in capec['Description'].items(){
                        if type(v) is list{
                            desc+='\n'.join(v)
                        }else{
                            desc+='\n{}'.format(v)
                        }
                    }
                    capec['Description']=desc
                }
                if 'Related_Attack_Patterns' in capec{
                    relationship={}
                    for rel in capec['Related_Attack_Patterns']{
                        if rel['Nature'] not in relationship{
                            rel['Nature']=[rel['CAPEC_ID']]
                        }else{
                            rel['Nature'].append(rel['CAPEC_ID'])
                        }
                    }
                    for k,v in relationship.items(){
                        capec[k]=v
                    }
                    capec.pop('Related_Attack_Patterns', None)
                }
                if 'Execution_Flow' in capec{
                    capec['Steps']=[]
                    for step in capec['Execution_Flow']{
                        if type(step) is dict{
                            capec['Steps'].append(step['Phase'])
                        }
                    }
                    capec.pop('Execution_Flow', None)
                }
                if 'Skills_Required' in capec{
                    capec['Skill']=[]
                    capec['Skill_level']=[]
                    for skill in capec['Skills_Required']{
                        if 'Skill' not in skill{
                            skill['Skill']='Not specified'
                        }
                        capec['Skill'].append(skill['Skill'])
                        capec['Skill_level'].append(skill['Level'])
                    }
                    capec.pop('Skills_Required', None)
                }
                if 'Resources_Required' in capec{
                    capec['Resources_req']=[]
                    if type(capec['Resources_Required']) is list{
                        for res in capec['Resources_Required']{
                            res=res['Resource']
                            if res.split(':',1)[0]!='None'{
                                capec['Resources_req'].append(res)
                            }
                        }
                    }else{
                        res=capec['Resources_Required']['Resource']
                        if type(res) is list{
                            for el in res{
                                capec['Resources_req'].append(el)
                            }
                        }elif res.split(':',1)[0]!='None'{
                            capec['Resources_req'].append(res)
                        }
                    }
                    capec.pop('Resources_Required', None)
                    if len(capec['Resources_req'])==0{
                        capec.pop('Resources_req', None)
                    }
                }
                if 'Consequences' in capec{
                    capec['Affected_Scopes']=set()
                    capec['Damage']=set()
                    for conseq in capec['Consequences']{
                        if type(conseq['Scope']) is not list{
                            conseq['Scope']=[conseq['Scope']]
                        }
                        for scope in conseq['Scope']{
                            capec['Affected_Scopes'].add(scope)
                        }
                        if type(conseq['Impact']) is not list{
                            conseq['Impact']=[conseq['Impact']]
                        }
                        for impac in conseq['Impact']{
                            capec['Damage'].add(impac)
                        }
                    }
                    capec.pop('Consequences', None)
                    capec['Affected_Scopes']=list(capec['Affected_Scopes'])
                    capec['Damage']=list(capec['Damage'])
                    if len(capec['Affected_Scopes'])==0{
                        capec.pop('Affected_Scopes', None)
                    }
                    if len(capec['Damage'])==0{
                        capec.pop('Damage', None)
                    }
                }
                if 'Example_Instances' in capec or 'value' in capec{
                    capec['Examples']=[]
                    if 'Example_Instances' in capec{
                        if type(capec['Example_Instances']['Example']) is dict{
                            if 'value' in capec['Example_Instances']['Example']{
                                capec['Example_Instances']=capec['Example_Instances']['Example']['value']
                            }else{
                                capec['Example_Instances']=[]
                            }
                        }elif type(capec['Example_Instances']['Example']) is not list{
                            capec['Example_Instances']=[capec['Example_Instances']['Example']]
                        }
                        for example in capec['Example_Instances']{
                            capec['Examples'].append(example)
                        }
                        capec.pop('Example_Instances', None)
                        if len(capec['Examples'])==0{
                            capec.pop('Examples', None)
                        }
                    }
                    if 'value' in capec{
                        for ex in capec['value']{
                            if ex not in capec['Examples']{
                                capec['Examples'].append(ex)
                            }
                        }
                        capec.pop('value', None)
                    }
                }
                if 'Related_Weaknesses' in capec{
                    capec['CWEs']=[]
                    if type(capec['Related_Weaknesses']) is not list{
                        capec['Related_Weaknesses']=[capec['Related_Weaknesses']]
                    }
                    for cwes in capec['Related_Weaknesses']{
                        capec['CWEs'].append(cwes['CWE_ID'])
                    }
                    capec.pop('Related_Weaknesses', None)
                }
                if 'References' in capec{
                    refs=[]
                    if type(capec['References']) is not list{
                        capec['References']=[capec['References']]
                    }
                    for ref in capec['References']{
                        if 'URL' in ref{
                            refs.append(ref['URL'])
                        }else{
                            refs.append(ref['Title'])
                        }
                        
                    }
                    capec.pop('References', None)
                    if len(refs)>0{
                        capec['References']=refs
                    }
                }
                if 'Content_History' in capec{
                    if 'Submission' in capec['Content_History']{
                        capec['submittedDate']=Utils.changeStrDateFormat(capec['Content_History']['Submission']['Submission_Date'],'%Y-%m-%d','%d/%m/%Y')
                    }
                    if 'Modification' in capec['Content_History']{
                        if type(capec['Content_History']['Modification']) is not list{
                            capec['Content_History']['Modification']=[capec['Content_History']['Modification']]
                        }
                        capec['modifiedDate']=Utils.changeStrDateFormat(capec['Content_History']['Modification'][0]['Modification_Date'],'%Y-%m-%d','%d/%m/%Y')
                        for entry in capec['Content_History']['Modification']{
                            date=Utils.changeStrDateFormat(entry['Modification_Date'],'%Y-%m-%d','%d/%m/%Y')
                            if Utils.isFirstStrDateOldest(capec['modifiedDate'],date,'%d/%m/%Y'){ # newest
                                capec['modifiedDate']=date
                            }
                        }
                    }
                    capec.pop('Content_History', None)
                }
                if 'Mitigations' in capec{
                    to_remove=[]
                    for i in range(len(capec['Mitigations'])){
                        if type(capec['Mitigations'][i]) is dict{
                            for k,v in capec['Mitigations'][i].items(){
                                if type(v) is list{
                                    for el in v{
                                        if type(el) is dict{
                                            if 'p' in el{
                                                el=el['p']
                                                if type(el) is list{
                                                    el='\n'.join(el)
                                                }
                                            }else{
                                                el=None
                                            }
                                        }
                                        if el{
                                            capec['Mitigations'].append(el)
                                        }
                                    }
                                }else{
                                    capec['Mitigations'].append(v)
                                }
                            }
                            to_remove.append(i)
                        }
                        if not capec['Mitigations'][i]{
                            to_remove.append(i)
                        }
                    }
                    capec['Mitigations'] = [i for j, i in enumerate(capec['Mitigations']) if j not in to_remove]
                }
                if 'Taxonomy_Mappings' in capec{
                    capec['Taxonomy']=[]
                    if type(capec['Taxonomy_Mappings']['Taxonomy_Mapping']) is not list{
                        capec['Taxonomy_Mappings']=[capec['Taxonomy_Mappings']['Taxonomy_Mapping']]
                    }else{
                        capec['Taxonomy_Mappings']=capec['Taxonomy_Mappings']['Taxonomy_Mapping']
                    }
                    for tax in capec['Taxonomy_Mappings']{
                        tmp_entry_id="Absent"
                        if 'Entry_ID' in tax{
                            tmp_entry_id=tax['Entry_ID']
                        }
                        capec['Taxonomy'].append('{}-{}'.format(tax['Taxonomy_Name'],tmp_entry_id))
                    }
                    capec.pop('Taxonomy_Mappings', None)
                }
                if 'Indicators' in capec{
                    capec['Indicators']=capec['Indicators']['Indicator']
                }
                if 'Category' in capec{
                    capec.pop('Category', None)
                }
                if 'Notes' in capec{
                    capec.pop('Notes', None)
                }
                if 'Alternate_Terms' in capec{
                    capec.pop('Alternate_Terms', None)
                }

                
                for k,v in capec.items(){
                    if type(v) not in (int,str,float) and k not in ('_id'){
                        if type(v) is list{
                            for el in v{
                                if type(el) not in (int,str,float){
                                    raise Exception('Non-flat field on {} inside list {}: type:{} v:{}'.format(capec['capec'],k,type(el),el))
                                }
                            }
                        }else{
                            raise Exception('Non-flat field on {}: type:{} k:{} v:{}'.format(capec['capec'],type(v),k,v))
                        }
                    }
                }
                if update_callback { update_callback() }
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),capec,'flat_capec','capec',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(capec)
            }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Flattern and Simplify\" on CAPEC Data...OK')
    }

    def flatternAndSimplifyCwe(self,update_callback=None){
        self.logger.info('Running \"Flattern and Simplify\" on CWE Data...')
        cwe_data=self.mongo.findAllOnDB(self.mongo.getRawDB(),'CWE_MITRE')
        verbose_frequency=1333
        iter_count=0
        data_size=0
        total_iters=cwe_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'flat_cwe')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for cwe in cwe_data{
            if cwe['cwe']!='__metadata__'{
                if 'Related_Weaknesses' in cwe{
                    relationship={}
                    for rel in cwe['Related_Weaknesses']{
                        if rel['Nature'] not in relationship{
                            rel['Nature']=[rel['CWE_ID']]
                        }else{
                            rel['Nature'].append(rel['CWE_ID'])
                        }
                    }
                    for k,v in relationship.items(){
                        cwe[k]=v
                    }
                    cwe.pop('Related_Weaknesses', None)
                }
                if 'Applicable_Platforms' in cwe{
                    if 'Language' in cwe['Applicable_Platforms']{
                        if type(cwe['Applicable_Platforms']['Language']) is not list{
                            cwe['Applicable_Platforms']['Language']=[cwe['Applicable_Platforms']['Language']]
                        }
                        for lang in cwe['Applicable_Platforms']['Language']{
                            if 'Name' in lang{
                                cwe['Language']=lang['Name']
                            }else{
                                cwe['Language']=lang['Class']
                            }
                        }
                    }
                    if 'Technology' in cwe['Applicable_Platforms']{
                        tech=[]
                        if type(cwe['Applicable_Platforms']['Technology']) is not list{
                            cwe['Applicable_Platforms']['Technology']=[cwe['Applicable_Platforms']['Technology']]
                        }
                        for tec in cwe['Applicable_Platforms']['Technology']{
                            if 'Class' in tec{
                                tech.append(tec['Class'])
                            }else{
                                tech.append(tec['Name'])
                            }
                        }
                        cwe['Technology']=tech
                    }
                    cwe.pop('Applicable_Platforms', None)
                }
                if 'Background_Details' in cwe{
                    cwe['Background_Details']=cwe['Background_Details']['Background_Detail']
                }
                if 'Modes_Of_Introduction' in cwe{
                    if type(cwe['Modes_Of_Introduction']['Introduction']) is not list{
                        cwe['Modes_Of_Introduction']['Introduction']=[cwe['Modes_Of_Introduction']['Introduction']]
                    } 
                    intros=[]
                    for intro in cwe['Modes_Of_Introduction']['Introduction']{
                        if type(intro) is dict{
                            intros.append(intro['Phase'])
                        }else{
                            intros.append('\n'.join(intro))
                        }
                    }
                    cwe['Modes_Of_Introduction']=intros
                }
                if 'Common_Consequences' in cwe{
                    cwe['Affected_Scopes']=set()
                    cwe['Damage']=set()
                    for conseq in cwe['Common_Consequences']{
                        if type(conseq['Scope']) is not list{
                            conseq['Scope']=[conseq['Scope']]
                        }
                        for scope in conseq['Scope']{
                            cwe['Affected_Scopes'].add(scope)
                        }
                        if 'Impact' in conseq{
                            if type(conseq['Impact']) is not list{
                                conseq['Impact']=[conseq['Impact']]
                            }
                            for impac in conseq['Impact']{
                                cwe['Damage'].add(impac)
                            }
                        }
                    }
                    cwe.pop('Common_Consequences', None)
                    cwe['Affected_Scopes']=list(cwe['Affected_Scopes'])
                    cwe['Damage']=list(cwe['Damage'])
                    if len(cwe['Affected_Scopes'])==0{
                        cwe.pop('Affected_Scopes', None)
                    }
                    if len(cwe['Damage'])==0{
                        cwe.pop('Damage', None)
                    }
                }
                if 'Potential_Mitigations' in cwe{
                    mitigations=[]
                    cwe['MitigationsEffectiveness']=[]
                    for mit in cwe['Potential_Mitigations']{
                        if type(mit) is dict{
                            if 'Description' in mit {
                                desc=mit['Description']
                                if desc {
                                    if type(desc) is dict and 'value' in desc{ 
                                        desc=desc['value']
                                    }
                                    if type(desc) is list{ 
                                        for dec in desc{
                                            mitigations.append(dec)
                                        }
                                    }else{
                                        mitigations.append(desc)
                                    }
                                }
                            }else{
                                phse=mit['Phase']
                                if type(phse) is list{ 
                                    for phe in phse{
                                        mitigations.append(phe)
                                    }
                                }else{
                                    mitigations.append(phse)
                                }
                            }
                            if 'Effectiveness' in mit {
                                cwe['MitigationsEffectiveness'].append(mit['Effectiveness'])
                            }else{
                                cwe['MitigationsEffectiveness'].append('UNKNOWN')
                            }
                        }else{
                            mitigations.append('\n'.join(mit))
                            cwe['MitigationsEffectiveness'].append('UNKNOWN')
                        }
                    }
                    cwe['Potential_Mitigations']=mitigations
                }
                if 'Demonstrative_Examples' in cwe{
                    examples=[]
                    for ex in cwe['Demonstrative_Examples']{
                        if 'Body_Text' in ex{
                            key='Body_Text'
                        }elif 'Intro_Text' in ex{
                            key='Intro_Text'
                        }else{
                           key='value'
                        }
                        for el in list(ex[key]){
                            if type(el) is dict{
                                ex[key].remove(el)
                            }elif type(el) is list{
                                examples.append('\n'.join(el))
                                ex[key].remove(el)
                            }
                        }
                        examples.append('\n'.join(ex[key]))
                    }
                    cwe['Demonstrative_Examples']=examples
                }
                if 'Observed_Examples' in cwe{
                    cves=[]
                    for ex in cwe['Observed_Examples']{
                        if 'CVE' in ex['Reference']{
                            cves.append(ex['Reference'])
                        }
                    }
                    cwe.pop('Observed_Examples', None)
                    if len(cves)>0{
                        cwe['CVEs']=cves
                    }
                }
                if 'References' in cwe{
                    refs=[]
                    if type(cwe['References']) is not list{
                        cwe['References']=[cwe['References']]
                    }
                    for ref in cwe['References']{
                        if 'URL' in ref{
                            refs.append(ref['URL'])
                        }else{
                            refs.append(ref['Title'])
                        }
                    }
                    cwe.pop('References', None)
                    if len(refs)>0{
                        cwe['References']=refs
                    }
                }
                if 'Content_History' in cwe{
                    if 'Submission' in cwe['Content_History']{
                        cwe['submittedDate']=Utils.changeStrDateFormat(cwe['Content_History']['Submission']['Submission_Date'],'%Y-%m-%d','%d/%m/%Y')
                    }
                    if 'Modification' in cwe['Content_History']{
                        if type(cwe['Content_History']['Modification']) is not list{
                            cwe['Content_History']['Modification']=[cwe['Content_History']['Modification']]
                        }
                        cwe['modifiedDate']=Utils.changeStrDateFormat(cwe['Content_History']['Modification'][0]['Modification_Date'],'%Y-%m-%d','%d/%m/%Y')
                        for entry in cwe['Content_History']['Modification']{
                            date=Utils.changeStrDateFormat(entry['Modification_Date'],'%Y-%m-%d','%d/%m/%Y')
                            if Utils.isFirstStrDateOldest(cwe['modifiedDate'],date,'%d/%m/%Y'){ # newest
                                cwe['modifiedDate']=date
                            }
                        }
                    }
                    cwe.pop('Content_History', None)
                }
                if 'Extended_Description' in cwe{
                    if type(cwe['Extended_Description']) is dict{
                        if 'value' in cwe['Extended_Description']{
                            cwe['Extended_Description']='\n'.join(cwe['Extended_Description']['value'])
                        }else{
                            cwe['Extended_Description']='\n'.join(cwe['Extended_Description'])
                        }
                    }
                }
                if 'Weakness_Ordinalities' in cwe{
                    ords=[]
                    if type(cwe['Weakness_Ordinalities']['Weakness_Ordinality']) is not list{
                        cwe['Weakness_Ordinalities']=[cwe['Weakness_Ordinalities']['Weakness_Ordinality']]
                    }
                    if type(cwe['Weakness_Ordinalities']) is dict{
                        cwe['Weakness_Ordinalities']=cwe['Weakness_Ordinalities']['Weakness_Ordinality']
                    }
                    for ordi in cwe['Weakness_Ordinalities']{
                        ords.append(ordi['Ordinality'])
                    }
                    cwe['Weakness_Ordinalities']=ords
                    if len(cwe['Weakness_Ordinalities'])==0{
                        cwe.pop('Weakness_Ordinalities', None)
                    }
                }
                if 'Alternate_Terms' in cwe{
                    cwe.pop('Alternate_Terms', None)
                }
                if 'Detection_Methods' in cwe{
                    cwe['Dectection']=[]
                    cwe['Dectection_Effectiveness']=[]
                    if type(cwe['Detection_Methods']['Detection_Method']) is not list{
                        cwe['Detection_Methods']['Detection_Method']=[cwe['Detection_Methods']['Detection_Method']]
                    }
                    for detec in cwe['Detection_Methods']['Detection_Method']{
                        if type(detec) is dict{
                            cwe['Dectection'].append(detec['Method'])
                            if 'Effectiveness' in detec{
                                cwe['Dectection_Effectiveness'].append(detec['Effectiveness'])
                            }else{
                                cwe['Dectection_Effectiveness'].append('Unknown')
                            }
                        }else{
                            cwe['Dectection'].append('\n'.join(detec))
                            cwe['Dectection_Effectiveness'].append('Unknown')
                        }
                    }
                    cwe.pop('Detection_Methods', None)
                    if len(cwe['Dectection'])==0{
                        cwe.pop('Dectection', None)
                    }
                    if len(cwe['Dectection_Effectiveness'])==0{
                        cwe.pop('Dectection_Effectiveness', None)
                    }
                }
                if 'Category' in cwe{
                    cwe['Category']=cwe['Category']['Summary']
                }
                if 'Taxonomy_Mappings' in cwe{
                    cwe['Taxonomy']=[]
                    if type(cwe['Taxonomy_Mappings']) is not list{
                        cwe['Taxonomy_Mappings']=[cwe['Taxonomy_Mappings']]
                    }
                    for tax in cwe['Taxonomy_Mappings']{
                        if 'Entry_ID' in tax{
                            cwe['Taxonomy'].append('{}-{}'.format(tax['Taxonomy_Name'],tax['Entry_ID']))
                        }else{
                            cwe['Taxonomy'].append(tax['Taxonomy_Name'])
                        }
                    }
                    cwe.pop('Taxonomy_Mappings', None)
                }
                if 'Related_Attack_Patterns' in cwe{
                    capecs=[]
                    for rel in cwe['Related_Attack_Patterns']{
                        capecs.append(rel['CAPEC_ID'])
                    }
                    cwe['Related_Attack_Patterns']=capecs
                }
                if 'Notes' in cwe{
                    cwe.pop('Notes', None)
                }
                if 'View' in cwe{
                    cwe['View']=cwe['View']['Name']
                }
                if 'Affected_Resources' in cwe{
                    cwe['Affected_Resources']=cwe['Affected_Resources']['Affected_Resource']
                }
                if 'Functional_Areas' in cwe{
                    cwe['Functional_Areas']=cwe['Functional_Areas']['Functional_Area']
                }
                if 'Background_Details' in cwe{
                    cwe.pop('Background_Details', None)
                }
                

                for k,v in cwe.items(){
                    if type(v) not in (int,str,float) and k not in ('_id'){
                        if type(v) is list{
                            for el in v{
                                if type(el) not in (int,str,float){
                                    raise Exception('Non-flat field on {} inside list {}: type:{} v:{}'.format(cwe['cwe'],k,type(el),el))
                                }
                            }
                        }else{
                            raise Exception('Non-flat field on {}: type:{} k:{} v:{}'.format(cwe['cwe'],type(v),k,v))
                        }
                    }
                }
                if update_callback { update_callback() }
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),cwe,'flat_cwe','cwe',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(cwe)
            }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Flattern and Simplify\" on CWE Data...OK')
    }

    def filterExploits(self,update_callback=None){
        self.logger.info('Running \"Filter\" on Exploit Data...')
        exploit_data=self.mongo.findAllOnDB(self.mongo.getRawDB(),'EXPLOIT_DB')
        verbose_frequency=1333
        iter_count=0
        data_size=0
        total_iters=exploit_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'exploits')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for exploit in exploit_data{
            if exploit['exploit']!='__metadata__' and 'cve' in exploit{
                if update_callback { update_callback() }
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),exploit,'exploits','exploit',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(exploit)
            }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Filter\" on Exploit Data...OK')
    }   

    def transformCve(self,update_callback=None){
        self.logger.info('Running \"Transform\" on CVE Data...')
        cve_data=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'flat_cve')
        verbose_frequency=5000
        iter_count=0
        data_size=0
        cves_refs=[]
        data_count=cve_data.count()
        total_iters=data_count*2 # read fields and transform
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'features_cve')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        fields_and_values={}
        bag_of_tags={}
        extracted_tags={}
        occurences_in_doc={}
        lemmatizer=WordNetLemmatizer()
        for cve in cve_data{
            if 'Status' not in cve or cve['Status']!='RESERVED'{
                cves_refs.append(cve['cve'])
                for k,v in cve.items(){
                    if k not in fields_and_values{
                        if k!='Description'{
                            fields_and_values[k]=set()
                        }else{
                            fields_and_values[k]={}
                        }
                    }
                    if k not in ('_id','cve','publishedDate','lastModifiedDate','modifiedDate','References','Description','assignedDate','CWEs','interimDate','weaponized_modules_count','CPEs_vulnerable','products','proposedDate','Comments','CVSS_score','CVSS_impactScore','CPEs_non_vulnerable','AffectedVersionsCount','CVSS_exploitabilityScore'){ # non enums
                        if k in ('References_class','vendors','weaponized_modules_types'){ # lists of enums
                            for el in v{
                                if type(el) is list{
                                    for el2 in el{
                                        fields_and_values[k].add(el2.replace(' ','_'))
                                    }
                                }else{
                                    fields_and_values[k].add(el.replace(' ','_'))
                                }
                            }
                        }else{
                            if type(v) is list{
                                for i in range(len(v)){
                                    if type(v[i]) is list{
                                        v[i]=tuple(v[i])
                                    }
                                }
                                v=tuple(v)
                            }
                            fields_and_values[k].add(v.replace(' ','_'))
                        }
                    }elif k=='Description'{
                        v=re.findall(r"[\w']+", v) # split text into words
                        for i in range(len(v)){
                            v[i]=lemmatizer.lemmatize(v[i])
                        }
                        v=' '.join(v) # join words into text
                        fields_and_values[k][cve['cve']]=v
                        keys=FeatureGenerator.extractKeywords(v)
                        filtered_keys=[]
                        not_allowed_patterns=[r'^[0-9\s]*$',r'^\/[a-zA-Z]*$',r'^(reference )?[cC][vV][eE][\-0-9]+$',r'`',r'^\/\/$',r'^>$',r'^<$',r'^&lt$',r'^&gt$',r'^&amp$',r'^\-$',r'^\/\/cwe$',r'^[\-0-9]+$',r'^\*$',r'^subject&#039$',r'^>?cwe[0-9\-]*$',r'^cvss$',r'^cvss vector$',r'^&#039$',r'^]$',r'^cvss [0-9]*$',r'^\*\* reject \*\*$',r'^0\/av$',r'^\/\/www$',r'^lead$',r'^result$',r'^wa$',r'^possibly$',r'^ac $',r'^pr$',r'^impact$',r'^created$',r'^covered$',r'^viewed$',r'^claim$',r'^cwe 426 untrusted search path$',r'^high$',r'^r2 sp1$',r'^sp2$',r'^unknown attack vector$',r'^commented$',r'^r2$',r'^sp1$',r'^present$',r'^cve$',r'^potentially$',r'^number$',r'^view$',r'^http www oracle$',r'^long$',r'^existence$',r'^unknown impact$',r'^[0-9]+ notes$',r'^determine$',r'^ha$']
                        occurences_in_doc[cve['cve']]={}
                        for key in keys{
                            insert=True
                            for pattern in not_allowed_patterns{
                                if re.match(pattern, key){
                                    insert=False 
                                    break
                                }
                            }
                            if insert{
                                filtered_keys.append(key)
                                occurences_in_doc[cve['cve']][key]=v.count(key)
                                occurences_in_doc[cve['cve']][key]=occurences_in_doc[cve['cve']][key] if occurences_in_doc[cve['cve']][key]>0 else 1
                                if key not in bag_of_tags{
                                    bag_of_tags[key]=1
                                }else{
                                    bag_of_tags[key]+=1
                                }
                            }
                        }
                        extracted_tags[cve['cve']]=filtered_keys
                    }
                }
            }
            if update_callback { update_callback() }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}%'.format((float(iter_count)/total_iters*100)))
            }
        }
        self.logger.info('Optimizing and caching data...')
        total_iters=data_count+len(cves_refs)
        cve_data=None
        bag_of_tags_sorted=sorted(bag_of_tags.items(), key=lambda x: x[1], reverse=True)
        bag_of_tags=[]
        bag_of_tags_and_occurences={}
        min_occurrences=200
        for tag,amount in list(bag_of_tags_sorted){
            if amount>=min_occurrences{
                bag_of_tags.append(tag)
                bag_of_tags_and_occurences[tag]=amount
            }
        }
        bag_of_tags_sorted=None
        # just to visualize 
        # for k,v in fields_and_values.items(){
        #     self.logger.clean(k)
        #     if k !='vendors'{
        #         for v2 in v{
        #             self.logger.clean('\t{}'.format(v2))
        #         }
        #     }elif k!='Description'{
        #         self.logger.clean('\t{}'.format(' | '.join(v)))
        #     }
        # }
        #  self.logger.clean('Len: {}'.format(len(bag_of_tags)))
        # for tag in bag_of_tags{
        #     self.logger.clean(tag)
        # }
        # just to visualize 

        for k,v in fields_and_values.items(){
            if k!='Description'{
                fields_and_values[k]=list(v)
            }
        }
        total_docs=len(fields_and_values['Description'])
        # the correct is number of documents containing, not occurences in documents (as below)
        # occurences_in_docs={}
        # for tag in bag_of_tags{
        #     occurences_in_docs[tag]=0
        #     for _,doc in fields_and_values['Description'].items(){
        #         occurences_in_docs[tag]+=doc.count(tag)
        #     }
        #     occurences_in_docs[tag]=occurences_in_docs[tag] if occurences_in_docs[tag] > 0 else 1
        # }
        self.logger.info('Optimized and cached data...OK')
        verbose_frequency=1333
        for cve_ref in cves_refs{
            cve=self.mongo.findOneOnDBFromIndex(self.mongo.getProcessedDB(),'flat_cve','cve',cve_ref)
            # _id - OK
            # cve - OK
            # Status - OK
            # Phase - OK
            # publishedDate - OK
            # lastModifiedDate - OK
            # References - OK
            # References_class - OK
            # Description - OK
            # assignedDate - OK
            # CWEs - OK
            # vendors - OK
            # products - OK
            # CPEs_vulnerable - OK
            # AffectedVersionsCount - OK
            # CVSS_version - OK
            # CVSS_score - OK
            # CVSS_AV - OK
            # CVSS_AC - OK
            # CVSS_AuPR - OK
            # CVSS_C - OK
            # CVSS_I - OK
            # CVSS_A - OK
            # CVSS_exploitabilityScore - OK
            # CVSS_impactScore - OK
            # CVSS_UI - OK
            # CVSS_S - OK
            # Type - OK
            # CPEs_non_vulnerable - OK
            # Comments - OK
            # proposedDate - OK
            # weaponized_modules_types - OK
            # weaponized_modules_count - OK
            # modifiedDate - OK
            # interimDate - OK
            featured_cve={}

            # _id ignore

            # enums
            if 'Status' not in cve{
                cve['Status']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('Status',cve['Status'],fields_and_values['Status']))

            if 'Phase' not in cve{
                cve['Phase']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('Phase',cve['Phase'],fields_and_values['Phase']))

            if 'References_class' not in cve{
                cve['References_class']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('Reference_type',FeatureGenerator.compressListOfLists(cve['References_class'],unique=True),fields_and_values['References_class']))

            if 'vendors' not in cve{
                cve['vendors']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('vendor',FeatureGenerator.compressListOfLists(cve['vendors'],unique=True),fields_and_values['vendors']))

            if 'CVSS_version' not in cve{
                cve['CVSS_version']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_version',cve['CVSS_version'],fields_and_values['CVSS_version']))

            if 'CVSS_AV' not in cve{
                cve['CVSS_AV']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_AV',cve['CVSS_AV'],fields_and_values['CVSS_AV']))

            if 'CVSS_AC' not in cve or cve['CVSS_AC']=='???'{
                cve['CVSS_AC']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_AC',cve['CVSS_AC'],[FeatureGenerator.ABSENT_FIELD_FOR_ENUM if x=='???' else x for x in fields_and_values['CVSS_AC']]))

            if 'CVSS_AuPR' not in cve or cve['CVSS_AuPR']=='???'{
                cve['CVSS_AuPR']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_AuPR',cve['CVSS_AuPR'],[FeatureGenerator.ABSENT_FIELD_FOR_ENUM if x=='???' else x for x in fields_and_values['CVSS_AuPR']]))

            if 'CVSS_C' not in cve or cve['CVSS_C']=='???'{
                cve['CVSS_C']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_C',cve['CVSS_C'],[FeatureGenerator.ABSENT_FIELD_FOR_ENUM if x=='???' else x for x in fields_and_values['CVSS_C']]))

            if 'CVSS_I' not in cve or cve['CVSS_I']=='???'{
                cve['CVSS_I']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_I',cve['CVSS_I'],[FeatureGenerator.ABSENT_FIELD_FOR_ENUM if x=='???' else x for x in fields_and_values['CVSS_I']]))

            if 'CVSS_A' not in cve or cve['CVSS_A']=='???'{
                cve['CVSS_A']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_A',cve['CVSS_A'],[FeatureGenerator.ABSENT_FIELD_FOR_ENUM if x=='???' else x for x in fields_and_values['CVSS_A']]))

            if 'CVSS_UI' not in cve{
                cve['CVSS_UI']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_UI',cve['CVSS_UI'],fields_and_values['CVSS_UI']))

            if 'CVSS_S' not in cve{
                cve['CVSS_S']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('CVSS_S',cve['CVSS_S'],fields_and_values['CVSS_S']))

            if 'Type' not in cve{
                cve['Type']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('Type',cve['Type'],fields_and_values['Type']))

            if 'weaponized_modules_types' not in cve{
                cve['weaponized_modules_types']=[FeatureGenerator.ABSENT_FIELD_FOR_ENUM]
            }
            if 'weaponized_modules_count' not in cve{
                cve['weaponized_modules_count']=[0]
            }
            weapon_exp=FeatureGenerator.buildFeaturesFromEnum('exploits_weaponized_type',cve['weaponized_modules_types'],fields_and_values['weaponized_modules_types'])
            for i in range(len(cve['weaponized_modules_types'])){
                for k,v in weapon_exp.items(){
                    if cve['weaponized_modules_types'][i].lower() in k and v==1{
                        count=int(cve['weaponized_modules_count'][i])
                        if count>0{
                            weapon_exp[k]=count
                        }
                    }
                }
            }
            featured_cve=dict(featured_cve,**weapon_exp) # weighted enum
            # enums

            # numbers
            featured_cve['exploits_weaponized_count']=sum(cve['weaponized_modules_count'])

            if 'References' not in cve{
                cve['References']=[]
            }
            featured_cve['references_count']=len(cve['References'])

            if 'products' not in cve{
                cve['products']=[]
            }
            featured_cve['products']=len(cve['products'])

            if 'AffectedVersionsCount' not in cve{
                cve['AffectedVersionsCount']=0
            }
            if featured_cve['products'] > cve['AffectedVersionsCount']{
                cve['AffectedVersionsCount']=featured_cve['products']
            }
            featured_cve['versions']=cve['AffectedVersionsCount']

            if 'CVSS_score' not in cve{
                cve['CVSS_score']=0
            }
            featured_cve['cvss_score']=cve['CVSS_score']

            if 'CVSS_exploitabilityScore' not in cve{
                cve['CVSS_exploitabilityScore']=0
                featured_cve['cvss_has_exploitability_score']=0
            }else{
                featured_cve['cvss_has_exploitability_score']=1
            }
            featured_cve['cvss_exploitability_score']=cve['CVSS_exploitabilityScore']

            if 'CVSS_impactScore' not in cve{
                cve['CVSS_impactScore']=0
                featured_cve['cvss_has_impact_score']=0
            }else{
                featured_cve['cvss_has_impact_score']=1
            }
            featured_cve['cvss_impact_score']=cve['CVSS_impactScore']

            if 'Comments' not in cve{
                cve['Comments']=0
            }
            featured_cve['comments']=cve['Comments']

            # ignore CPEs_vulnerable, already present on other fields

            if 'CPEs_non_vulnerable' not in cve{
                cve['CPEs_non_vulnerable']=[]
            }
            featured_cve['cpes_non_vulnerable_count']=len(cve['CPEs_non_vulnerable'])
            # numbers

            # Dates - extract features on enrich
            if 'lastModifiedDate' in cve or 'modifiedDate' in cve{
                if 'lastModifiedDate' in cve{
                    featured_cve['lastModifiedDate']=cve['lastModifiedDate']
                }
                if 'modifiedDate' in cve{
                    if 'lastModifiedDate' not in featured_cve{
                        featured_cve['lastModifiedDate']=cve['modifiedDate']
                    }else{
                        if Utils.isFirstStrDateOldest(featured_cve['lastModifiedDate'],cve['modifiedDate'],'%d/%m/%Y'){ # newest
                            featured_cve['lastModifiedDate']=cve['modifiedDate']
                        }
                    }
                }
            }

            if 'interimDate' in cve{
                featured_cve['interimDate']=cve['interimDate']
            }

            if 'proposedDate' in cve{
                featured_cve['proposedDate']=cve['proposedDate']
            }

            if 'assignedDate' in cve{
                featured_cve['assignedDate']=cve['assignedDate']
            }

            if 'publishedDate' in cve{
                featured_cve['publishedDate']=cve['publishedDate']
            }
            # Dates - extract features on enrich

            if 'Description' in cve{
                description_features=FeatureGenerator.buildFeaturesFromEnum('Description',extracted_tags[cve['cve']],bag_of_tags,has_absent=False)
                for tag in extracted_tags[cve['cve']]{
                    if tag in bag_of_tags{
                        tag_feature_name=FeatureGenerator.buildEnumKeyName('Description',tag)
                        if tag_feature_name in description_features and description_features[tag_feature_name]==1{
                            # description_features[tag_feature_name]=occurences_in_doc[cve['cve']][tag]*math.log(float(total_docs)/float(occurences_in_docs[tag]))
                            description_features[tag_feature_name]=occurences_in_doc[cve['cve']][tag]*math.log(float(total_docs)/float(bag_of_tags_and_occurences[tag]))
                        }
                    }
                }
                featured_cve=dict(featured_cve,**description_features)
            }else{
                featured_cve=dict(featured_cve,**FeatureGenerator.buildFeaturesFromEnum('Description','',bag_of_tags,has_absent=False))
            }

            # references
            compressed_cve={}
            compressed_cve['cve']=cve['cve']
            if 'CWEs' in cve{
                compressed_cve['CWEs']=cve['CWEs']
            }
            # references

            compressed_cve['data']=featured_cve

            if update_callback { update_callback() }
            self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),compressed_cve,'features_cve','cve',verbose=False,ignore_lock=True)
            data_size+=Utils.sizeof(compressed_cve)
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Transform\" on CVE Data...OK')
    }

    def transformOval(self,update_callback=None){
        self.logger.info('Running \"Transform\" on OVAL Data...')
        oval_data=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'flat_oval')
        verbose_frequency=1333
        iter_count=0
        data_size=0
        total_iters=oval_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'features_oval')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for oval in oval_data{
            featured_oval={}
            # _id ignore
            # references
            featured_oval['cve']=oval['CVE']
            if oval['type'].lower()=='patch'{
                featured_oval['has_patch']=1
                featured_oval['has_vuln_oval']=0
                featured_oval['patchDate']=oval['submittedDate']
            }elif oval['type'].lower()=='vulnerability'{
                featured_oval['has_patch']=0
                featured_oval['has_vuln_oval']=1
                featured_oval['ovalDate']=oval['submittedDate'] # useless?
            }else{
                raise Exception('Invalid oval type ({}) on oval {}'.format(oval['type'],oval['oval']))
            }
            featured_oval['oval']=oval['oval']
            if update_callback { update_callback() }
            self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),featured_oval,'features_oval','oval',verbose=False,ignore_lock=True)
            data_size+=Utils.sizeof(featured_oval)
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Transform\" on OVAL Data...OK')
    }

    def transformCapec(self,update_callback=None){ 
        self.logger.info('Running \"Transform\" on CAPEC Data...')
        capec_data=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'flat_capec')
        verbose_frequency=50
        iter_count=0
        data_size=0
        capecs_refs=[]
        data_count=capec_data.count()
        total_iters=data_count*2 # read fields and transform
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'features_capec')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        fields_and_values={}
        bag_of_tags={}
        extracted_tags={}
        occurences_in_doc={}
        lemmatizer=WordNetLemmatizer()
        for capec in capec_data{
            capecs_refs.append(capec['capec'])
            for k,v in capec.items(){
                if k not in fields_and_values{
                    if k!='Description'{
                        fields_and_values[k]=set()
                    }else{
                        fields_and_values[k]={}
                    }
                }
                if k not in ('_id','capec','submittedDate','modifiedDate','CWEs','Examples','Skill','Mitigations','Prerequisites','Description','Name','Taxonomy','Indicators','Reference','Resources_req'){ # non enums 
                    if k in ('Steps','Affected_Scopes','Damage','Skill_level'){ # lists of enums 
                        for el in v{
                            if type(el) is list{
                                for el2 in el{
                                    fields_and_values[k].add(el2.replace(' ','_'))
                                }
                            }else{
                                fields_and_values[k].add(el.replace(' ','_'))
                            }
                        }
                    }else{
                        if type(v) is list{
                            for i in range(len(v)){
                                if type(v[i]) is list{
                                    v[i]=tuple(v[i])
                                }
                            }
                            v=tuple(v)
                        }
                        if type(v) is tuple{
                            v=' '.join(v)
                        }
                        fields_and_values[k].add(v.replace(' ','_'))
                    }
                }elif k=='Description'{
                    v=re.findall(r"[\w']+", v) # split text into words
                    for i in range(len(v)){
                        v[i]=lemmatizer.lemmatize(v[i])
                    }
                    v=' '.join(v) # join words into text
                    fields_and_values[k][capec['capec']]=v
                    keys=FeatureGenerator.extractKeywords(v)
                    filtered_keys=[]
                    not_allowed_patterns=[r'^type$',r'^order$',r'^result$',r'^victim$',r'^attempt$',r'^goal$',r'^determine$',r'^advantage$',r'^part$',r'^case$',r'^doe$',r'^number$',r'^time$',r'^lead$',r'^accomplished$',r'^intended$',r'^fact$',r'^manner$',r'^similar$',r'^target software$',r'^target application$',r'^intent$',r'^interacting$',r'^ha$',r'^identify$',r'^configuration$',r'^behavior$',r'^provide$',r'^implementation$',r'^presence$',r'^point$',r'^variety$',r'^interpreted$',r'^man$',r'^configured$',r'^expected$',r'^achieve$',r'^product$']
                    occurences_in_doc[capec['capec']]={}
                    for key in keys{
                        insert=True
                        for pattern in not_allowed_patterns{
                            if re.match(pattern, key){
                                insert=False 
                                break
                            }
                        }
                        if insert{
                            filtered_keys.append(key)
                            occurences_in_doc[capec['capec']][key]=v.count(key)
                            occurences_in_doc[capec['capec']][key]=occurences_in_doc[capec['capec']][key] if occurences_in_doc[capec['capec']][key]>0 else 1
                            if key not in bag_of_tags{
                                bag_of_tags[key]=1
                            }else{
                                bag_of_tags[key]+=1
                            }
                        }
                    }
                    extracted_tags[capec['capec']]=filtered_keys
                }
            }
            if update_callback { update_callback() }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}%'.format((float(iter_count)/total_iters*100)))
            }
        }
        self.logger.info('Optimizing and caching data...')
        total_iters=data_count+len(capecs_refs)
        capec_data=None
        bag_of_tags_sorted=sorted(bag_of_tags.items(), key=lambda x: x[1], reverse=True)
        bag_of_tags=[]
        bag_of_tags_and_occurences={}
        min_occurrences=10
        for tag,amount in list(bag_of_tags_sorted){
            if amount>=min_occurrences{
                bag_of_tags.append(tag)
                bag_of_tags_and_occurences[tag]=amount
            }
        }
        bag_of_tags_sorted=None
        # just to visualize
        # for k,v in fields_and_values.items(){
        #     self.logger.clean(k)
        #     if k !='vendors'{
        #         for v2 in v{
        #             self.logger.clean('\t{}'.format(v2))
        #         }
        #     }elif k!='Description'{
        #         self.logger.clean('\t{}'.format(' | '.join(v)))
        #     }
        # }
        # just to visualize 
        for k,v in fields_and_values.items(){
            fields_and_values[k]=list(v)
        }
        total_docs=len(fields_and_values['Description'])
        self.logger.info('Optimized and cached data...OK')
        for capec_ref in capecs_refs{
            capec=self.mongo.findOneOnDBFromIndex(self.mongo.getProcessedDB(),'flat_capec','capec',capec_ref)
            # _id - OK
            # capec - OK
            # Name - Ok
            # Description - OK
            # CWEs - OK
            # Abstraction - OK
            # Status - OK
            # Likelihood_Of_Attack - OK
            # Typical_Severity - OK
            # Steps - OK
            # Skill_level - OK
            # Affected_Scopes - OK
            # Damage - OK
            # submittedDate - OK
            # modifiedDate - OK
            # Prerequisites - OK
            # Mitigations - OK
            # Skill - OK
            # Examples - OK
            # Taxonomy - OK
            # Indicators - OK
            # Reference - OK
            # Resources_req - OK
            featured_capec={}

            # _id ignore
            # Name ignore
            # Taxonomy ignore
            
            # references
            featured_capec['capec']=capec['capec']
            if 'CWEs' in capec{
                featured_capec['CWEs']=capec['CWEs']
            }
            # references

            # enums
            if 'Abstraction' not in capec{
                capec['Abstraction']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Abstraction',capec['Abstraction'],fields_and_values['Abstraction']))

            if 'Status' not in capec{
                capec['Status']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Status',capec['Status'],fields_and_values['Status']))

            if 'Likelihood_Of_Attack' not in capec{
                capec['Likelihood_Of_Attack']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Likelihood_Of_Attack',capec['Likelihood_Of_Attack'],fields_and_values['Likelihood_Of_Attack']))

            if 'Typical_Severity' not in capec{
                capec['Typical_Severity']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Typical_Severity',capec['Typical_Severity'],fields_and_values['Typical_Severity']))

            if 'Steps' not in capec{
                capec['Steps']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Steps',capec['Steps'],fields_and_values['Steps']))

            if 'Skill_level' not in capec{
                capec['Skill_level']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Skill_level',capec['Skill_level'],fields_and_values['Skill_level']))

            if 'Affected_Scopes' not in capec or capec['Affected_Scopes']=='Other'{
                capec['Affected_Scopes']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Affected_Scopes',capec['Affected_Scopes'],fields_and_values['Affected_Scopes']))

            if 'Damage' not in capec or capec['Damage']=='Other'{
                capec['Damage']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Damage',capec['Damage'],fields_and_values['Damage']))
            # enums

            # Dates - extract features on enrich
            if 'submittedDate' in capec{
                featured_capec['capecDate']=capec['submittedDate']
            }
            # modifiedDate ignore        
            # Dates - extract features on enrich

            # numbers
            if 'Prerequisites' not in capec{
                capec['Prerequisites']=''
            }else{
                if type(capec['Prerequisites']) is list{
                     capec['Prerequisites']=' '.join(capec['Prerequisites'])
                }
            }
            featured_capec['prerequisites_wc_log']=math.log(1+len(re.findall(r"[\w']+", capec['Prerequisites'])))

            if 'Mitigations' not in capec{
                capec['Mitigations']=''
            }else{
                if type(capec['Mitigations']) is list{
                     capec['Mitigations']=' '.join( capec['Mitigations'])
                }
            }
            featured_capec['mitigations_wc_log']=math.log(1+len(re.findall(r"[\w']+", capec['Mitigations'])))

            if 'Skill' not in capec{
                capec['Skill']=''
            }else{
                if type(capec['Skill']) is list{
                     capec['Skill']=' '.join( capec['Skill'])
                }
            }
            featured_capec['skill_wc_log']=math.log(1+len(re.findall(r"[\w']+", capec['Skill'])))

            if 'Examples' not in capec{
                capec['Examples']=''
            }else{
                if type(capec['Examples']) is list{
                     capec['Examples']=' '.join( capec['Examples'])
                }
            }
            featured_capec['examples_wc_log']=math.log(1+len(re.findall(r"[\w']+", capec['Examples'])))

            if 'Indicators' not in capec{
                capec['Indicators']=''
            }else{
                if type(capec['Indicators']) is list{
                     capec['Indicators']=' '.join( capec['Indicators'])
                }
            }
            featured_capec['indicators_wc_log']=math.log(1+len(re.findall(r"[\w']+", capec['Indicators'])))

             if 'Resources_req' not in capec{
                capec['Resources_req']=''
            }else{
                if type(capec['Resources_req']) is list{
                     capec['Resources_req']=' '.join( capec['Resources_req'])
                }
            }
            featured_capec['required_res_wc_log']=math.log(1+len(re.findall(r"[\w']+", capec['Resources_req'])))

             if 'Reference' not in capec{
                capec['Reference']=[]
            }
            featured_capec['references_count']=len(capec['Reference'])
            # numbers


            if 'Description' in capec{
                description_features=FeatureGenerator.buildFeaturesFromEnum('Description',extracted_tags[capec['capec']],bag_of_tags,has_absent=False)
                for tag in extracted_tags[capec['capec']]{
                    if tag in bag_of_tags{
                        tag_feature_name=FeatureGenerator.buildEnumKeyName('Description',tag)
                        if tag_feature_name in description_features and description_features[tag_feature_name]==1{
                            description_features[tag_feature_name]=occurences_in_doc[capec['capec']][tag]*math.log(float(total_docs)/float(bag_of_tags_and_occurences[tag]))
                        }
                    }
                }
                featured_capec=dict(featured_capec,**description_features)
            }else{
                featured_capec=dict(featured_capec,**FeatureGenerator.buildFeaturesFromEnum('Description','',bag_of_tags,has_absent=False))
            }
            featured_capec['capec']=capec['capec']


            if update_callback { update_callback() }
            self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),featured_capec,'features_capec','capec',verbose=False,ignore_lock=True)
            data_size+=Utils.sizeof(featured_capec)
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Transform\" on CAPEC Data...OK')
    }

    def transformCwe(self,update_callback=None){ 
        self.logger.info('Running \"Transform\" on CWE Data...')
        cwe_data=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'flat_cwe')
        verbose_frequency=50
        iter_count=0
        data_size=0
        cwes_refs=[]
        data_count=cwe_data.count()
        total_iters=data_count*2 # read fields and transform
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'features_cwe')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        fields_and_values={}
        bag_of_tags={}
        extracted_tags={}
        occurences_in_doc={}
        lemmatizer=WordNetLemmatizer()
        for cwe in cwe_data{
            cwes_refs.append(cwe['cwe'])
            if 'Extended_Description' in cwe{
                cwe['Description']+='\n{}'.format(cwe['Extended_Description'])
            }
            if 'Potential_Mitigations' in cwe{
                if type(cwe['Potential_Mitigations']) is list{
                    size_is=len(cwe['Potential_Mitigations'])
                    for i in range(size_is){
                        splitted=cwe['Potential_Mitigations'][i].split('\n')
                        if len(splitted)>1{
                            for j in range(len(splitted)){
                                if j==0{
                                    cwe['Potential_Mitigations'][i]=splitted[j]
                                }else{
                                    cwe['Potential_Mitigations'].append(splitted[j])
                                }
                            }
                        }
                    }
                }
            }
            for k,v in cwe.items(){
                if k not in fields_and_values{
                    if k!='Description'{
                        fields_and_values[k]=set()
                    }else{
                        fields_and_values[k]={}
                    }
                }
                if k not in ('_id','cwe','Name','Description','Extended_Description','CVEs','References','submittedDate','modifiedDate','Potential_Mitigations','Demonstrative_Examples','Category','Taxonomy','Related_Attack_Patterns','value','View','Dectection'){ # non enums
                    if k in ('Modes_Of_Introduction','Technology','Affected_Scopes','Damage','MitigationsEffectiveness','Weakness_Ordinalities','Dectection','Dectection_Effectiveness','Affected_Resources','Functional_Areas'){ # lists of enums 
                        if type(v) is list{
                            for el in v{
                                if type(el) is list{
                                    for el2 in el{
                                        if 'REALIZATION: ' not in el2 and 'OMISSION: ' not in el2{
                                            fields_and_values[k].add(el2.replace(' ','_'))
                                        }
                                    }
                                }else{
                                    if 'REALIZATION: ' not in el and 'OMISSION: ' not in el{
                                        fields_and_values[k].add(el.replace(' ','_'))
                                    }
                                }
                            }
                        }else{
                            if 'REALIZATION: ' not in v and 'OMISSION: ' not in v{
                                fields_and_values[k].add(v.replace(' ','_'))
                            }
                        }
                    }else{
                        if type(v) is list{
                            for i in range(len(v)){
                                if type(v[i]) is list{
                                    v[i]=tuple(v[i])
                                }
                            }
                            v=tuple(v)
                        }
                        fields_and_values[k].add(v.replace(' ','_'))
                    }
                }elif k=='Description'{
                    v=re.findall(r"[\w']+", v) # split text into words
                    for i in range(len(v)){
                        v[i]=lemmatizer.lemmatize(v[i])
                    }
                    v=' '.join(v) # join words into text
                    fields_and_values[k][cwe['cwe']]=v
                    keys=FeatureGenerator.extractKeywords(v)
                    filtered_keys=[]
                    not_allowed_patterns=[r'^doe$',r'^software doe$',r'^product$',r'^order$',r'^lead$',r'^result$',r'^make$',r'^weakness$',r'^ha$',r'^actor$',r'^product doe$',r'^expected$',r'^occur$',r'^ensure$',r'^issue$',r'^intended$',r'^easier$',r'^reference$',r'^making$',r'^modified$',r'^introduce$']
                    occurences_in_doc[cwe['cwe']]={}
                    for key in keys{
                        insert=True
                        for pattern in not_allowed_patterns{
                            if re.match(pattern, key){
                                insert=False 
                                break
                            }
                        }
                        if insert{
                            filtered_keys.append(key)
                            occurences_in_doc[cwe['cwe']][key]=v.count(key)
                            occurences_in_doc[cwe['cwe']][key]=occurences_in_doc[cwe['cwe']][key] if occurences_in_doc[cwe['cwe']][key]>0 else 1
                            if key not in bag_of_tags{
                                bag_of_tags[key]=1
                            }else{
                                bag_of_tags[key]+=1
                            }
                        }
                    }
                    extracted_tags[cwe['cwe']]=filtered_keys
                }
            }
            if update_callback { update_callback() }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}%'.format((float(iter_count)/total_iters*100)))
            }
        }
        self.logger.info('Optimizing and caching data...')
        total_iters=data_count+len(cwes_refs)
        cwe_data=None
        bag_of_tags_sorted=sorted(bag_of_tags.items(), key=lambda x: x[1], reverse=True)
        bag_of_tags=[]
        bag_of_tags_and_occurences={}
        min_occurrences=22
        for tag,amount in list(bag_of_tags_sorted){
            if amount>=min_occurrences{
                bag_of_tags.append(tag)
                bag_of_tags_and_occurences[tag]=amount
            }
        }
        bag_of_tags_sorted=None
        # just to visualize 
        # for k,v in fields_and_values.items(){
        #     self.logger.clean(k)
        #     if k !='Description'{
        #         for v2 in v{
        #             self.logger.clean('\t{}'.format(v2))
        #         }
        #     }else{
        #         self.logger.clean('\t{}'.format(' | '.join(bag_of_tags)))
        #     }
        # }
        # just to visualize 
        for k,v in fields_and_values.items(){
            fields_and_values[k]=list(v)
        }
        total_docs=len(fields_and_values['Description'])
        self.logger.info('Optimized and cached data...OK')
        for cwe_ref in cwes_refs{
            cwe=self.mongo.findOneOnDBFromIndex(self.mongo.getProcessedDB(),'flat_cwe','cwe',cwe_ref)
            # _id - OK
            # cwe - OK
            # Name - OK
            # Extended_Description - OK
            # CVEs - OK
            # Abstraction - OK
            # Structure - OK
            # Status - OK
            # Modes_Of_Introduction - OK
            # Likelihood_Of_Exploit - OK
            # Language - OK
            # Technology - OK
            # Affected_Scopes - OK
            # Damage - OK
            # MitigationsEffectiveness - OK
            # modifiedDate - OK
            # submittedDate - OK
            # Demonstrative_Examples - OK
            # References - OK
            # Description - OK
            # Taxonomy - OK
            # Related_Attack_Patterns - OK
            # value - OK
            # Weakness_Ordinalities - OK
            # Dectection_Effectiveness - OK
            # Affected_Resources - OK
            # Functional_Areas - OK
            # View - OK
            # Dectection - OK
            # Potential_Mitigations - OK
            # Category - OK
            
            featured_cwe={}

            # _id ignore
            # Name ignore
            # Taxonomy ignore
            # value ignore
            # View ignore
            
            # references
            featured_cwe['cwe']=cwe['cwe']
            if 'CVEs' in cwe{
                featured_cwe['CVEs']=cwe['CVEs']
            }
            if 'Related_Attack_Patterns' in cwe{
                featured_cwe['Related_Attack_Patterns']=cwe['Related_Attack_Patterns']
            }
            # references

            # enums
            if 'Abstraction' not in cwe{
                cwe['Abstraction']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Abstraction',cwe['Abstraction'],fields_and_values['Abstraction']))

            if 'Structure' not in cwe{
                cwe['Structure']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Structure',cwe['Structure'],fields_and_values['Structure']))

            if 'Status' not in cwe{
                cwe['Status']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Status',cwe['Status'],fields_and_values['Status']))

            if 'Modes_Of_Introduction' not in cwe{
                cwe['Modes_Of_Introduction']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Modes_Of_Introduction',cwe['Modes_Of_Introduction'],fields_and_values['Modes_Of_Introduction']))

            if 'Likelihood_Of_Exploit' not in cwe{
                cwe['Likelihood_Of_Exploit']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Likelihood_Of_Exploit',cwe['Likelihood_Of_Exploit'],fields_and_values['Likelihood_Of_Exploit']))

            if 'Language' not in cwe{
                cwe['Language']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Language',cwe['Language'],fields_and_values['Language']))

            if 'Technology' not in cwe{
                cwe['Technology']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Technology',cwe['Technology'],fields_and_values['Technology']))

            if 'Affected_Scopes' not in cwe{
                cwe['Affected_Scopes']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Affected_Scopes',cwe['Affected_Scopes'],fields_and_values['Affected_Scopes']))

            if 'Damage' not in cwe{
                cwe['Damage']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Damage',cwe['Damage'],fields_and_values['Damage']))

            if 'MitigationsEffectiveness' not in cwe or cwe['MitigationsEffectiveness']=='UNKNOWN'{
                cwe['MitigationsEffectiveness']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Mitigations_Effectiveness',cwe['MitigationsEffectiveness'],fields_and_values['MitigationsEffectiveness']))

            if 'Dectection_Effectiveness' not in cwe or cwe['Dectection_Effectiveness']=='Unknown'{
                cwe['Dectection_Effectiveness']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Dectection_Effectiveness',cwe['Dectection_Effectiveness'],fields_and_values['Dectection_Effectiveness']))

            if 'Weakness_Ordinalities' not in cwe {
                cwe['Weakness_Ordinalities']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Weakness_Ordinalities',cwe['Weakness_Ordinalities'],fields_and_values['Weakness_Ordinalities']))

            if 'Affected_Resources' not in cwe {
                cwe['Affected_Resources']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Affected_Resources',cwe['Affected_Resources'],fields_and_values['Affected_Resources']))

            if 'Functional_Areas' not in cwe {
                cwe['Functional_Areas']=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
            featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Functional_Areas',cwe['Functional_Areas'],fields_and_values['Functional_Areas']))
            # enums

            # Dates - extract features on enrich
            if 'submittedDate' in cwe{
                featured_cwe['cweDate']=cwe['submittedDate']
            }
            # modifiedDate ignore        
            # Dates - extract features on enrich


            # numbers
            if 'Demonstrative_Examples' not in cwe{
                cwe['Demonstrative_Examples']=''
            }else{
                if type(cwe['Demonstrative_Examples']) is list{
                     cwe['Demonstrative_Examples']=' '.join(cwe['Demonstrative_Examples'])
                }
            }
            featured_cwe['demo_examples_wc_log']=math.log(1+len(re.findall(r"[\w']+", cwe['Demonstrative_Examples'])))

            if 'Dectection' not in cwe{
                cwe['Dectection']=''
            }else{
                if type(cwe['Dectection']) is list{
                     cwe['Dectection']=' '.join(cwe['Dectection'])
                }
            }
            featured_cwe['dectection_wc_log']=math.log(1+len(re.findall(r"[\w']+", cwe['Dectection'])))

            if 'Potential_Mitigations' not in cwe{
                cwe['Potential_Mitigations']=''
            }else{
                if type(cwe['Potential_Mitigations']) is list{
                     cwe['Potential_Mitigations']=' '.join(cwe['Potential_Mitigations'])
                }
            }
            featured_cwe['pot_mitigations_wc_log']=math.log(1+len(re.findall(r"[\w']+", cwe['Potential_Mitigations'])))

            if 'Category' not in cwe{
                cwe['Category']=''
            }else{
                if type(cwe['Category']) is list{
                     cwe['Category']=' '.join(cwe['Category'])
                }
            }
            featured_cwe['category_wc_log']=math.log(1+len(re.findall(r"[\w']+", cwe['Category'])))

            if 'References' not in cwe{
                cwe['References']=[]
            }
            featured_cwe['references_count']=len(cwe['References'])
            # numbers


            if 'Description' in cwe{
                description_features=FeatureGenerator.buildFeaturesFromEnum('Description',extracted_tags[cwe['cwe']],bag_of_tags,has_absent=False)
                for tag in extracted_tags[cwe['cwe']]{
                    if tag in bag_of_tags{
                        tag_feature_name=FeatureGenerator.buildEnumKeyName('Description',tag)
                        if tag_feature_name in description_features and description_features[tag_feature_name]==1{
                            description_features[tag_feature_name]=occurences_in_doc[cwe['cwe']][tag]*math.log(float(total_docs)/float(bag_of_tags_and_occurences[tag]))
                        }
                    }
                }
                featured_cwe=dict(featured_cwe,**description_features)
            }else{
                featured_cwe=dict(featured_cwe,**FeatureGenerator.buildFeaturesFromEnum('Description','',bag_of_tags,has_absent=False))
            }

            if update_callback { update_callback() }
            self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),featured_cwe,'features_cwe','cwe',verbose=False,ignore_lock=True)
            data_size+=Utils.sizeof(featured_cwe)
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Transform\" on CWE Data...OK')
    }

    def transformExploits(self,update_callback=None){ 
        self.logger.info('Running \"Transform\" on EXPLOIT Data...')
        exploit_data=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'exploits')
        verbose_frequency=666
        iter_count=0
        data_size=0
        total_iters=exploit_data.count()
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'features_exploit')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        for exploit in exploit_data{
            featured_exploit={}
            # _id ignore
            # references
            featured_exploit['cve']=exploit['cve']
            featured_exploit['exploitDate']=Utils.changeStrDateFormat(exploit['date'],'%Y-%m-%d','%d/%m/%Y')
            
            if exploit['verified'].lower()=='true'{
                featured_exploit['exploit_was_verified']=1
            }else{
                featured_exploit['exploit_was_verified']=0
            }
            featured_exploit['has_exploit']=1

            featured_exploit['exploit']=exploit['exploit']

            if update_callback { update_callback() }
            self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),featured_exploit,'features_exploit','exploit',verbose=False,ignore_lock=True)
            data_size+=Utils.sizeof(featured_exploit)
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Transform\" on EXPLOIT Data...OK')
    }
    
    def enrichData(self,update_callback=None){
        self.references=self.mongo.loadReferences()
        self.logger.info('Running \"Enrich\" on data...')
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'full_dataset')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        verbose_frequency=666
        data_size=0
        cves_refs=[]
        iter_count=1
        total_iters=len(self.references['cve'])
        for cve_ref in self.references['cve']{
            cve=self.mongo.findOneOnDBFromIndex(self.mongo.getProcessedDB(),'features_cve','cve','CVE-{}'.format(cve_ref))
            if cve{
                cve_id=cve['cve']
                full_cve={}
                cve_labels={}
                CWEs=[]
                lastModifiedDate=None
                interimDate=None
                proposedDate=None
                assignedDate=None
                publishedDate=None
                if 'CWEs' in cve{
                    CWEs=cve['CWEs']
                }
                cve=cve['data']
                for k,v in cve.items(){
                    if 'exploits_weaponized' in k{
                        cve_labels[k]=v
                    }elif k=='interimDate'{
                        interimDate=v
                    }elif k=='proposedDate'{
                        proposedDate=v
                    }elif k=='assignedDate'{
                        assignedDate=v
                    }elif k=='publishedDate'{
                        publishedDate=v
                    }elif k=='lastModifiedDate'{
                        lastModifiedDate=v
                    }elif k!='_id'{
                        full_cve[k]=v
                    }
                }
                capecs=[]
                cweDate=[]
                merged_cwes={}
                for cwe in CWEs{
                    cwe=self.mongo.findOneOnDBFromIndex(self.mongo.getProcessedDB(),'features_cwe','cwe',cwe)
                    if cwe {
                        for k,v in cwe.items(){
                            k_cwe='cwe_{}'.format(k)
                            if '_ENUM_' in k{
                                if k_cwe not in merged_cwes{
                                    merged_cwes[k_cwe]=v
                                }else{
                                    if merged_cwes[k_cwe]==0{
                                        merged_cwes[k_cwe]=v
                                    }elif merged_cwes[k_cwe]==1 and (v==1 or v==0) {
                                        merged_cwes[k_cwe]=1
                                    }else{
                                        merged_cwes[k_cwe]+=v
                                    }
                                }
                            }elif k=='cweDate'{
                                cweDate.append(v)
                            }elif k=='Related_Attack_Patterns'{
                                for el in v{
                                    capecs.append(el)
                                }
                            }elif k not in ('_id','CVEs','cwe'){
                                if k_cwe not in merged_cwes{
                                    merged_cwes[k_cwe]=v
                                }else{
                                    merged_cwes[k_cwe]+=v
                                }
                            }
                        }
                    }
                }
                full_cve=dict(full_cve,**merged_cwes)
                capecDate=[]
                merged_capecs={}
                for capec in capecs{
                    capec=self.mongo.findOneOnDBFromIndex(self.mongo.getProcessedDB(),'features_capec','capec',capec)
                    if capec{
                        for k,v in capec.items(){
                            k_capec='capec_{}'.format(k)
                            if '_ENUM_' in k{
                                if k_capec not in merged_capecs{
                                    merged_capecs[k_capec]=v
                                }else{
                                    if merged_capecs[k_capec]==0{
                                        merged_capecs[k_capec]=v
                                    }elif merged_capecs[k_capec]==1 and (v==1 or v==0) {
                                        merged_capecs[k_capec]=1
                                    }else{
                                        merged_capecs[k_capec]+=v
                                    }
                                }
                            }elif k=='capecDate'{
                                capecDate.append(v)
                            }elif k not in ('_id','CWEs','capec'){
                                if k_capec not in merged_capecs{
                                    merged_capecs[k_capec]=v
                                }else{
                                    merged_capecs[k_capec]+=v
                                }
                            }
                        }
                    }
                }
                full_cve=dict(full_cve,**merged_capecs)
                query={'cve':cve_id}
                ovals=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'features_oval',query=query)
                ovalDate=[]
                patchDate=[]
                merged_ovals={}
                for oval in ovals{
                    for k,v in oval.items(){
                        k_oval='oval_{}'.format(k)
                        if 'has_' in k{
                            if k_oval not in merged_ovals{
                                merged_ovals[k_oval]=v
                            }else{
                                if merged_ovals[k_oval]==0{
                                    merged_ovals[k_oval]=v
                                }elif merged_ovals[k_oval]==1 and (v==1 or v==0) {
                                    merged_ovals[k_oval]=1
                                }else{
                                    merged_ovals[k_oval]+=v
                                }
                            }
                        }elif k=='ovalDate'{
                            ovalDate.append(v)
                        }elif k=='patchDate'{
                            patchDate.append(v)
                        }
                    }
                }
                full_cve=dict(full_cve,**merged_ovals)
                query={'cve':cve_id.split('CVE-')[1]}
                exploits=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'features_exploit',query=query)
                exploitDate=[]
                cve_labels['exploits_has']=0
                cve_labels['exploits_verified']=0
                for exploit in exploits{
                    if 'exploitDate' in exploit{
                        exploitDate.append(exploit['exploitDate'])
                    }
                    if 'has_exploit' in exploit and exploit['has_exploit']==1{
                        cve_labels['exploits_has']=1
                    }
                    if 'exploit_was_verified' in exploit and exploit['exploit_was_verified']==1{
                        cve_labels['exploits_verified']=1
                    }
                }
                if len(cweDate)==0{
                    cweDate=None
                }else{
                    finalDate=cweDate[0]
                    for date in cweDate{
                        if Utils.isFirstStrDateOldest(date,finalDate,'%d/%m/%Y'){ # oldest
                            finalDate=date
                        }
                    }
                    cweDate=finalDate
                }
                if len(capecDate)==0{
                    capecDate=None
                }else{
                    finalDate=capecDate[0]
                    for date in capecDate{
                        if Utils.isFirstStrDateOldest(date,finalDate,'%d/%m/%Y'){ # oldest
                            finalDate=date
                        }
                    }
                    capecDate=finalDate
                }
                if len(ovalDate)==0{
                    ovalDate=None
                }else{
                    finalDate=ovalDate[0]
                    for date in ovalDate{
                        if Utils.isFirstStrDateOldest(date,finalDate,'%d/%m/%Y'){ # oldest
                            finalDate=date
                        }
                    }
                    ovalDate=finalDate
                }
                if len(patchDate)==0{
                    patchDate=None
                }else{
                    finalDate=patchDate[0]
                    for date in patchDate{
                        if Utils.isFirstStrDateOldest(date,finalDate,'%d/%m/%Y'){ # oldest
                            finalDate=date
                        }
                    }
                    patchDate=finalDate
                }
                if len(exploitDate)==0{
                    exploitDate=None
                }else{
                    finalDate=exploitDate[0]
                    for date in exploitDate{
                        if Utils.isFirstStrDateOldest(date,finalDate,'%d/%m/%Y'){ # oldest
                            finalDate=date
                        }
                    }
                    exploitDate=finalDate
                }
                # cweDate
                # capecDate
                # ovalDate
                # patchDate
                # exploitDate
                # lastModifiedDate
                # interimDate
                # proposedDate
                # assignedDate
                # publishedDate
                # publishedDate-proposedDate (td-t0)
                # patchDate-publishedDate (tp-td)
                # exploitDate-patchDate (te-tp)
                # exploitDate-publishedDate (te-td)
                # exploitDate-proposedDate (te-t0)
                # now()-proposedDate (age)
                cve_labels['exploits_delta_days']=0
                full_cve['delta_days_cwe']=0
                full_cve['delta_days_capec']=0
                full_cve['delta_days_oval']=0
                full_cve['delta_days_patch']=0
                full_cve['delta_days_proposed']=0
                full_cve['delta_days_assigned']=0
                if publishedDate {
                    if exploitDate {
                        cve_labels['exploits_delta_days']=Utils.daysBetweenStrDate(publishedDate,exploitDate,'%d/%m/%Y')
                    }
                    if cweDate {
                        full_cve['delta_days_cwe']=Utils.daysBetweenStrDate(publishedDate,cweDate,'%d/%m/%Y')
                    }
                    if capecDate {
                        full_cve['delta_days_capec']=Utils.daysBetweenStrDate(publishedDate,capecDate,'%d/%m/%Y')
                    }
                    if ovalDate {
                        full_cve['delta_days_oval']=Utils.daysBetweenStrDate(publishedDate,ovalDate,'%d/%m/%Y')
                    }
                    if patchDate {
                        full_cve['delta_days_patch']=Utils.daysBetweenStrDate(publishedDate,patchDate,'%d/%m/%Y')
                    }
                    if proposedDate {
                        full_cve['delta_days_proposed']=Utils.daysBetweenStrDate(publishedDate,proposedDate,'%d/%m/%Y')
                    }
                    if assignedDate {
                        full_cve['delta_days_assigned']=Utils.daysBetweenStrDate(publishedDate,assignedDate,'%d/%m/%Y')
                    }
                }

                full_cve=dict(cve_labels,**full_cve)
                compressed_cve={}
                compressed_cve['cve']=cve_id
                compressed_cve['data']=full_cve
                if update_callback { update_callback() }
                self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),compressed_cve,'full_dataset','cve',verbose=False,ignore_lock=True)
                data_size+=Utils.sizeof(compressed_cve)
                if iter_count%verbose_frequency==0{
                    lock.refresh()
                    self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
                }
            }   
            iter_count+=1
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Enrich\" on data...OK')
    }

    def analyzeFullDataset(self,update_callback=None){
        self.logger.info('Running \"Analyze\" on data...')
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'statistics')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        dataset=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'full_dataset')
        verbose_frequency=666
        iter_count=0
        total_entries=dataset.count()
        total_iters=total_entries
        data_size=0
        field_precesence={}
        # field name -> value -> amount
        field_values_compressed={} 
        # field name -> value -> amount
        field_vendor_values_compressed={} 
        for data in dataset{
            data=data['data']
            for k,v in data.items(){
                if k!='_id'{
                    v=str(v)
                    if v.replace('.','',1).isdigit(){ # check if is float
                        if k not in field_precesence{
                            field_precesence[k]=1
                        }else{
                            field_precesence[k]+=1
                        }
                        if 'vendor_ENUM_' not in k {
                            if k not in field_values_compressed{
                                field_values_compressed[k]={}
                            }
                            if v not in field_values_compressed[k]{
                                field_values_compressed[k][v]=1
                            }else{
                                field_values_compressed[k][v]+=1
                            }
                        }else{
                            if k not in field_vendor_values_compressed{
                                field_vendor_values_compressed[k]={}
                            }
                            if v not in field_vendor_values_compressed[k]{
                                field_vendor_values_compressed[k][v]=1
                            }else{
                                field_vendor_values_compressed[k][v]+=1
                            }
                        }
                    }
                }
            }
            if update_callback { update_callback() }
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                data_size=Utils.sizeof(field_precesence)+Utils.sizeof(field_values_compressed)+Utils.sizeof(field_vendor_values_compressed)
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        total_entries={'total_entries':total_entries}
        min_values={}
        max_values={}
        for k,dic in field_values_compressed.items() {
            for v,_ in dic.items() {
                v=float(v)
                if k not in min_values {
                    min_values[k]=v
                }elif v<min_values[k]{
                    min_values[k]=v
                }
                if k not in max_values {
                    max_values[k]=v
                }elif v>max_values[k]{
                    max_values[k]=v
                }
            }
        }
        # wrapping the document inside 'data' avoid slowness on mongo express
        field_precesence={'data':field_precesence}
        field_values_compressed={'data':field_values_compressed}
        field_vendor_values_compressed={'data':field_vendor_values_compressed}
        min_values={'data':min_values}
        max_values={'data':max_values}
        total_entries={'data':total_entries}
        field_precesence['__name__']='field_precesence'
        field_values_compressed['__name__']='field_values_compressed'
        field_vendor_values_compressed['__name__']='field_vendor_values_compressed'
        min_values['__name__']='min_values'
        max_values['__name__']='max_values'
        total_entries['__name__']='total_entries'
        self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),total_entries,'statistics','__name__',verbose=False,ignore_lock=True)
        self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),field_values_compressed,'statistics','__name__',verbose=False,ignore_lock=True)
        self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),min_values,'statistics','__name__',verbose=False,ignore_lock=True)
        self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),max_values,'statistics','__name__',verbose=False,ignore_lock=True)
        self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),field_precesence,'statistics','__name__',verbose=False,ignore_lock=True)
        self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),field_vendor_values_compressed,'statistics','__name__',verbose=False,ignore_lock=True)
        data_size=Utils.sizeof(field_precesence)+Utils.sizeof(field_values_compressed)+Utils.sizeof(field_vendor_values_compressed)+Utils.sizeof(min_values)+Utils.sizeof(max_values)+Utils.sizeof(total_entries)
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Analyze\" on data...OK')
    }

    def filterAndNormalizeFullDataset(self,update_callback=None){
        self.logger.info('Running \"Filter and Normalize\" on data...')
        lock=self.mongo.getLock(self.mongo.getProcessedDB(),'dataset')
        while self.mongo.checkIfCollectionIsLocked(lock=lock){
            time.sleep(1)
        }
        lock.acquire()
        threshold_presence=.8
        minimum_vendor_occurrences=33
        total_entries=None
        field_values_compressed=None
        min_values=None
        max_values=None
        field_precesence=None
        field_vendor_values_compressed=None
        metadata=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'statistics')
        for data in metadata{
            if data['__name__']=='total_entries'{
                total_entries=data['data']['total_entries']
            }elif data['__name__']=='field_values_compressed'{
                field_values_compressed=data['data']
            }elif data['__name__']=='min_values'{
                min_values=data['data']
            }elif data['__name__']=='max_values'{
                max_values=data['data']
            }elif data['__name__']=='field_precesence'{
                field_precesence=data['data']
            }elif data['__name__']=='field_vendor_values_compressed'{
                field_vendor_values_compressed=data['data']
            }
        }

        list_of_features_to_be_normalized=[]
        features_to_be_normalized={}
        for k,_ in min_values.items(){
            min_value=min_values[k]
            max_value=max_values[k]
            if min_value!=max_value and (min_value not in (-1,0,1) or  max_value not in (-1,0,1)){
                features_to_be_normalized[k]=(float(min_value),float(max_value-min_value)) # offset, multiplier
                list_of_features_to_be_normalized.append(k)
            }
        }
        features_to_be_removed=[]
        features_to_be_filled_with_zero=[]
        for k,v in field_values_compressed.items() {
            if len(v)<=1{
                features_to_be_removed.append(k)
            }
            if field_precesence[k]!=total_entries{
                if threshold_presence*total_entries>field_precesence[k]{
                    features_to_be_removed.append(k)
                }else{
                    features_to_be_filled_with_zero.append(k)
                }
            }
        }
        vendor_features_to_became_absent=['vendor_ENUM_-']
        for k,v in field_vendor_values_compressed.items() {
            if len(v)<=1 or v['1']<minimum_vendor_occurrences{
                vendor_features_to_became_absent.append(k)
            }
        }

        features_to_be_removed.append('_id') # random generation
        features_to_be_removed.append('cve') # outside features
        dataset=self.mongo.findAllOnDB(self.mongo.getProcessedDB(),'full_dataset').sort('cve',1)  
        verbose_frequency=666
        iter_count=0
        total_iters=dataset.count()
        data_size=0

        list_of_features_to_be_normalized.sort()
        features_to_be_removed.sort()
        vendor_features_to_became_absent.sort()
        list_of_features_to_be_normalized=tuple(list_of_features_to_be_normalized)
        features_to_be_removed=tuple(features_to_be_removed)
        vendor_features_to_became_absent=tuple(vendor_features_to_became_absent)
        for data in dataset{
            features={}
            labels={}
            cve=data['cve']
            data=data['data']
            for el in features_to_be_filled_with_zero {
                features[el]=0
            }
            for k,v in data.items(){
                if not Utils.binarySearch(features_to_be_removed,k){
                    if Utils.binarySearch(list_of_features_to_be_normalized,k){
                        scaler=features_to_be_normalized[k]
                        if 'exploits_' in k {
                            labels[k]=(float(v)-scaler[0])/scaler[1]
                        }else{
                            features[k]=(float(v)-scaler[0])/scaler[1]
                        }
                    }elif k.startswith('vendor_ENUM') and Utils.binarySearch(vendor_features_to_became_absent,k){
                        features['vendor_ENUM_{}'.format(FeatureGenerator.ABSENT_FIELD_FOR_ENUM.lower())]=1
                    }else{
                        if 'exploits_' in k {
                            labels[k]=v
                        }else{
                            features[k]=v
                        }
                    }
                }
            }
            entry={'index':iter_count,'cve':cve,'features':features,'labels':labels}
            if update_callback { update_callback() }
            self.mongo.insertOneOnDB(self.mongo.getProcessedDB(),entry,'dataset','cve',verbose=False,ignore_lock=True)
            data_size+=Utils.sizeof(entry)
            iter_count+=1
            if iter_count%verbose_frequency==0{
                lock.refresh()
                self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
            }
        }
        self.logger.verbose('Percentage done {:.2f}% - Total data size: {}'.format((float(iter_count)/total_iters*100),Utils.bytesToHumanReadable(data_size)))
        lock.release()
        self.logger.info('Runned \"Filter and Normalize\" on data...OK')
    }

    def runPipeline(self,update_callback=None){
        self.mergeCve(update_callback=update_callback)
        self.flatternAndSimplifyCve(update_callback=update_callback)
        self.flatternAndSimplifyOval(update_callback=update_callback)
        self.flatternAndSimplifyCapec(update_callback=update_callback)
        self.flatternAndSimplifyCwe(update_callback=update_callback)
        self.filterExploits(update_callback=update_callback)
        self.transformCve(update_callback=update_callback)
        self.transformOval(update_callback=update_callback)
        self.transformCapec(update_callback=update_callback)
        self.transformCwe(update_callback=update_callback)
        self.transformExploits(update_callback=update_callback)
        self.enrichData(update_callback=update_callback)
        self.analyzeFullDataset(update_callback=update_callback)
        self.filterAndNormalizeFullDataset(update_callback=update_callback)
    }

     def loopOnQueue(self){
        while True{
            job=self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].next()
            if job is not None{
                payload=job.payload
                task=payload['task']
                try{
                    self.logger.info('Running job {}-{}...'.format(task,job.job_id))
                    if task=='Run Pipeline'{
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Merge'})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Flattern and Simplify','args':{'type':'CVE'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Flattern and Simplify','args':{'type':'OVAL'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Flattern and Simplify','args':{'type':'CAPEC'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Flattern and Simplify','args':{'type':'CWE'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Filter Exploits'})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Transform','args':{'type':'CVE'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Transform','args':{'type':'OVAL'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Transform','args':{'type':'CAPEC'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Transform','args':{'type':'CWE'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Transform','args':{'type':'EXPLOITS'}})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Enrich'})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Analyze'})
                        self.mongo.getQueues()[MongoDB.QUEUE_COL_PROCESSOR_NAME].put({'task': 'Filter and Normalize'})
                    }elif task=='Merge'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getRawDB(),['CVE_MITRE','CVE_NVD','CVE_DETAILS']){
                            self.logger.warn('Returning {} job to queue, because it does not have its requirements fulfilled'.format(task))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.mergeCve(update_callback=lambda: job.progress())
                        }
                    }elif task=='Flattern and Simplify' and payload['args']['type']=='CVE'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['merged_cve']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.flatternAndSimplifyCve(update_callback=lambda: job.progress())
                        }
                    }elif task=='Flattern and Simplify' and payload['args']['type']=='OVAL'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getRawDB(),['OVAL']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.flatternAndSimplifyOval(update_callback=lambda: job.progress())
                        }
                    }elif task=='Flattern and Simplify' and payload['args']['type']=='CAPEC'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getRawDB(),['CAPEC_MITRE']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.flatternAndSimplifyCapec(update_callback=lambda: job.progress())
                        }
                    }elif task=='Flattern and Simplify' and payload['args']['type']=='CWE'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getRawDB(),['CWE_MITRE']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.flatternAndSimplifyCwe(update_callback=lambda: job.progress())
                        }
                    }elif task=='Filter Exploits'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getRawDB(),['EXPLOIT_DB']){
                            self.logger.warn('Returning {} job to queue, because it does not have its requirements fulfilled'.format(task))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.filterExploits(update_callback=lambda: job.progress())
                        }
                    }elif task=='Transform' and payload['args']['type']=='CVE'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['flat_cve']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.transformCve(update_callback=lambda: job.progress())
                        }
                    }elif task=='Transform' and payload['args']['type']=='OVAL'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['flat_oval']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.transformOval(update_callback=lambda: job.progress())
                        }
                    }elif task=='Transform' and payload['args']['type']=='CAPEC'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['flat_capec']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.transformCapec(update_callback=lambda: job.progress())
                        }
                    }elif task=='Transform' and payload['args']['type']=='CWE'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['flat_cwe']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.transformCwe(update_callback=lambda: job.progress())
                        }
                    }elif task=='Transform' and payload['args']['type']=='EXPLOITS'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['exploits']){
                            self.logger.warn('Returning {}-{} job to queue, because it does not have its requirements fulfilled'.format(task,payload['args']['type']))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.transformExploits(update_callback=lambda: job.progress())
                        }
                    }elif task=='Enrich'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['features_capec','features_cve','features_cwe','features_exploit','features_oval']){
                            self.logger.warn('Returning {} job to queue, because it does not have its requirements fulfilled'.format(task))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.enrichData(update_callback=lambda: job.progress())
                        }
                    }elif task=='Analyze'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['full_dataset']){
                            self.logger.warn('Returning {} job to queue, because it does not have its requirements fulfilled'.format(task))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.analyzeFullDataset(update_callback=lambda: job.progress())
                        }
                    }elif task=='Filter and Normalize'{
                        if not self.mongo.checkIfListOfCollectionsExistsAndItsNotLocked(self.mongo.getProcessedDB(),['full_dataset','statistics']){
                            self.logger.warn('Returning {} job to queue, because it does not have its requirements fulfilled'.format(task))
                            job.put_back()
                            job=None
                            time.sleep(20)
                        }else{
                            time.sleep(10)
                        }
                        if job{
                            self.filterAndNormalizeFullDataset(update_callback=lambda: job.progress())
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