#!/bin/python
# -*- coding: utf-8 -*-

from Utils import Utils

class FeatureGenerator(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    ABSENT_FIELD_FOR_ENUM='absent'
    rake=None

    @staticmethod
    def buildFeaturesFromEnum(field_name,value,every_value,has_absent=True){
        output={}
        if has_absent{
            every_value.append(FeatureGenerator.ABSENT_FIELD_FOR_ENUM)
            if value is None{
                value=FeatureGenerator.ABSENT_FIELD_FOR_ENUM
            }
        }
        if type(value) is not list{
            value=[value]
        }
        for enum in every_value{
            presence=1 if enum.replace(' ','_') in value else 0
            output[FeatureGenerator.buildEnumKeyName(field_name,enum)]=presence
        }
        return output
    }

    @staticmethod
    def buildEnumKeyName(field_name,value){
        return '{}_ENUM_{}'.format(field_name.lower(),value.lower().replace('.','_'))
    }

    @staticmethod
    def compressListOfLists(lofl,unique=False){
        if unique{
            out=set()
        }else{
            out=[]
        }
        for l in lofl{
            if type(l) is list{
                for el in l{
                    if unique{
                        out.add(el)
                    }else{
                        out.append(el)
                    }
                }
            }else{
                if unique{
                    out.add(l)
                }else{
                    out.append(l)
                }
            }
        }
        if unique{
            out=list(out)
        }
        return out
    }

    @staticmethod
    def extractKeywords(text){
        import RAKE
        if not FeatureGenerator.rake{
            FeatureGenerator.rake=RAKE.Rake('res/SmartStoplist.txt')
        }
        keywords=FeatureGenerator.rake.run(text)
        only_words=[]
        for word,_ in keywords{
            only_words.append(word)
        }
        return only_words#, keywords
        # from rake_nltk import Rake
        # max_key_words=10
        # rake=Rake(max_length=3)
        # rake.extract_keywords_from_text(text)
        # return rake.get_ranked_phrases()[:max_key_words]
    }

}