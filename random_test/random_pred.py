import random as rd
import numpy as np
import time
import sys

def genDataset(neg,pos,n=3):
	dataset=[]
	for _ in range(pos):
		dataset.append(1)
	for _ in range(neg):
		dataset.append(0)
	for _ in range(n):
		rd.shuffle(dataset)
	return dataset


def predict(size):
	return np.random.choice(a=[0,1], size=(size,))

def seedRandom(n=7):
	rd.seed(rd.randint(-666,66)+time.time())
	for _ in range(n):
		rd.seed(rd.randint(0,sys.maxsize))
	try:
		np.random.seed(rd.randint(-666,66)+time.time())
		for _ in range(n):
			np.random.seed(rd.randint(0,sys.maxsize))
	except:
		pass

def statisticalAnalysis(predictions,corrects):
	size = len(predictions)
	if(len(corrects)<size):
		size=len(corrects)
	total=0.0
	hits=0.0
	true_negative=0
	true_positive=0
	false_negative=0
	false_positive=0
	wrong=0
	pos_count=0
	for i in range(size):
		cur_pred=predictions[i]
		cur_correct=corrects[i]
		equal=True
		positive=True
		if(cur_pred!=cur_correct):
			equal=False
		
		if(cur_correct==0):
			positive=False
		total+=1
		if(equal):
			hits+=1
			if(positive):
				true_positive+=1
				pos_count+=1
			else:
				true_negative+=1
		else:
			wrong+=1
			if(positive):
				false_negative+=1  # yes, its correct
				pos_count+=1
			else:
				false_positive+=1  # yes, its correct
	stats={}
	stats['accuracy']=float(hits)/float(total)
	A=true_positive+false_positive
	if A!=0:
		stats['precision']=float(true_positive)/float(A)
	else:
		stats['precision']=0
	B=true_positive+false_negative
	if B!=0:
		stats['recall']=float(true_positive)/float(B)
	else:
		stats['recall']=0
	C=stats['precision']+stats['recall']
	if C!=0:
		stats['f1_score']=2*(stats['precision']*stats['recall'])/(C)
		if stats['f1_score'] > 1 :
			print('ERROR - F1 Score higher than 1')
	else:
		stats['f1_score']=0
	return stats

def printDict(the_dict,name='Some dict'):
	print(f'{name}:')
	for k,v in the_dict.items():
		print(f'\t{k}: {v}')

def sumDicts(a,b):
	sum_d={}
	for k,v in a.items():
		if k in b:
			sum_d[k] = v+b[k]
		else:
			sum_d[k] = v
	return sum_d

def divDict(a,val):
	divided={}
	for k,v in a.items():
		divided[k]=v/val
	return divided

seedRandom()

amount_of_tests = 100
tests = [
	{'name': 'All data', 'neg':135677, 'pos':21341},
	{'name': '1999', 'neg':1176, 'pos':403},
	{'name': '2000', 'neg':744, 'pos':498},
	{'name': '2001', 'neg':1122, 'pos':434},
	{'name': '2002', 'neg':1790, 'pos':600},
	{'name': '2003', 'neg':1081, 'pos':496},
	{'name': '2004', 'neg':1896, 'pos':811},
	{'name': '2005', 'neg':3527, 'pos':1234},
	{'name': '2006', 'neg':4731, 'pos':2404},
	{'name': '2007', 'neg':4267, 'pos':2305},
	{'name': '2008', 'neg':4148, 'pos':3012},
	{'name': '2009', 'neg':3277, 'pos':1724},
	{'name': '2010', 'neg':3918, 'pos':1255},
	{'name': '2011', 'neg':4290, 'pos':524},
	{'name': '2012', 'neg':5097, 'pos':730},
	{'name': '2013', 'neg':6045, 'pos':585},
	{'name': '2014', 'neg':8242, 'pos':606},
	{'name': '2015', 'neg':7929, 'pos':590},
	{'name': '2016', 'neg':9875, 'pos':478},
	{'name': '2017', 'neg':15402, 'pos':993},
	{'name': '2018', 'neg':15690, 'pos':855},
	{'name': '2019', 'neg':15706, 'pos':612},
	{'name': '2020', 'neg':15088, 'pos':218},
	{'name': '50/50', 'neg':10000, 'pos':10000},
]
tests_dict_base = {}
for el in tests:
	tests_dict_base[el['name']]=el

tests.append({'name': 'HDL vs ICNN', 'neg':tests_dict_base['2019']['neg'], 'pos':tests_dict_base['2019']['pos']})
tests.append({'name': 'vs CVSS+', 'neg':tests_dict_base['2018']['neg'], 'pos':tests_dict_base['2018']['pos']})
tests.append({'name': 'vs Improving', 'neg':tests_dict_base['2017']['neg']+tests_dict_base['2018']['neg'], 'pos':tests_dict_base['2017']['pos']+tests_dict_base['2018']['pos']})
tests.append({'name': 'vs Fast Embed', 'neg':tests_dict_base['2017']['neg']+tests_dict_base['2018']['neg'], 'pos':tests_dict_base['2017']['pos']+tests_dict_base['2018']['pos']})

for test in tests:
	ground_truth = genDataset(test['neg'],test['pos'])
	stats={}
	for _ in range(amount_of_tests):
		pred = predict(len(ground_truth))
		stats = sumDicts(statisticalAnalysis(pred,ground_truth),stats)
	stats = divDict(stats,amount_of_tests)
		
	printDict(stats,test['name'])
	print(f'\tPositive %: {round((test["pos"]/len(ground_truth))*100,2)}')
	print()
