#!/bin/python

from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB
from SearchSpace import SearchSpace
from HallOfFame import HallOfFame
from StandardNeuralNetwork import StandardNeuralNetwork
from EnhancedNeuralNetwork import EnhancedNeuralNetwork
from NeuralNetwork import NeuralNetwork
from PopulationManager import PopulationManager
from StandardGeneticAlgorithm import StandardGeneticAlgorithm
from EnhancedGeneticAlgorithm import EnhancedGeneticAlgorithm
from Hyperparameters import Hyperparameters
from Enums import CrossValidation,Metric,LabelEncoding,GeneticAlgorithmType,NodeType,Loss,Optimizers,NeuralNetworkType
from Genome import Genome
from Dataset import Dataset

class Core(object){
	# 'Just':'to fix vscode coloring':'when using pytho{\}'

	LOGGER=Logger.DEFAULT()
	FREE_MEMORY_MANUALLY=True
	CACHE_WEIGHTS=True
	STORE_GEN_POP_ONLY_ON_LAST=False
	WRITE_POPULATION_WEIGHTS=False
	K_FOLDS=10
	ROLLING_FORECASTING_ORIGIN_MIN_PERCENTAGE=.5
	FIXED_VALIDATION_PERCENT=.2
	THRESHOLD=0.5

	def __init__(self, mongo, logger){
		Core.LOGGER=logger
		self.mongo=mongo
		NeuralNetwork.CLASSES_THRESHOLD=Core.THRESHOLD
	}

	def runGeneticSimulation(self,simulation_id){
		Core.LOGGER.info('Running genetic simulation {}...'.format(simulation_id))
		Core.LOGGER.info('Loading simulation...')
		genetic_metadata=self.fetchGeneticSimulationData(simulation_id)
		environment_name=genetic_metadata[0]
		cve_years=genetic_metadata[1]
		train_data_limit=genetic_metadata[2]
		hall_of_fame_id=genetic_metadata[3] 
		population_id=genetic_metadata[4]
		population_start_size=genetic_metadata[5]
		max_gens=genetic_metadata[6]
		max_age=genetic_metadata[7]
		max_children=genetic_metadata[8] 
		mutation_rate=genetic_metadata[9] 
		recycle_rate=genetic_metadata[10] 
		sex_rate=genetic_metadata[11]
		max_notables=genetic_metadata[12]
		cross_validation=genetic_metadata[13]
		metric_mode=genetic_metadata[14]
		algorithm=genetic_metadata[15] 
		label_type=genetic_metadata[16] 
		nn_type=genetic_metadata[17] 
		search_maximum=metric_mode!=Metric.RAW_LOSS
		Core.LOGGER.info('Genetic metadata')
		Core.LOGGER.info('\t{}: {}'.format('environment_name',environment_name))
		Core.LOGGER.info('\t{}: {}'.format('cve_years',cve_years))
		Core.LOGGER.info('\t{}: {}'.format('train_data_limit',train_data_limit))
		Core.LOGGER.info('\t{}: {}'.format('hall_of_fame_id',hall_of_fame_id))
		Core.LOGGER.info('\t{}: {}'.format('population_id',population_id))
		Core.LOGGER.info('\t{}: {}'.format('population_start_size',population_start_size))
		Core.LOGGER.info('\t{}: {}'.format('max_gens',max_gens))
		Core.LOGGER.info('\t{}: {}'.format('max_age',max_age))
		Core.LOGGER.info('\t{}: {}'.format('max_children',max_children))
		Core.LOGGER.info('\t{}: {}'.format('mutation_rate',mutation_rate))
		Core.LOGGER.info('\t{}: {}'.format('recycle_rate',recycle_rate))
		Core.LOGGER.info('\t{}: {}'.format('sex_rate',sex_rate))
		Core.LOGGER.info('\t{}: {}'.format('max_notables',max_notables))
		Core.LOGGER.info('\t{}: {}'.format('cross_validation',cross_validation))
		Core.LOGGER.info('\t{}: {}'.format('metric_mode',metric_mode))
		Core.LOGGER.info('\t{}: {}'.format('algorithm',algorithm))
		Core.LOGGER.info('\t{}: {}'.format('label_type',label_type))
		Core.LOGGER.info('\t{}: {}'.format('nn_type',nn_type))
		Core.LOGGER.info('\t{}: {}'.format('search_maximum',search_maximum))
		Genome.CACHE_WEIGHTS=Core.CACHE_WEIGHTS
		Core.LOGGER.info('Loaded simulation...OK')
		Core.LOGGER.info('Loading search space...')
		search_space,output_layer_node_type,multiple_networks=self.fetchEnvironmentDataV2(environment_name,metric_mode,label_type)
		Core.LOGGER.info('\t{}: {}'.format('multiple_networks',multiple_networks))
		Core.LOGGER.info('Loaded search space...OK')
		Core.LOGGER.info('Loading dataset...')
		if multiple_networks {
			train_data_ids,train_features,train_labels=self.loadDatasetMultiNet(cve_years,train_data_limit)
		}else{
			train_data_ids,train_features,train_labels=self.loadDataset(cve_years,train_data_limit)
		}
		train_features,train_labels=Dataset.balanceDataset(train_features,train_labels,multiple_networks)
		train_labels,labels_equivalence=Dataset.encodeDatasetLabels(train_labels,label_type)
		# train_features,scale=Dataset.normalizeDatasetFeatures(train_features) # already normalized
		Core.LOGGER.info('Loaded dataset...OK')
		search_space=Genome.enrichSearchSpace(search_space,multi_net_enhanced_nn=multiple_networks)
		Core.LOGGER.info('\t{}: {}'.format('output_layer_node_type',output_layer_node_type))
		Core.LOGGER.info('\t{}: {}'.format('multiple_networks',multiple_networks))
		Core.LOGGER.multiline(str(search_space))
		def train_callback(genome){
			nonlocal train_features,train_labels,cross_validation,output_layer_node_type,multiple_networks,nn_type
			preserve_weights=False # TODO fix when true, to avoid nan outputs
			if multiple_networks {
				input_size=[len(train_features[i][0]) for i in range(len(train_features))]
			}else{
				input_size=len(train_features[0])
			}
			output_size=len(train_labels[0])
			hyperparameters=genome.toHyperparameters(output_size,output_layer_node_type,multi_net_enhanced_nn=multiple_networks)
			search_maximum=hyperparameters.monitor_metric!=Metric.RAW_LOSS
			if nn_type==NeuralNetworkType.ENHANCED or multiple_networks{
				nn=EnhancedNeuralNetwork(hyperparameters,name='core_gen_{}'.format(genome.id),verbose=False)
			}else{
				nn=StandardNeuralNetwork(hyperparameters,name='core_gen_{}'.format(genome.id),verbose=False)
			}
			nn.buildModel(input_size=input_size)
        	nn.saveModelSchemaToFile('population_nets')
			if preserve_weights {
				nn.setWeights(genome.getWeights())
			}
			if cross_validation==CrossValidation.NONE{
				nn.trainNoValidation(train_features,train_labels)
			}elif cross_validation==CrossValidation.ROLLING_FORECASTING_ORIGIN{
				nn.trainRollingForecast(train_features,train_labels,min_size_percentage=Core.ROLLING_FORECASTING_ORIGIN_MIN_PERCENTAGE)
			}elif cross_validation==CrossValidation.KFOLDS{
				nn.trainKFolds(train_features,train_labels,Core.K_FOLDS)
			}elif cross_validation==CrossValidation.FIXED_PERCENT{
				train,val=Dataset.splitDataset(train_features,train_labels,1-Core.FIXED_VALIDATION_PERCENT)
				nn.trainCustomValidation(train[0],train[1],test[0],test[1])
			}else{
				raise Exception('Unknown cross validation method {}'.format(cross_validation))
			}
			if hyperparameters.model_checkpoint{
				nn.restoreCheckpointWeights()
			}
			output=nn.getMetricMean(hyperparameters.monitor_metric.toKerasName(),cross_validation!=CrossValidation.NONE)
			if output!=output{ # Not a Number, ignore this genome
				Core.LOGGER.warn('Not a number metric mean')
				output=float('-inf') if search_maximum else float('inf')
			}
			if preserve_weights {
				genome.setWeights(nn.mergeWeights(genome.getWeights()))
			}else{
				genome.setWeights(nn.getWeights())
			}
			if Utils.LazyCore.freeMemManually(){
				del nn
			}
			return output
		}

		def after_gen_callback(args_list){
			nonlocal max_gens
			pop_size=args_list[0]
			g=args_list[1]
			best_out=args_list[2]
			timestamp_s=args_list[3]
			population=args_list[4]
			hall_of_fame=args_list[5]
			if hall_of_fame is not None{
				Core.LOGGER.info('\tStoring Hall of Fame Best Individuals...')
				self.updateBestOnGeneticSimulation(simulation_id,hall_of_fame.best,Utils.getTodayDatetime())
				Core.LOGGER.info('\tStored Hall of Fame Best Individuals...OK')
			}
			Core.LOGGER.info('\tStoring generation metadata...')
			self.appendResultOnGeneticSimulation(simulation_id,pop_size,g,best_out,timestamp_s)
			Core.LOGGER.info('\tStored generation metadata...OK')
			if (not Core.STORE_GEN_POP_ONLY_ON_LAST) or g>=max_gens {
				Core.LOGGER.info('\tStoring population...')
				self.clearPopulation(population_id,Utils.getTodayDatetime())
				for individual in population{
					self.appendIndividualToPopulation(population_id,individual,Utils.getTodayDatetime())
				}
				Core.LOGGER.info('\tStored population...OK')
			}
		}

		self.clearResultOnGeneticSimulation(simulation_id)
		self.claimGeneticSimulation(simulation_id,Utils.getTodayDatetime(),Utils.getHostname())
		elite=HallOfFame(max_notables, search_maximum)
		ga=None
		if algorithm == GeneticAlgorithmType.ENHANCED{
			ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
		}elif algorithm == GeneticAlgorithmType.STANDARD{
			ga=StandardGeneticAlgorithm(search_maximum,mutation_rate,sex_rate)
		}else{
			raise Exception('Unknown algorithm {}'.format(algorithm))
		}
		population=PopulationManager(ga,search_space,train_callback,population_start_size,neural_genome=True,print_deltas=True,after_gen_callback=after_gen_callback)
		population.hall_of_fame=elite
		Core.LOGGER.info('Starting natural selection...')
		population.naturalSelection(max_gens,True)
		Core.LOGGER.info('Finished natural selection...OK')
		Core.LOGGER.info('Best output {:.5f} at gen {}, with genome {}'.format(elite.best['output'],elite.best['generation'],elite.best['genome']))
		self.clearHallOfFameIndividuals(hall_of_fame_id,Utils.getTodayDatetime())
		accept_weights_on_hall=True
		for individual in elite.notables {
			try {
				self.appendIndividualToHallOfFame(hall_of_fame_id,individual,Utils.getTodayDatetime(),write_weights=accept_weights_on_hall)
			}except Exception as e{
				Core.LOGGER.exception(e)
				self.appendIndividualToHallOfFame(hall_of_fame_id,individual,Utils.getTodayDatetime(),write_weights=False)
				accept_weights_on_hall=False
			}
		}
		self.finishGeneticSimulation(simulation_id,Utils.getTodayDatetime())
		Core.LOGGER.info('Runned genetic simulation {}...OK'.format(simulation_id))
		if Utils.LazyCore.freeMemManually(){
			del elite
		}
	}

	def trainNeuralNetwork(self,independent_net_id,load=False,just_train=False){
		continue_str='*continue ' if load and not just_train else ''
		Core.LOGGER.info('Training neural network {}{}...'.format(continue_str,independent_net_id))
		Core.LOGGER.info('Parsing train settings...')
		independent_net_metadata = self.fetchNeuralNetworkMetadata(independent_net_id)
		hyper_name=independent_net_metadata[0]
		cve_years_train=independent_net_metadata[1]
		train_data_limit=independent_net_metadata[2]
		cve_years_test=independent_net_metadata[3]
		test_data_limit=independent_net_metadata[4]
		epochs=independent_net_metadata[5]
		patience_epochs=independent_net_metadata[6]
		cross_validation=independent_net_metadata[7]
		train_metric=independent_net_metadata[8]
		test_metric=independent_net_metadata[9]
		hyperparameters=self.fetchHyperparametersDataV2(hyper_name,epochs,patience_epochs,train_metric)
		multiple_networks=hyperparameters.amount_of_networks>1
		Genome.CACHE_WEIGHTS=Core.CACHE_WEIGHTS
		Core.LOGGER.info('\t{}: {}'.format('hyper_name',hyper_name))
		Core.LOGGER.info('\t{}: {}'.format('cve_years_train',cve_years_train))
		Core.LOGGER.info('\t{}: {}'.format('train_data_limit',train_data_limit))
		Core.LOGGER.info('\t{}: {}'.format('cve_years_test',cve_years_test))
		Core.LOGGER.info('\t{}: {}'.format('test_data_limit',test_data_limit))
		Core.LOGGER.info('\t{}: {}'.format('epochs',epochs))
		Core.LOGGER.info('\t{}: {}'.format('patience_epochs',patience_epochs))
		Core.LOGGER.info('\t{}: {}'.format('cross_validation',cross_validation))
		Core.LOGGER.info('\t{}: {}'.format('train_metric',train_metric))
		Core.LOGGER.info('\t{}: {}'.format('test_metric',test_metric))
		Core.LOGGER.info('\t{}: {}'.format('multiple_networks',multiple_networks))
		Core.LOGGER.multiline(str(hyperparameters))
		Core.LOGGER.info('Parsed train settings...OK')
		Core.LOGGER.info('Loading dataset...')
		if multiple_networks {
			train_data_ids,train_features,train_labels=self.loadDatasetMultiNet(cve_years_train,train_data_limit)
		}else{
			train_data_ids,train_features,train_labels=self.loadDataset(cve_years_train,train_data_limit)
		}
		train_features,train_labels=Dataset.balanceDataset(train_features,train_labels,multiple_networks)
		train_labels,labels_equivalence=Dataset.encodeDatasetLabels(train_labels,hyperparameters.label_type)
		# train_features,scale=Dataset.normalizeDatasetFeatures(train_features) # already normalized
		Core.LOGGER.info('Loaded dataset...OK')
		self.claimNeuralNetTrain(independent_net_id,Utils.getTodayDatetime(),Utils.getHostname())
		trained_weights=None
		if load {
			Core.LOGGER.info('Loading weights...')
			trained_weights=self.loadWeightsFromNeuralNet(independent_net_id)
			Core.LOGGER.info('Loaded weights...OK')
		}else{
			Core.LOGGER.info('Creating train network...')
			if multiple_networks {
				input_size=[len(train_features[i][0]) for i in range(len(train_features))]
			}else{
				input_size=len(train_features[0])
			}
			output_size=len(train_labels[0])
			hyperparameters.setLastLayerOutputSize(output_size)
			if hyperparameters.nn_type==NeuralNetworkType.ENHANCED or multiple_networks{
				nn=EnhancedNeuralNetwork(hyperparameters,name='core_train_{}'.format(independent_net_id),verbose=True)
			}else{
				nn=StandardNeuralNetwork(hyperparameters,name='core_train_{}'.format(independent_net_id),verbose=True)
			}
			nn.buildModel(input_size=input_size)
			Core.LOGGER.info('Created train network...OK')
			Core.LOGGER.info('Training network...')
			if cross_validation==CrossValidation.NONE{
				nn.trainNoValidation(train_features,train_labels)
			}elif cross_validation==CrossValidation.ROLLING_FORECASTING_ORIGIN{
				nn.trainRollingForecast(train_features,train_labels,min_size_percentage=Core.ROLLING_FORECASTING_ORIGIN_MIN_PERCENTAGE)
			}elif cross_validation==CrossValidation.KFOLDS{
				nn.trainKFolds(train_features,train_labels,Core.K_FOLDS)
			}elif cross_validation==CrossValidation.FIXED_PERCENT{
				train,val=Dataset.splitDataset(train_features,train_labels,1-Core.FIXED_VALIDATION_PERCENT)
				nn.trainCustomValidation(train[0],train[1],test[0],test[1])
			}else{
				raise Exception('Unknown cross validation method {}'.format(cross_validation))
			}
			if hyperparameters.model_checkpoint{
				nn.restoreCheckpointWeights()
			}
			train_metrics=nn.getMetricMean(train_metric.toKerasName(),cross_validation!=CrossValidation.NONE)
			Core.LOGGER.info('Trained network...OK')
			Core.LOGGER.info('Writing weights..')
			trained_weights=Genome.encodeWeights(nn.getWeights())
			self.appendTMetricsOnNeuralNet(independent_net_id,train_metrics)
			self.saveWeightsOnNeuralNet(independent_net_id,trained_weights)
			Core.LOGGER.info('Wrote weights...OK')
			if Utils.LazyCore.freeMemManually(){
				del nn
			}
		}
		if not just_train {
			Core.LOGGER.info('Loading dataset...')
			if len(cve_years_test)>0{
				if multiple_networks {
					test_data_ids,test_features,test_labels=self.loadDatasetMultiNet(cve_years_test,test_data_limit)
				}else{
					test_data_ids,test_features,test_labels=self.loadDataset(cve_years_test,test_data_limit)
				}
				# no balance for testing
				test_labels,labels_equivalence=Dataset.encodeDatasetLabels(test_labels,hyperparameters.label_type)
				# test_features,scale=Dataset.normalizeDatasetFeatures(test_features) # already normalized
			}
			Core.LOGGER.info('Loaded dataset...OK')
			Core.LOGGER.info('Creating test network...')
			if multiple_networks {
				input_size=[len(train_features[i][0]) for i in range(len(train_features))]
			}else{
				input_size=len(train_features[0])
			}
			output_size=len(train_labels[0])
			hyperparameters.setLastLayerOutputSize(output_size)
			if hyperparameters.nn_type==NeuralNetworkType.ENHANCED or multiple_networks{
				nn=EnhancedNeuralNetwork(hyperparameters,name='core_train-p2_{}'.format(independent_net_id),verbose=True)
			}else{
				nn=StandardNeuralNetwork(hyperparameters,name='core_train-p2_{}'.format(independent_net_id),verbose=True)
			}
			nn.buildModel(input_size=input_size)
			nn.setWeights(Genome.decodeWeights(trained_weights))
			Core.LOGGER.info('Created test network...OK')
			Core.LOGGER.info('Evaluating network...')
			train_eval_res=nn.eval(train_features,train_labels)
			train_eval_res=train_eval_res[train_metric.toKerasName()]
			if len(cve_years_test)>0{
				test_eval_res=nn.eval(test_features,test_labels)
				test_eval_res=test_eval_res[test_metric.toKerasName()]
			}else{
				test_eval_res=None
			}
			Core.LOGGER.info('Evaluated network...OK')
			Core.LOGGER.info(train_metric.toKerasName())
			Core.LOGGER.info('\t'+str(train_eval_res))
			if test_eval_res is not None{
				Core.LOGGER.info(test_metric.toKerasName())
				Core.LOGGER.info('\t'+str(test_eval_res))
			}
			Core.LOGGER.info('Writing results...')
			self.appendStatsOnNeuralNet(independent_net_id,'train_stats',train_eval_res,train_metric.toKerasName())
			if test_eval_res is not None{
				self.appendStatsOnNeuralNet(independent_net_id,'test_stats',test_eval_res,test_metric.toKerasName())
			}
			Core.LOGGER.info('Wrote results...OK')
			if Utils.LazyCore.freeMemManually(){
				del nn
			}
		}
		self.finishNeuralNetTrain(independent_net_id,Utils.getTodayDatetime())
		Core.LOGGER.info('Trained neural network {}{}...OK'.format(continue_str,independent_net_id))
	}

	def predictNeuralNetwork(self,independent_net_id,result_id,eval_data){
		Core.LOGGER.info('Evaluating neural network {}...'.format(independent_net_id))
		Core.LOGGER.info('Parsing evaluate settings...')
		independent_net_metadata = self.fetchNeuralNetworkMetadata(independent_net_id)
		hyper_name=independent_net_metadata[0]
		cross_validation=independent_net_metadata[7]
		test_metric=independent_net_metadata[9]
		hyperparameters=self.fetchHyperparametersDataV2(hyper_name,0,0,test_metric)
		multiple_networks=hyperparameters.amount_of_networks>1
		cve_years_test,test_data_limit=self.parseDatasetMetadataStrRepresentation(eval_data)
		Core.LOGGER.info('\t{}: {}'.format('hyper_name',hyper_name))
		Core.LOGGER.info('\t{}: {}'.format('cross_validation',cross_validation))
		Core.LOGGER.info('\t{}: {}'.format('test_metric',test_metric))
		Core.LOGGER.info('\t{}: {}'.format('cve_years_test',cve_years_test))
		Core.LOGGER.info('\t{}: {}'.format('test_data_limit',test_data_limit))
		Core.LOGGER.info('\t{}: {}'.format('multiple_networks',multiple_networks))
		Core.LOGGER.multiline(str(hyperparameters))
		Core.LOGGER.info('Parsed evaluate settings...OK')
		Core.LOGGER.info('Loading dataset...')
		if multiple_networks {
			test_data_ids,test_features,test_labels=self.loadDatasetMultiNet(cve_years_test,test_data_limit)
		}else{
			test_data_ids,test_features,test_labels=self.loadDataset(cve_years_test,test_data_limit)
		}
		# no balance for testing
		test_labels,labels_equivalence=Dataset.encodeDatasetLabels(test_labels,hyperparameters.label_type)
		# test_features,scale=Dataset.normalizeDatasetFeatures(test_features) # already normalized
		Core.LOGGER.info('Loaded dataset...OK')
		Core.LOGGER.info('Creating eval network...')
		if multiple_networks {
			input_size=[len(test_features[i][0]) for i in range(len(test_features))]
		}else{
			input_size=len(test_features[0])
		}
		output_size=len(test_labels[0])
		hyperparameters.setLastLayerOutputSize(output_size)
		if hyperparameters.nn_type==NeuralNetworkType.ENHANCED or multiple_networks{
			nn=EnhancedNeuralNetwork(hyperparameters,name='core_eval_{}'.format(independent_net_id),verbose=True)
		}else{
			nn=StandardNeuralNetwork(hyperparameters,name='core_eval_{}'.format(independent_net_id),verbose=True)
		}
		nn.buildModel(input_size=input_size)
		nn.setWeights(Genome.decodeWeights(self.loadWeightsFromNeuralNet(independent_net_id)))
		Core.LOGGER.info('Created eval network...OK')
		Core.LOGGER.info('Predicting...')
		classes,activations=nn.predict(test_features,True,True)
		statistics=Dataset.statisticalAnalysis(classes,test_labels)
		Core.LOGGER.info('Predicted...OK')
		Core.LOGGER.logDict(statistics,'Statistics')
		Dataset.compareAndPrintLabels(classes,activations,test_labels,show_positives=True,equivalence_table_1=labels_equivalence,logger=Core.LOGGER)
		Core.LOGGER.info('Writing results...')
		self.storeEvalNeuralNetResult(result_id,test_data_ids,classes,activations,test_labels,statistics)
		Core.LOGGER.info('Wrote results...OK')
		if Utils.LazyCore.freeMemManually(){
			del nn
		}
		Core.LOGGER.info('Evaluated neural network {}...OK'.format(independent_net_id))
	}

	def loadDataset(self,years,limit){
		processed_db=self.mongo.getProcessedDB()
		data_ids=[]
		data_features=[]
		data_labels=[]
		for year in years {
			cur_cves=self.mongo.findAllOnDB(processed_db,'dataset',query={'cve':{'$regex':'CVE-{}-.*'.format(year)}}).sort('cve',1)
			if cur_cves is None {
				raise Exception('Unable to find cves from {}:{}'.format(year,limit))
			}
			parsed_cves=0
			for cur_cve in cur_cves {
				data_ids.append(cur_cve['cve'])
				parsed_cve_features=[]
				for k,v in sorted(cur_cve['features'].items()){
					if not ('reference_' in k and 'exploit' in k){
						parsed_cve_features.append(float(v))
					}
				}
				data_features.append(parsed_cve_features)
				parsed_cve_labels=[]
				parsed_cve_labels.append(int(cur_cve['labels']['exploits_has']))
				if len(parsed_cve_labels)==1{
					parsed_cve_labels=parsed_cve_labels[0] # to avoid enumfication of the dataset later
				}
				data_labels.append(parsed_cve_labels)
				parsed_cves+=1
				if limit>0 and parsed_cves >= limit {
					break
				}
			}
		}
		return data_ids,data_features,data_labels
	}

	def loadDatasetMultiNet(self,years,limit){
		processed_db=self.mongo.getProcessedDB()
		amount_of_groups=5
		data_ids=[]
		data_features=[[] for _ in range(amount_of_groups)]
		data_labels=[]
		for year in years {
			cur_cves=self.mongo.findAllOnDB(processed_db,'dataset',query={'cve':{'$regex':'CVE-{}-.*'.format(year)}}).sort('cve',1)
			if cur_cves is None {
				raise Exception('Unable to find cves from {}:{}'.format(year,limit))
			}
			parsed_cves=0
			for cur_cve in cur_cves {
				data_ids.append(cur_cve['cve'])
				parsed_cve_features=[[] for _ in range(amount_of_groups)]
				for k,v in sorted(cur_cve['features'].items()){
					index=-1
					if 'cvss_' in k and '_ENUM_' in k {
						index=1
					}elif 'description_' in k {
						index=2
					}elif 'reference_' in k {
						if 'exploit' not in k{
							index=3
						}else{
							index=None
						}
					}elif 'vendor_' in k {
						index=4
					}else{
						index=0
					}
					if index is not None{
						parsed_cve_features[index].append(float(v))
					}
				}
				for x in range(amount_of_groups){
					data_features[x].append(parsed_cve_features[x])
				}
				parsed_cve_labels=[]
				parsed_cve_labels.append(int(cur_cve['labels']['exploits_has']))
				if len(parsed_cve_labels)==1{
					parsed_cve_labels=parsed_cve_labels[0] # to avoid enumfication of the dataset later
				}
				data_labels.append(parsed_cve_labels)
				parsed_cves+=1
				if limit>0 and parsed_cves >= limit {
					break
				}
			}
		}
		return data_ids,data_features,data_labels
	}

	def parseDatasetMetadataStrRepresentation(self,data_meta){
		limit=0
		if ':' in data_meta{
			cve_years_arr=data_meta.split(':')
			limit=int(cve_years_arr[1])
			cve_years_arr=cve_years_arr[0]
		}else{
			cve_years_arr=data_meta
		}
		cve_years_arr=cve_years_arr.split(',')
		cve_years=[int(el) for el in cve_years_arr]
		return cve_years,limit
	}

	def fetchGeneticSimulationData(self,simulation_id){
		genetic_db=self.mongo.getGeneticDB()
		simu_data=self.mongo.findOneOnDBFromIndex(genetic_db,'simulations','_id',simulation_id)
		if simu_data is None {
			raise Exception('Unable to find simulation {}'.format(simulation_id))
		}
		environment_name=str(simu_data['env_name'])
		cve_years,train_data_limit=self.parseDatasetMetadataStrRepresentation(str(simu_data['train_data']))
		hall_of_fame_id=str(simu_data['hall_of_fame_id'])
		population_id=str(simu_data['population_id'])
		population_start_size=int(simu_data['pop_start_size'])
		max_gens=int(simu_data['max_gens'])
		max_age=int(simu_data['max_age'])
		max_children=int(simu_data['max_children'])
		mutation_rate=float(simu_data['mutation_rate'])
		recycle_rate=float(simu_data['recycle_rate'])
		sex_rate=float(simu_data['sex_rate'])
		max_notables=int(simu_data['max_notables'])
		cross_validation=CrossValidation(simu_data['cross_validation'])
		metric_mode=Metric(simu_data['metric'])
		algorithm=GeneticAlgorithmType(simu_data['algorithm'])
		label_type=LabelEncoding(simu_data['label_type'])
		if 'neural_type' in simu_data{
			enhanced_neural_network=NeuralNetworkType(simu_data['neural_type'])
		}else{
			enhanced_neural_network=NeuralNetworkType.STANDARD
		}
		return environment_name, cve_years, train_data_limit, hall_of_fame_id, population_id, population_start_size, max_gens, max_age, max_children, mutation_rate, recycle_rate, sex_rate, max_notables, cross_validation, metric_mode, algorithm, label_type, enhanced_neural_network
	}

	def fetchEnvironmentDataV1(self,environment_name){
		genetic_db=self.mongo.getGeneticDB()
		search_space_db=self.mongo.findOneOnDBFromIndex(genetic_db,'environments','name',environment_name)
		if search_space_db is None {
			raise Exception('Unable to find environment {}'.format(environment_name))
		}
		search_space_db=search_space_db['space_search'] # keep it misspelled
		search_space=SearchSpace()
		search_space.add('amount_of_layers',search_space_db['amount_of_layers']['min'],search_space_db['amount_of_layers']['max'],SearchSpace.Type.INT)
		search_space.add('epochs',search_space_db['epochs']['min'],search_space_db['epochs']['max'],SearchSpace.Type.INT)
		search_space.add('batch_size',search_space_db['batch_size']['min'],search_space_db['batch_size']['max'],SearchSpace.Type.INT)
		search_space.add('layer_sizes',search_space_db['layer_sizes']['min'],search_space_db['layer_sizes']['max'],SearchSpace.Type.INT)
		search_space.add('activation_functions',search_space_db['activation_functions']['min'],search_space_db['activation_functions']['max'],SearchSpace.Type.INT)
		search_space.add('alpha',search_space_db['alpha']['min'],search_space_db['alpha']['max'],SearchSpace.Type.FLOAT)
		# K, L and sparcity are useless, since we are not using SLIDE
		return search_space
	}

	def fetchEnvironmentDataV2(self,environment_name,metric,encoder){
		genetic_db=self.mongo.getGeneticDB()
		env_db=self.mongo.findOneOnDBFromIndex(genetic_db,'environments','name',environment_name)
		if env_db is None {
			raise Exception('Unable to find environment {}'.format(environment_name))
		}
		networks=env_db['amount_of_networks']
		search_space_db=env_db['search_space']
		multiple_networks=networks>1
		search_space=SearchSpace()
		if multiple_networks {
			search_space.add(networks,networks,SearchSpace.Type.INT,'networks')
			search_space.add(search_space_db['batch_size']['min'],search_space_db['batch_size']['max'],SearchSpace.Type.INT,'batch_size')
			search_space.add(False,False,SearchSpace.Type.BOOLEAN,'shuffle') # always false
			search_space.add(True,True,SearchSpace.Type.BOOLEAN,'model_checkpoint') # always true
			search_space.add(search_space_db['patience_epochs']['min'],search_space_db['patience_epochs']['max'],SearchSpace.Type.INT,'patience_epochs')
			search_space.add(search_space_db['epochs']['min'],search_space_db['epochs']['max'],SearchSpace.Type.INT,'max_epochs')
			search_space.add(encoder,encoder,SearchSpace.Type.INT,'label_type')
			search_space.add(metric,metric,SearchSpace.Type.INT,'monitor_metric')
			for n in range(networks){
				search_space.add(search_space_db['amount_of_layers']['min'][n],search_space_db['amount_of_layers']['max'][n],SearchSpace.Type.INT,'layers'+'_{}'.format(n))
				search_space.add(search_space_db['layer_sizes']['min'][n],search_space_db['layer_sizes']['max'][n],SearchSpace.Type.INT,'layer_sizes'+'_{}'.format(n))
				search_space.add(NodeType(search_space_db['activation_functions']['min'][n]),NodeType(search_space_db['activation_functions']['max'][n]),SearchSpace.Type.INT,'node_types'+'_{}'.format(n))
				search_space.add(search_space_db['dropouts']['min'][n],search_space_db['dropouts']['max'][n],SearchSpace.Type.FLOAT,'dropouts'+'_{}'.format(n))
				search_space.add(search_space_db['alpha']['min'][n],search_space_db['alpha']['max'][n],SearchSpace.Type.FLOAT,'alpha'+'_{}'.format(n))
				search_space.add(search_space_db['bias']['min'][n] not in (0,False),search_space_db['bias']['max'][n] not in (0,False),SearchSpace.Type.BOOLEAN,'bias'+'_{}'.format(n))
				search_space.add(Loss(search_space_db['loss']['min'][n]),Loss(search_space_db['loss']['max'][n]),SearchSpace.Type.INT,'loss'+'_{}'.format(n))
				search_space.add(Optimizers(search_space_db['optimizer']['min'][n]),Optimizers(search_space_db['optimizer']['max'][n]),SearchSpace.Type.INT,'optimizer'+'_{}'.format(n))
			}
		}else{
			search_space.add(search_space_db['amount_of_layers']['min'],search_space_db['amount_of_layers']['max'],SearchSpace.Type.INT,'layers')
			search_space.add(search_space_db['batch_size']['min'],search_space_db['batch_size']['max'],SearchSpace.Type.INT,'batch_size')
			search_space.add(search_space_db['alpha']['min'],search_space_db['alpha']['max'],SearchSpace.Type.FLOAT,'alpha')
			search_space.add(False,False,SearchSpace.Type.BOOLEAN,'shuffle') # always false
			search_space.add(search_space_db['patience_epochs']['min'],search_space_db['patience_epochs']['max'],SearchSpace.Type.INT,'patience_epochs')
			search_space.add(search_space_db['epochs']['min'],search_space_db['epochs']['max'],SearchSpace.Type.INT,'max_epochs')
			search_space.add(Loss(search_space_db['loss']['min']),Loss(search_space_db['loss']['max']),SearchSpace.Type.INT,'loss')
			search_space.add(encoder,encoder,SearchSpace.Type.INT,'label_type')
			search_space.add(Optimizers(search_space_db['optimizer']['min']),Optimizers(search_space_db['optimizer']['max']),SearchSpace.Type.INT,'optimizer')
			search_space.add(metric,metric,SearchSpace.Type.INT,'monitor_metric')
			search_space.add(True,True,SearchSpace.Type.BOOLEAN,'model_checkpoint') # always true
			search_space.add(search_space_db['layer_sizes']['min'],search_space_db['layer_sizes']['max'],SearchSpace.Type.INT,'layer_sizes')
			search_space.add(NodeType(search_space_db['activation_functions']['min']),NodeType(search_space_db['activation_functions']['max']),SearchSpace.Type.INT,'node_types')
			search_space.add(search_space_db['dropouts']['min'],search_space_db['dropouts']['max'],SearchSpace.Type.FLOAT,'dropouts')
			search_space.add(search_space_db['bias']['min'] not in (0,False),search_space_db['bias']['max'] not in (0,False),SearchSpace.Type.BOOLEAN,'bias')
		}
		output_layer_node_type=NodeType(search_space_db['output_layer_node_type'])
		return search_space,output_layer_node_type,multiple_networks
	}

	def fetchNeuralNetworkMetadata(self,independent_net_id){
		neural_db=self.mongo.getNeuralDB()
		train_metadata=self.mongo.findOneOnDBFromIndex(neural_db,'independent_net','_id',independent_net_id)
		if train_metadata is None {
			raise Exception('Unable to find network {}'.format(independent_net_id))
		}
		hyper_name=str(train_metadata['hyperparameters_name'])
		cve_years_train,train_data_limit=self.parseDatasetMetadataStrRepresentation(str(train_metadata['train_data']))
		cve_years_test,test_data_limit=self.parseDatasetMetadataStrRepresentation(str(train_metadata['test_data']))
		epochs=train_metadata['epochs']
		patience_epochs=train_metadata['patience_epochs']
		cross_validation=CrossValidation(train_metadata['cross_validation'])
		train_metric=Metric(train_metadata['train_metric'])
		test_metric=Metric(train_metadata['test_metric'])
		return hyper_name, cve_years_train, train_data_limit, cve_years_test, test_data_limit, epochs, patience_epochs, cross_validation, train_metric, test_metric
	}

	def fetchHyperparametersDataV1(self,hyper_name){
		neural_db=self.mongo.getNeuralDB()
		hyperparameters=self.mongo.findOneOnDBFromIndex(neural_db,'snn_hyperparameters','name',hyper_name)
		if hyperparameters is None {
			raise Exception('Unable to find hyperparameters {}'.format(hyper_name))
		}
		hyperparameters=Hyperparameters(int(hyperparameters['batch_size']), float(hyperparameters['alpha']), bool(hyperparameters['shuffle']), bool(hyperparameters['adam']), LabelEncoding(hyperparameters['label_type']), int(hyperparameters['layers']), [int(el) for el in hyperparameters['layer_sizes']], [NodeType(el) for el in hyperparameters['node_types']])
		# K, L, rehash, rebuild, range_pow and sparcity are useless, since we are not using SLIDE
		return hyperparameters
	}

	def fetchHyperparametersDataV2(self,hyper_name,epochs,pat_epochs,metric){
		neural_db=self.mongo.getNeuralDB()
		hyperparameters=self.mongo.findOneOnDBFromIndex(neural_db,'snn_hyperparameters','name',hyper_name)
		if hyperparameters is None {
			raise Exception('Unable to find hyperparameters {}'.format(hyper_name))
		}
		amount_of_networks=int(hyperparameters['amount_of_networks'])
		if amount_of_networks > 1 {
			layers=[int(el) for el in hyperparameters['layers']]
			dropouts=[[float(el) for el in layer ] for layer in hyperparameters['dropouts']]
			bias=[[bool(el) for el in layer ] for layer in hyperparameters['bias']]
			layer_sizes=[[int(el) for el in layer ] for layer in hyperparameters['layer_sizes']]
			node_types=[[NodeType(el) for el in layer ] for layer in hyperparameters['node_types']]
			batch_size=int(hyperparameters['batch_size'])
			alpha=[float(el) for el in hyperparameters['alpha']]
			shuffle=bool(hyperparameters['shuffle'])
			optimizer=[Optimizers(el) for el in hyperparameters['optimizer']]
			label_type=LabelEncoding(hyperparameters['label_type'])
			loss=[Loss(el) for el in hyperparameters['loss']]
		} else {
			layers=int(hyperparameters['layers'])
			dropouts=[float(el) for el in hyperparameters['dropouts']]
			bias=[bool(el) for el in hyperparameters['bias']]
			layer_sizes=[int(el) for el in hyperparameters['layer_sizes']]
			node_types=[NodeType(el) for el in hyperparameters['node_types']]
			batch_size=int(hyperparameters['batch_size'])
			alpha=float(hyperparameters['alpha'])
			shuffle=bool(hyperparameters['shuffle'])
			optimizer=Optimizers(hyperparameters['optimizer'])
			label_type=LabelEncoding(hyperparameters['label_type'])
			loss=Loss(hyperparameters['loss'])
		}
		nn_type=NeuralNetworkType(hyperparameters['neural_type'])
		max_epochs=epochs
		patience_epochs=pat_epochs
		monitor_metric=metric
		return Hyperparameters(batch_size, alpha, shuffle, optimizer, label_type, layers, layer_sizes, node_types, dropouts, patience_epochs, max_epochs, bias, loss,monitor_metric=monitor_metric,amount_of_networks=amount_of_networks,nn_type=nn_type)
	}

	# Disclaim, I'm to lazy to fix Pytho{\} dicts or to make a real Pytho{\} "compiler" with lexemes and stuff
	def claimGeneticSimulation(self,simulation_id,now_str,hostname){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(simulation_id),'started_at':{'$eq':None}} # TODO ? why? it was 'started_at':{'$ne':None} on core
		query_update={'$set':{'started_at':now_str}}
		self.mongo.updateDocumentOnDB(self.mongo.getGeneticDB(),'simulations',query_fetch,query_update,verbose=False,ignore_lock=False)
		
		query_fetch={'_id':self.mongo.getObjectId(simulation_id)}
		query_update={'$set':{'started_by':hostname}}
		self.mongo.updateDocumentOnDB(self.mongo.getGeneticDB(),'simulations',query_fetch,query_update,verbose=False,ignore_lock=False)
		# Pytho{\}: End regular Python
	}

	def appendResultOnGeneticSimulation(self,simulation_id,pop_size,g,best_out,timestamp_s){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(simulation_id)}
		query_update={'$push':{'results':{'pop_size':pop_size,'cur_gen':g,'gen_best_out':best_out,'delta_ms':int(timestamp_s*1000)}}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getGeneticDB(),'simulations',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def updateBestOnGeneticSimulation(self,simulation_id,best,now_str){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(simulation_id)}
		query_update={'$set':{'updated_at':now_str,'best':best}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getGeneticDB(),'simulations',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def appendIndividualToPopulation(self,population_id,individual,now_str){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(population_id)}
		query_update={'$push':{'neural_genomes':self.genomeToDict(individual,Core.WRITE_POPULATION_WEIGHTS)},'$set':{'updated_at':now_str}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'populations',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def clearPopulation(self,population_id,now_str){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(population_id)}
		query_update={'$set':{'updated_at':now_str,'neural_genomes':[]}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'populations',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def clearResultOnGeneticSimulation(self,simulation_id){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(simulation_id)}
		query_update={'$set':{'results':[]}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getGeneticDB(),'simulations',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def finishGeneticSimulation(self,simulation_id,now_str){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(simulation_id)}
		query_update={'$set':{'finished_at':now_str}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getGeneticDB(),'simulations',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def claimNeuralNetTrain(self,independent_net_id,now_str,hostname){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(independent_net_id),'started_at':{'$eq':None}} # TODO ? why? it was 'started_at':{'$ne':None} on core
		query_update={'$set':{'started_at':now_str}}
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'independent_net',query_fetch,query_update,verbose=False,ignore_lock=False)
		
		query_fetch={'_id':self.mongo.getObjectId(independent_net_id)}
		query_update={'$set':{'started_by':hostname}}
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'independent_net',query_fetch,query_update,verbose=False,ignore_lock=False)
		# Pytho{\}: End regular Python
	}

	def appendIndividualToHallOfFame(self,hall_of_fame_id,taylor_swift,now_str,write_weights=True){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(hall_of_fame_id)}
		query_update={'$set':{'updated_at':now_str},'$push':{'neural_genomes':self.genomeToDict(taylor_swift,write_weights=write_weights)}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'hall_of_fame',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def clearHallOfFameIndividuals(self,hall_of_fame_id,now_str){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(hall_of_fame_id)}
		query_update={'$set':{'updated_at':now_str,'neural_genomes':[]}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'hall_of_fame',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def appendTMetricsOnNeuralNet(self,independent_net_id,train_metrics){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(independent_net_id)}
		query_update={'$set':{'train_metrics':train_metrics}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'independent_net',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def saveWeightsOnNeuralNet(self,independent_net_id,trained_weights){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(independent_net_id)}
		query_update={'$set':{'weights':trained_weights}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'independent_net',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def loadWeightsFromNeuralNet(self,independent_net_id){
		# Pytho{\}: Start regular Python
		query={'_id':self.mongo.getObjectId(independent_net_id)}
		# Pytho{\}: End regular Python
		net=self.mongo.findOneOnDB(self.mongo.getNeuralDB(),'independent_net',query,wait_unlock=True)
		if net is not None and 'weights' in net{
			return net['weights']
		}
		return None
	}

	def appendStatsOnNeuralNet(self,independent_net_id,name,res,metric){
        # Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(independent_net_id)}
		query_update={'$set':{name:{'metric':metric,'values':res}}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'independent_net',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def storeEvalNeuralNetResult(self,result_id,test_data_ids,classes,activations,test_labels,statistics,log_result=True){
        correct=0
        results=[]
        if log_result {
            Core.LOGGER.info('Results to be stored:')
        }
        for i in range(len(test_data_ids)){
            match=classes[i][0]==test_labels[i][0]
            if match{
                correct+=1
            }
            confidence=activations[i][0]*100.0
            result='{}: Label: {} | Predicted Exploit: {} | Conficende {:.2f}% | Prediction Match: {}'.format(test_data_ids[i],test_labels[i][0],classes[i][0],confidence,match)
            if log_result {
                Core.LOGGER.info('\t'+log_result)
            }
            results.append(result)
        }
        # Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(result_id)}
		query_update={'$set':{'result_stats':statistics,'total_test_cases':len(test_data_ids),'matching_preds':correct,'results':results}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'eval_results',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def finishNeuralNetTrain(self,independent_net_id,now_str){
		# Pytho{\}: Start regular Python
		query_fetch={'_id':self.mongo.getObjectId(independent_net_id)}
		query_update={'$set':{'finished_at':now_str}}
		# Pytho{\}: End regular Python
		self.mongo.updateDocumentOnDB(self.mongo.getNeuralDB(),'independent_net',query_fetch,query_update,verbose=False,ignore_lock=False)
	}

	def genomeToDict(self,individual,write_weights=True){
		genome={}
		genome['id']=individual.id
		genome['output']=individual.output
		genome['fitness']=individual.fitness
		genome['gen']=individual.gen
		if individual.age is not None{
			genome['age']=individual.age
		}
		genome['mt_dna']=individual.mt_dna
		genome['dna']='[ '+' '.join([str(i) for i in individual.dna])+' ]'
		if individual.is_neural and write_weights{
			genome['weights']=individual.getWeights(raw=True)
		}
		return genome
	}
}