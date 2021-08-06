#!/bin/python

from Utils import Utils
from Logger import Logger
from MongoDB import MongoDB
from SearchSpace import SearchSpace
from Hyperparameters import Hyperparameters
from Enums import CrossValidation,Metric,LabelEncoding,GeneticAlgorithm,NodeType

class Core(object){
    # 'Just':'to fix vscode coloring':'when using pytho{\}'

    LOGGER=None

    def __init__(self, mongo, logger){
        Core.LOGGER=logger
		self.mongo=mongo
    }

    def runGeneticSimulation(self,simulation_id){
        Core.LOGGER.info('Running genetic simulation {}...'.format(simulation_id))
        Core.LOGGER.info('Loading simulation...')
        environment_name, cve_years, train_data_limit, hall_of_fame_id, population_id, population_start_size, max_gens, max_age, max_children, mutation_rate, recycle_rate, sex_rate, max_notables, cross_validation, metric_mode, algorithm, label_encoding = self.fetchGeneticSimulationData(simulation_id)
        Core.LOGGER.info('Loaded simulation...OK')
        Core.LOGGER.info('Loading search space...')
        search_space=self.fetchEnvironmentData(environment_name)
        Core.LOGGER.info('Loaded search space...OK')
        Core.LOGGER.info('Loading dataset...')
        train_data_ids,train_data_features,train_data_labels=self.loadDataset(cve_years,train_data_limit)
        # TODO encode dataset
        # TODO balance dataset
        Core.LOGGER.info('Loaded dataset...OK')
        shuffle_train_data=False
        adam_optimizer=True
        use_neural_genome=True
        search_maximum=metric_mode!=Metric.RAW_LOSS
        Core.LOGGER.info('Starting natural selection...')
        # TODO code me
        Core.LOGGER.info('Finished natural selection...OK')
        Core.LOGGER.info('Runned genetic simulation {}...OK'.format(simulation_id))
    }

    def trainNeuralNetwork(self,independent_net_id,load=False,just_train=False){
        Core.LOGGER.info('Training neural network {}...'.format(independent_net_id))
        Core.LOGGER.info('Parsing train settings...')
        hyper_name, cve_years_train, train_data_limit, cve_years_test, test_data_limit, epochs, cross_validation, train_metric, test_metric = self.fetchNeuralNetworkMetadata(independent_net_id)
        hyperparameters=self.fetchHyperparametersData(hyper_name)
        Core.LOGGER.info('Parsed train settings...OK')
        Core.LOGGER.info('Loading dataset...')
        train_data_ids,train_data_features,train_data_labels=self.loadDataset(cve_years_train,train_data_limit)
        # TODO encode dataset
        # TODO balance dataset
        Core.LOGGER.info('Loaded dataset...OK')
        if load {
            Core.LOGGER.info('Loading weights...')
            # TODO code me
            Core.LOGGER.info('Loaded weights...OK')
        }else{
            Core.LOGGER.info('Training network...')
            # TODO code me 
            Core.LOGGER.info('Trained network...OK')
        }
        if not just_train {
            Core.LOGGER.info('Evaluating network...')
            Core.LOGGER.info('Loading dataset...')
            test_data_ids,test_data_features,test_data_labels=self.loadDataset(cve_years_test,test_data_limit)
            # TODO encode dataset
            # TODO balance dataset
            Core.LOGGER.info('Loaded dataset...OK')
            # TODO code me 
            Core.LOGGER.info('Evaluated network...OK')
        }
        # TODO code me
        Core.LOGGER.info('Trained neural network {}...OK'.format(independent_net_id))
    }

    def evalNeuralNetwork(self,independent_net_id,result_id,eval_data){
        Core.LOGGER.info('Evaluating neural network {}...'.format(independent_net_id))
        Core.LOGGER.info('Parsing evaluate settings...')
        hyper_name,_,_,_,_,_,cross_validation,_,test_metric = self.fetchNeuralNetworkMetadata(independent_net_id)
        hyperparameters=self.fetchHyperparametersData(hyper_name)
        cve_years_test,test_data_limit=self.parseDatasetMetadataStrRepresentation(eval_data)
        Core.LOGGER.info('Parsed evaluate settings...OK')
        Core.LOGGER.info('Loading dataset...')
        test_data_ids,test_data_features,test_data_labels=self.loadDataset(cve_years_test,test_data_limit)
        # TODO encode dataset
        Core.LOGGER.info('Loaded dataset...OK')
        # TODO code me
        Core.LOGGER.info('Evaluated neural network {}...OK'.format(independent_net_id))
    }

    def loadDataset(self,years,limit){
        processed_db=self.mongo.getProcessedDB()
        data_ids=[]
        data_features=[]
        data_labels=[]
        for year in years {
            cur_cves=mongo.findAllOnDB(processed_db,'dataset',query={'cve':{'$regex':'CVE-{}-.*'.format(year)}}]).sort('cve',1)
            if cur_cves is None {
                raise Exception('Unable to find cves from {}:{}'.format(year,limit))
            }
            parsed_cves=0
            for cur_cve in cur_cves {
                data_ids.append(cur_cve['cve'])
                parsed_cve_features=[]
                for k,v in cur_cve['features'].items(){
                    parsed_cve_features.append(float(v))
                }
                data_features.append(parsed_cve_features)
                parsed_cve_labels=[]
                parsed_cve_labels.append(int(cur_cve['labels']['exploits_has']))
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
        return ,limit
    }

    def fetchGeneticSimulationData(self,simulation_id){
        genetic_db=self.mongo.getGeneticDB()
        simu_data=mongo.findOneOnDBFromIndex(genetic_db,'simulations','_id',simulation_id)
        if simu_data is None {
            raise Exception('Unable to find simulation {}'.format(simulation_id))
        }
        environment_name=str(simu_data['env_name'])
        cve_years,train_data_limit=self.parseDatasetMetadataStrRepresentation(str(simu_data['train_data']))
        hall_of_fame_id=str(simu_data['hall_of_fame_id'])
        population_id=str(simu_data['population_id'])
        population_start_size=int(data['pop_start_size'])
        max_gens=int(data['max_gens'])
        max_age=int(data['max_age'])
        max_children=int(data['max_children'])
        mutation_rate=float(data['mutation_rate'])
        recycle_rate=float(data['recycle_rate'])
        sex_rate=float(data['sex_rate'])
        max_notables=int(data['max_notables'])
        cross_validation=CrossValidation(data['cross_validation'])
        metric_mode=Metric(data['metric'])
        algorithm=GeneticAlgorithm(data['algorithm'])
        label_encoding=LabelEncoding(data['label_type'])
        return environment_name, cve_years, train_data_limit, hall_of_fame_id, population_id, population_start_size, max_gens, max_age, max_children, mutation_rate, recycle_rate, sex_rate, max_notables, cross_validation, metric_mode, algorithm, label_encoding
    }

    def fetchEnvironmentData(self,environment_name){
        genetic_db=self.mongo.getGeneticDB()
        search_space_db=mongo.findOneOnDBFromIndex(genetic_db,'environments','name',environment_name)
        if search_space_db is None {
            raise Exception('Unable to find environment {}'.format(environment_name))
        }
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

    def fetchNeuralNetworkMetadata(self,independent_net_id){
        neural_db=self.mongo.getNeuralDB()
        train_metadata=mongo.findOneOnDBFromIndex(neural_db,'independent_net','_id',independent_net_id)
        if train_metadata is None {
            raise Exception('Unable to find network {}'.format(independent_net_id))
        }
        hyper_name=str(train_metadata['hyperparameters_name'])
        cve_years_train,train_data_limit=self.parseDatasetMetadataStrRepresentation(str(train_metadata['train_data']))
        cve_years_test,test_data_limit=self.parseDatasetMetadataStrRepresentation(str(train_metadata['test_data']))
        epochs=train_metadata['epochs']
        cross_validation=CrossValidation(train_metadata['cross_validation'])
        train_metric=Metric(train_metadata['train_metric'])
        test_metric=Metric(train_metadata['test_metric'])
        return hyper_name, cve_years_train, train_data_limit, cve_years_test, test_data_limit, epochs, cross_validation, train_metric, test_metric
    }

    def fetchHyperparametersData(self,hyper_name){
        neural_db=self.mongo.getNeuralDB()
        hyperparameters=mongo.findOneOnDBFromIndex(neural_db,'snn_hyperparameters','name',hyper_name)
        if train_metadata is None {
            raise Exception('Unable to find hyperparameters {}'.format(hyper_name))
        }
        hyperparameters=Hyperparameters(int(hyperparameters['batch_size']), float(hyperparameters['alpha']), bool(hyperparameters['shuffle']), bool(hyperparameters['adam']), LabelEncoding(hyperparameters['label_type']), int(hyperparameters['layers']), [int(el) for el in hyperparameters['layer_sizes']], [NodeType(el) for el in hyperparameters['node_types']])
        # K, L, rehash, rebuild, range_pow and sparcity are useless, since we are not using SLIDE
        return hyperparameters
    }
}