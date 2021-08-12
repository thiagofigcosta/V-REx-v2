#include <iostream>
#include <signal.h>
#include <boost/stacktrace.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <stdexcept>
#include <sys/resource.h>
#include <omp.h>

#include "Utils.hpp"
#include "NeuralGenome.hpp"
#include "test.hpp"
#include "Slide.hpp"

using namespace std;

MongoDB* mongo=nullptr;
bool trap_signals=false;
bool connect_mongo=true;
bool set_stack_size=false;
bool cache_genetic=true;
bool store_genetic_populations_only_on_last_gen=false;
bool single_thread=false;
bool verbose=false;
string custom_mongo_ip=""; 

// Function used to check that 'opt1' and 'opt2' are not specified at the same time.
void conflicting_options(const boost::program_options::variables_map& vm, const char* opt1, const char* opt2){
    if (vm.count(opt1) && !vm[opt1].defaulted() 
        && vm.count(opt2) && !vm[opt2].defaulted())
        throw logic_error(string("Conflicting options '") 
                          + opt1 + "' and '" + opt2 + "'.");
}

// Function used to check that of 'for_what' is specified, then 'required_option' is specified too.
void option_dependency(const boost::program_options::variables_map& vm, const char* for_what, const char* required_option){
    if (vm.count(for_what) && !vm[for_what].defaulted())
        if (vm.count(required_option) == 0 || vm[required_option].defaulted())
            throw logic_error(string("Option '") + for_what 
                              + "' requires option '" + required_option + "'.");
}

void exceptionHandler(int signum) {
    ::signal(signum, SIG_DFL);
    // boost::stacktrace::safe_dump_to("./backtrace.dump");
    cerr << endl << "Stack trace:" << endl << boost::stacktrace::stacktrace() << endl;
    try {
        exception_ptr eptr=current_exception();
        if (eptr) {
            rethrow_exception(eptr);
        }
    } catch(const exception& e) {
        cerr << "Caught exception:\n\t\"" << e.what() << "\"\n\n";
    }
    if(!NeuralGenome::last_print_str.empty()){
        cout<<"\n\nOn "+NeuralGenome::last_print_str<<endl;
    }
    ::raise(SIGABRT);
}

vector<pair<vector<int>, vector<float>>> encodeData(vector<pair<vector<int>, vector<float>>> data, SlideLabelEncoding label_type){
    switch(label_type){
        case SlideLabelEncoding::INT_CLASS:
            cout<<"Using INT label\n";
            break;
        case SlideLabelEncoding::NEURON_BY_NEURON:
            cout<<"Using NEURON_BY_NEURON label\n";
            data=Utils::encodeDatasetLabelsUsingFirst(data,DataEncoder::BINARY,1);
            break;
        case SlideLabelEncoding::NEURON_BY_N_LOG_LOSS:
            cout<<"Using NEURON_BY_NEURON_WITH_LOG_LOSS label\n";
            data=Utils::encodeDatasetLabelsUsingFirst(data,DataEncoder::BINARY,1);
            break;
    }
    return data;
}

void setup(){
    if (trap_signals){
        cout<<"Setting up signal trap...\n";
        ::signal(SIGSEGV, &exceptionHandler);
        ::signal(SIGABRT, &exceptionHandler);
        cout<<"Set up signal trap...OK\n";
    }
    if (set_stack_size){
        cout<<"Setting up stack size...\n";
        const rlim_t kStackSize = 1 * 1024 * 1024 * 1024;   // 1 * 1024 * 1024 * 1024 = 1 GB
        struct rlimit rl;
        int result;
        result = getrlimit(RLIMIT_STACK, &rl);
        if (result == 0){
            if (rl.rlim_cur < kStackSize){
                rl.rlim_cur = kStackSize;
                result = setrlimit(RLIMIT_STACK, &rl);
                if (result != 0){
                    fprintf(stderr, "setrlimit returned result = %d\n", result);
                }
            }
        }
        cout<<"Set up stack size...OK\n";
    }
    if(connect_mongo){
        cout<<"Connecting on Mongo...\n";
        string mongo_host;
        if (custom_mongo_ip==""){
            if (Utils::runningOnDockerContainer()){
                mongo_host="mongo";
            }else{
                mongo_host="127.0.0.1";
            }
        }else{
            mongo_host=custom_mongo_ip;
        }
        mongo = new MongoDB(mongo_host,"root","123456");
        cout<<"Connected on Mongo...OK\n";
    }
    if (Slide::MAX_THREADS > 0){
        cout<<"Limiting amount of threads to "+to_string(Slide::MAX_THREADS)+"...\n";
        omp_set_dynamic(0);
        omp_set_num_threads(Slide::MAX_THREADS);
        cout<<"Limited amount of threads to "+to_string(Slide::MAX_THREADS)+"...OK\n";
    }
}

void tearDown(){
    if(mongo){
        delete mongo;
        mongo=nullptr;
    }
    Utils::rmFile(NeuralGenome::getBaseFolder());
}

void runGeneticSimulation(string simulation_id){
    cout<<"Running genetic simulation "+simulation_id+"...\n";
    cout<<"Parsing genetic settings...\n";
    pair<vector<string>,vector<float>> simu_data=mongo->fetchGeneticSimulationData(simulation_id);
    string environment_name=simu_data.first[0];
    vector<string> str_cve_years=Utils::splitString(simu_data.first[1],",");
    string hall_of_fame_id=simu_data.first[2];
    string population_id=simu_data.first[3];
    int population_start_size=(int)simu_data.second[0];
    int max_gens=(int)simu_data.second[1];
    int max_age=(int)simu_data.second[2];
    int max_children=(int)simu_data.second[3];
    float mutation_rate=simu_data.second[4];
    float recycle_rate=simu_data.second[5];
    float sex_rate=simu_data.second[6];
    int max_notables=(int)simu_data.second[7];
    SlideCrossValidation cross_validation;
    switch((int)simu_data.second[8]){
        case 0:
            cross_validation=SlideCrossValidation::NONE;
            break;
        case 1:
            cross_validation=SlideCrossValidation::ROLLING_FORECASTING_ORIGIN;
            break;
        case 2:
            cross_validation=SlideCrossValidation::KFOLDS;
            break;
        case 3:
            cross_validation=SlideCrossValidation::TWENTY_PERCENT;
            break;
    }
    SlideMetric metric_mode;
    switch((int)simu_data.second[9]){
        case 0:
            metric_mode=SlideMetric::RAW_LOSS;
            break;
        case 1:
            metric_mode=SlideMetric::F1;
            break;
        case 2:
            metric_mode=SlideMetric::RECALL;
            break;
        case 3:
            metric_mode=SlideMetric::ACCURACY;
            break;
        case 4:
            metric_mode=SlideMetric::PRECISION;
            break;
    }
    int train_data_limit=(int)simu_data.second[10];
    int algorithm=(int)simu_data.second[11];
    SlideLabelEncoding label_encoding=SlideLabelEncoding::INT_CLASS;
    switch((int)simu_data.second[12]){
        case 0:
            label_encoding=SlideLabelEncoding::INT_CLASS;
            break;
        case 1:
            label_encoding=SlideLabelEncoding::NEURON_BY_NEURON;
            break;
        case 2:
            label_encoding=SlideLabelEncoding::NEURON_BY_N_LOG_LOSS;
            break;
    }
    SPACE_SEARCH search_space = mongo->fetchEnvironmentData(environment_name);
    vector<int> cve_years;
    for(string y:str_cve_years){
        cve_years.push_back(stoi(y));
    }
    str_cve_years.clear();
    cout<<"data: "<<simu_data.first[1]<<" limit: "<<train_data_limit<<endl;
    cout<<"population_start_size: "<<population_start_size<<endl;
    cout<<"max_gens: "<<max_gens<<endl;
    cout<<"max_age: "<<max_age<<endl;
    cout<<"max_children: "<<max_children<<endl;
    cout<<"mutation_rate: "<<mutation_rate<<endl;
    cout<<"recycle_rate: "<<recycle_rate<<endl;
    cout<<"sex_rate: "<<sex_rate<<endl;
    cout<<"max_notables: "<<max_notables<<endl;
    cout<<"cross_validation: "<<static_cast<underlying_type<SlideCrossValidation>::type>(cross_validation)<<endl;
    cout<<"metric_mode: "<<static_cast<underlying_type<SlideMetric>::type>(metric_mode)<<endl;
    cout<<"label_encoding: "<<static_cast<underlying_type<SlideLabelEncoding>::type>(label_encoding)<<endl;
    cout<<"algorithm: "<<algorithm<<endl;
    cout<<"environment_name: "<<environment_name<<endl;
    cout<<"Int space search:\n";
    for(INT_SPACE_SEARCH int_s:search_space.first){
        cout<<"\tmin: "<<int_s.first<<" max: "<<int_s.second<<endl;
    }   
    cout<<"Float space search:\n";
    for(FLOAT_SPACE_SEARCH float_s:search_space.second){
        cout<<"\tmin: "<<float_s.first<<" max: "<<float_s.second<<endl;
    }    
    simu_data.first.clear();
    simu_data.second.clear();
    cout<<"Parsed genetic settings...OK\n";
    cout<<"Loading CVE data...\n";
    vector<pair<vector<int>, vector<float>>> train_data = mongo->loadCvesFromYears(cve_years, train_data_limit).second;
    train_data=encodeData(train_data,label_encoding);
    train_data=Utils::balanceSingleLabelDataset(train_data);
    cout<<"Loaded CVE data...OK\n";
    const bool shuffle_train_data=false;

    const int rehash=6400;
    const int rebuild=128000;

    const int border_sparsity=1; // first and last layers
    const int input_size=train_data[0].second.size();
    const int output_size=train_data[0].first.size();
    const bool adam_optimizer=true;
    const bool use_neural_genome=true;
    const bool search_maximum=(metric_mode!=SlideMetric::RAW_LOSS);
    NeuralGenome::CACHE_WEIGHTS=cache_genetic;
    auto train_callback = [&](Genome *self) -> float {
        auto self_neural=dynamic_cast<NeuralGenome*>(self);
        tuple<Slide*,int,function<void()>> net=self_neural->buildSlide(self->getDna(),input_size,output_size,label_encoding,rehash,rebuild,border_sparsity,metric_mode,shuffle_train_data,cross_validation,adam_optimizer);
        if (self_neural->hasWeights()){
            get<0>(net)->setWeights(self_neural->getWeights());
            self_neural->clearWeights();
            get<0>(net)->eagerInit();
        }
        vector<pair<float,float>> metric=get<0>(net)->train(train_data,get<1>(net));
        self_neural->setWeights(get<0>(net)->getWeights());
        delete get<0>(net); // free memory
        get<2>(net)(); // free memory
        float output=0;
        for(pair<float,float> l:metric){
            if (cross_validation!=SlideCrossValidation::NONE){
                output+=l.second; // use validation
            }else{
                output+=l.first; // use validation
            }
        }
        output/=metric.size();
        return output;
    };
    auto after_gen_callback = [&](int pop_size,int g,float best_out,long timestamp_ms,vector<Genome*> population,HallOfFame *hall_of_fame) -> void {
        if (hall_of_fame){
            cout<<"\tStoring Hall of Fame Best Individuals..."<<endl;
            mongo->updateBestOnGeneticSimulation(simulation_id,hall_of_fame->getBest(),Utils::getStrNow());
            cout<<"\tStored Hall of Fame Best Individuals...OK"<<endl;
        }
        cout<<"\tStoring generation metadata..."<<endl;
        mongo->appendResultOnGeneticSimulation(simulation_id,pop_size,g,best_out,timestamp_ms);
        cout<<"\tStored generation metadata...OK"<<endl;
        if ((!store_genetic_populations_only_on_last_gen)||g==max_gens){
            cout<<"\tStoring population..."<<endl;
            mongo->clearPopulationNeuralGenomeVector(population_id,Utils::getStrNow());
            for (Genome* g:population){
                mongo->addToPopulationNeuralGenomeVector(population_id,dynamic_cast<NeuralGenome*>(g),Utils::getStrNow());
            }
            cout<<"\tStored population...OK"<<endl;
        }
    };
    mongo->clearResultOnGeneticSimulation(simulation_id);
    mongo->claimGeneticSimulation(simulation_id,Utils::getStrNow(),Utils::getHostname());
    HallOfFame* elite=new HallOfFame(max_notables, search_maximum);
    GeneticAlgorithm* ga;
    switch(algorithm){
        default: // 0
            ga=new EnchancedGenetic(max_children,max_age,mutation_rate,sex_rate,recycle_rate);
            break;
        case 1:
            ga=new StandardGenetic(mutation_rate,sex_rate);
            break;
    }
    PopulationManager enchanced_population=PopulationManager(ga,search_space,train_callback,population_start_size,search_maximum,use_neural_genome,true,after_gen_callback);
    enchanced_population.setHallOfFame(elite);
    cout<<"Starting natural selection..."<<endl;
    enchanced_population.naturalSelection(max_gens,true);
    cout<<"Finished natural selection...OK"<<endl;
    cout<<"Best loss ("<<elite->getBest().second<<"): "<<elite->getBest().first<<endl;
    mongo->clearHallOfFameNeuralGenomeVector(hall_of_fame_id,Utils::getStrNow());
    for (Genome* g:elite->getNotables()){
        mongo->addToHallOfFameNeuralGenomeVector(hall_of_fame_id,dynamic_cast<NeuralGenome*>(g),Utils::getStrNow());
    }
    mongo->finishGeneticSimulation(simulation_id,Utils::getStrNow());
    cout<<"Runned genetic simulation "+simulation_id+"...OK\n";
    delete elite;
}

void trainNeuralNetwork(string independent_net_id,bool load, bool just_train){
    cout<<"Training neural network "+independent_net_id+"...\n";
    cout<<"Parsing training settings...\n";
    pair<vector<string>,vector<int>> train_mdata=mongo->fetchNeuralNetworkTrainMetadata(independent_net_id);
    string hyper_name=train_mdata.first[0];
    vector<string> str_cve_years_train=Utils::splitString(train_mdata.first[1],",");
    vector<string> str_cve_years_test;
    if (!train_mdata.first[2].empty()){
        str_cve_years_test=Utils::splitString(train_mdata.first[2],",");
    }
    int epochs=train_mdata.second[0];
    SlideCrossValidation cross_validation=SlideCrossValidation::NONE;
    switch(train_mdata.second[1]){
        case 0:
            cross_validation=SlideCrossValidation::NONE;
            break;
        case 1:
            cross_validation=SlideCrossValidation::ROLLING_FORECASTING_ORIGIN;
            break;
        case 2:
            cross_validation=SlideCrossValidation::KFOLDS;
            break;
        case 3:
            cross_validation=SlideCrossValidation::TWENTY_PERCENT;
            break;
    }
    SlideMetric train_metric=SlideMetric::RAW_LOSS;
    switch(train_mdata.second[2]){
        case 0:
            train_metric=SlideMetric::RAW_LOSS;
            break;
        case 1:
            train_metric=SlideMetric::F1;
            break;
        case 2:
            train_metric=SlideMetric::RECALL;
            break;
        case 3:
            train_metric=SlideMetric::ACCURACY;
            break;
        case 4:
            train_metric=SlideMetric::PRECISION;
            break;
    }
    // SlideMetric test_metric=SlideMetric::RAW_LOSS; // useless
    int train_limit=train_mdata.second[4];
    int test_limit=train_mdata.second[5];
    vector<int> cve_years_train;
    for(string y:str_cve_years_train){
        cve_years_train.push_back(stoi(y));
    }
    str_cve_years_train.clear();
    Hyperparameters* hyper=mongo->fetchHyperparametersData(hyper_name);
    cout<<"Parsed training settings...OK\n";
    cout<<"Loading CVE data...\n";
    vector<pair<vector<int>, vector<float>>> train_data = mongo->loadCvesFromYears(cve_years_train, train_limit).second;
    train_data=encodeData(train_data,hyper->label_type);
    train_data=Utils::balanceSingleLabelDataset(train_data);
    // train_data=mongo->filterFeatures(train_data,vector<string> {"reference_type_","vendor_","description_"});
    cout<<"Loaded CVE data...OK\n";
    size_t maxNodes=(size_t)train_data[0].first.size();
    cout<<"train_metric: "<<static_cast<underlying_type<SlideMetric>::type>(train_metric)<<endl;
    cout<<"cross_validation: "<<static_cast<underlying_type<SlideCrossValidation>::type>(cross_validation)<<endl;
    cout<<"epochs: "<<epochs<<endl;
    cout<<"train data: "<<train_mdata.first[1]<<" limit: "<<train_limit<<endl;
    cout<<"test data: "<<train_mdata.first[2]<<" limit: "<<test_limit<<endl;
    cout<<"hyper_name: "<<hyper_name<<endl;
    cout<<"\tbatch_size: "<<hyper->batch_size<<endl;
    cout<<"\talpha: "<<hyper->alpha<<endl;
    cout<<"\tshuffle: "<<hyper->shuffle<<endl;
    cout<<"\tadam: "<<hyper->adam<<endl;
    cout<<"\trehash: "<<hyper->rehash<<endl;
    cout<<"\trebuild: "<<hyper->rebuild<<endl;
    cout<<"\tlabel_type: "<<static_cast<underlying_type<SlideLabelEncoding>::type>(hyper->label_type)<<endl;
    cout<<"\tlayers: "<<hyper->layers<<endl;
    for(int i=0;i<hyper->layers;i++){
        cout<<"\t\tlayer_sizes["<<i<<"]: "<<hyper->layer_sizes[i]<<endl;
        if ((size_t)hyper->layer_sizes[i]>maxNodes){
                maxNodes=(size_t)hyper->layer_sizes[i];
        }
    }
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\trange_pow["<<i<<"]: "<<hyper->range_pow[i]<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tK["<<i<<"]: "<<hyper->K[i]<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tL["<<i<<"]: "<<hyper->L[i]<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tnode_types["<<i<<"]: "<<static_cast<underlying_type<NodeType>::type>(hyper->node_types[i])<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tsparcity["<<i<<"]: "<<hyper->sparcity[i]<<endl;
    train_mdata.first.clear();
    train_mdata.second.clear();
    mongo->claimNeuralNetTrain(independent_net_id,Utils::getStrNow(),Utils::getHostname());
    const bool print_deltas=true;
    map<string, vector<float>> trained_weights;
    Slide* slide;
    if(load){
        trained_weights=mongo->loadWeightsFromNeuralNet(independent_net_id);
    }else{
        cout<<"Creating network...\n";
        slide=new Slide(hyper->layers,hyper->layer_sizes,hyper->node_types,train_data[0].second.size(),train_data[0].first.size(),hyper->alpha,hyper->batch_size,hyper->adam,hyper->label_type,hyper->range_pow,hyper->K,hyper->L,hyper->sparcity,hyper->rehash,hyper->rebuild,train_metric,train_metric,hyper->shuffle,cross_validation,SlideMode::SAMPLING,SlideHashingFunction::DENSIFIED_WTA,print_deltas,maxNodes);
        slide->eagerInit();
        cout<<"Created network...OK\n";
        cout<<"Training network...\n";
        vector<pair<float,float>> train_metrics=slide->train(train_data,epochs);
        cout<<"Trained network...OK\n";
        cout<<"Writing weights...\n";
        trained_weights=slide->getWeights();
        mongo->appendTMetricsOnNeuralNet(independent_net_id,train_metrics);
        mongo->appendWeightsOnNeuralNet(independent_net_id,trained_weights);
        cout<<"Wrote weights...OK\n";
        delete slide;
    }
    if (!just_train){
        Hyperparameters* hyper_2=hyper->clone();
        cout<<"Creating eval network...\n";
        Slide* slide_2=new Slide(hyper_2->layers,hyper_2->layer_sizes,hyper_2->node_types,train_data[0].second.size(),train_data[0].first.size(),hyper_2->alpha,hyper_2->batch_size,hyper_2->adam,hyper_2->label_type,hyper_2->range_pow,hyper_2->K,hyper_2->L,hyper_2->sparcity,hyper_2->rehash,hyper_2->rebuild,train_metric,train_metric,hyper_2->shuffle,cross_validation,SlideMode::SAMPLING,SlideHashingFunction::DENSIFIED_WTA,print_deltas,maxNodes);
        slide_2->setWeights(trained_weights);
        slide_2->eagerInit();
        cout<<"Created eval network...OK\n";
        cout<<"Evaluating for statistics...\n";
        vector<vector<pair<int,float>>> train_predicted=slide_2->evalData(train_data).second;
        snn_stats train_stats=Utils::statisticalAnalysis(train_data,train_predicted);
        cout<<"Evaluated for statistics...OK\n";
        if (verbose){
            Utils::compareAndPrintLabel(train_data,train_predicted);
            Utils::printStats(train_stats);
        }
        cout<<"Writing results...\n";
        mongo->appendStatsOnNeuralNet(independent_net_id,"train_stats",train_stats);
        if (str_cve_years_test.size()>0){
            for (pair<vector<int>, vector<float>> v:train_data){
                v.first.clear();
                v.second.clear();
            }
            train_data.clear();
            vector<int> cve_years_test;
            for(string y:str_cve_years_test){
                cve_years_test.push_back(stoi(y));
            }
            str_cve_years_test.clear();
            vector<pair<vector<int>, vector<float>>> test_data = mongo->loadCvesFromYears(cve_years_test, test_limit).second;
            test_data=encodeData(test_data,hyper_2->label_type);
            vector<vector<pair<int,float>>> test_predicted=slide_2->evalData(test_data).second;
            snn_stats test_stats=Utils::statisticalAnalysis(test_data,train_predicted);
            if (verbose){
                Utils::printStats(test_stats);
            }
            mongo->appendStatsOnNeuralNet(independent_net_id,"test_stats",test_stats);
        }
        cout<<"Wrote results...OK\n";
        mongo->finishNeuralNetTrain(independent_net_id,Utils::getStrNow());
        cout<<"Trained neural network "+independent_net_id+"...OK\n";
        delete hyper_2;
        delete slide_2;
    }
    delete hyper;
}

void evalNeuralNetwork(string independent_net_id, string result_id, string eval_data){
    cout<<"Evaluating neural network "+independent_net_id+" for data: "+eval_data+"...\n";
    vector<pair<vector<int>, vector<float>>> cve_data;
    vector<string> cve_ids;
    cout<<"Parsing evaluate settings...\n";
    pair<vector<string>,vector<int>> eval_mdata=mongo->fetchNeuralNetworkTrainMetadata(independent_net_id);
    string hyper_name=eval_mdata.first[0];
    SlideCrossValidation cross_validation=SlideCrossValidation::NONE;
    switch(eval_mdata.second[1]){
        case 0:
            cross_validation=SlideCrossValidation::NONE;
            break;
        case 1:
            cross_validation=SlideCrossValidation::ROLLING_FORECASTING_ORIGIN;
            break;
        case 2:
            cross_validation=SlideCrossValidation::KFOLDS;
            break;
        case 3:
            cross_validation=SlideCrossValidation::TWENTY_PERCENT;
            break;
    }
    SlideMetric test_metric=SlideMetric::RAW_LOSS;
    switch(eval_mdata.second[2]){
        case 0:
            test_metric=SlideMetric::RAW_LOSS;
            break;
        case 1:
            test_metric=SlideMetric::F1;
            break;
        case 2:
            test_metric=SlideMetric::RECALL;
            break;
        case 3:
            test_metric=SlideMetric::ACCURACY;
            break;
        case 4:
            test_metric=SlideMetric::PRECISION;
            break;
    }
    Hyperparameters* hyper=mongo->fetchHyperparametersData(hyper_name);
    cout<<"Parsed evaluate settings...OK\n";
    cout<<"Loading CVE data...\n";
    if (eval_data.rfind("CVE", 0) == 0) {
        cve_data = mongo->loadCveFromId(eval_data);
        cve_ids.push_back(eval_data);
    }else{
        int limit=0;
        string::size_type pos=eval_data.find(':');
        if (pos!=string::npos){
            limit=stof(eval_data.substr(pos+1,eval_data.size()-pos-1));
            eval_data=eval_data.substr(0,pos);
        }
        vector<string> str_cve_years=Utils::splitString(eval_data,",");
        vector<int> cve_years;
        for(string y:str_cve_years){
            cve_years.push_back(stoi(y));
        }
        str_cve_years.clear();
        pair<vector<string>,vector<pair<vector<int>, vector<float>>>> loaded_cves=mongo->loadCvesFromYears(cve_years, limit);
        cve_ids = loaded_cves.first;
        cve_data = loaded_cves.second;
    }
    cve_data=encodeData(cve_data,hyper->label_type);
    cout<<"Loaded CVE data...OK\n";
    // cve_data=mongo->filterFeatures(cve_data,vector<string> {"reference_type_","vendor_","description_"});
    size_t maxNodes=(size_t)cve_data[0].first.size();
    cout<<"test_metric: "<<static_cast<underlying_type<SlideMetric>::type>(test_metric)<<endl;
    cout<<"cross_validation: "<<static_cast<underlying_type<SlideCrossValidation>::type>(cross_validation)<<endl;
    cout<<"eval data: "<<eval_data;
    cout<<"hyper_name: "<<hyper_name<<endl;
    cout<<"\tbatch_size: "<<hyper->batch_size<<endl;
    cout<<"\talpha: "<<hyper->alpha<<endl;
    cout<<"\tshuffle: "<<hyper->shuffle<<endl;
    cout<<"\tadam: "<<hyper->adam<<endl;
    cout<<"\trehash: "<<hyper->rehash<<endl;
    cout<<"\trebuild: "<<hyper->rebuild<<endl;
    cout<<"\tlabel_type: "<<static_cast<underlying_type<SlideLabelEncoding>::type>(hyper->label_type)<<endl;
    cout<<"\tlayers: "<<hyper->layers<<endl;
    for(int i=0;i<hyper->layers;i++){
        cout<<"\t\tlayer_sizes["<<i<<"]: "<<hyper->layer_sizes[i]<<endl;
        if ((size_t)hyper->layer_sizes[i]>maxNodes){
                maxNodes=(size_t)hyper->layer_sizes[i];
        }
    }
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\trange_pow["<<i<<"]: "<<hyper->range_pow[i]<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tK["<<i<<"]: "<<hyper->K[i]<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tL["<<i<<"]: "<<hyper->L[i]<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tnode_types["<<i<<"]: "<<static_cast<underlying_type<NodeType>::type>(hyper->node_types[i])<<endl;
    for(int i=0;i<hyper->layers;i++)
        cout<<"\t\tsparcity["<<i<<"]: "<<hyper->sparcity[i]<<endl;
    eval_mdata.first.clear();
    eval_mdata.second.clear();
    const bool print_deltas=true;
    cout<<"Creating network...\n";
    Slide* slide=new Slide(hyper->layers,hyper->layer_sizes,hyper->node_types,cve_data[0].second.size(),cve_data[0].first.size(),hyper->alpha,hyper->batch_size,hyper->adam,hyper->label_type,hyper->range_pow,hyper->K,hyper->L,hyper->sparcity,hyper->rehash,hyper->rebuild,test_metric,test_metric,hyper->shuffle,cross_validation,SlideMode::SAMPLING,SlideHashingFunction::DENSIFIED_WTA,print_deltas,maxNodes);
    slide->setWeights(mongo->loadWeightsFromNeuralNet(independent_net_id));
    slide->eagerInit();
    cout<<"Created network...OK\n";
    cout<<"Evaluating data...\n";
    pair<int,vector<vector<pair<int,float>>>> predicted = slide->evalData(cve_data);
    snn_stats cve_stats=Utils::statisticalAnalysis(cve_data,predicted.second);
    cout<<"Evaluated data...OK\n";
    if (verbose){
        Utils::compareAndPrintLabel(cve_data,predicted.second);
        Utils::printStats(cve_stats);
    }
    cout<<"Writing results...\n";
    mongo->storeEvalNeuralNetResult(result_id,predicted.first,cve_ids,predicted.second,cve_data,cve_stats);
    cout<<"Wrote results...OK\n";
    cout<<"Evaluated neural network "+independent_net_id+"...\n";
    delete slide;
    delete hyper;
}

int main(int argc, char* argv[]) {
    bool run_test;
    bool run_genetic;
    bool run_train_net;
    bool run_eval_net;
    bool just_print_args;
    bool load_weights_instead_of_training;
    bool train_without_eval;
    int test_function;
    string simulation_id;
    string independent_net_id;
    string result_id;
    string eval_data;
    try {
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "print usage message")
            ("trap-signals", boost::program_options::bool_switch()->default_value(false), "trap signals to try to print stacktrace in failure scenarios")
            ("ignore-mongo", boost::program_options::bool_switch()->default_value(false), "skip mongo connection (e.g. when there is no server)")
            ("set-stack-size", boost::program_options::bool_switch()->default_value(false), "sets stack max size programmatically")
            ("test", boost::program_options::bool_switch()->default_value(false), "run test functions instead of main ones")
            ("run-genetic", boost::program_options::bool_switch()->default_value(false), "run genetic population")
            ("train-neural", boost::program_options::bool_switch()->default_value(false), "train smart neural network")
            ("eval-neural", boost::program_options::bool_switch()->default_value(false), "eval smart neural network")
            ("debug-args", boost::program_options::bool_switch()->default_value(false), "just print arguments (DEBUG)")
            ("continue", boost::program_options::bool_switch()->default_value(false), "load weigths instead of training them (useful to when exceptions happens)")
            ("just-train", boost::program_options::bool_switch()->default_value(false), "train weigths and then quit (useful avoid exceptions)")
            ("verbose-neural", boost::program_options::bool_switch()->default_value(false), "a lot of printings for neural training")
            ("do-not-cache-genetic", boost::program_options::bool_switch()->default_value(false), "avoid disk cache during genetic algorithm")
            ("single-thread", boost::program_options::bool_switch()->default_value(false), "use a single thread to run V-REx")
            ("test-function", boost::program_options::value<int>()->default_value(0), "specify the test function to run <function id>:\n\t1 - testCsvRead\n\t2 - testMongo\n\t3 - testSlide_IntLabel\n\t4 - testSlide_NeuronByNeuronLabel\n\t5 - testStdGeneticsOnMath\n\t6 - testEnchancedGeneticsOnMath\n\t7 - testSlide_Validation\n\t8 - testGeneticallyTunedNeuralNetwork\n\t9 - testMongoCveRead\n\t10 - testSmartNeuralNetwork_cveData\n\t11 - testGeneticallyTunedSmartNeuralNetwork_cveData")
            ("simulation-id", boost::program_options::value<string>()->default_value(""), "mongo genetic simulation id to fetch data <id>")
            ("network-id", boost::program_options::value<string>()->default_value(""), "mongo neural network id to fetch data <id>")
            ("eval-result-id", boost::program_options::value<string>()->default_value(""), "mongo neural network result id to write results <id>")
            ("eval-data", boost::program_options::value<string>()->default_value(""), "data info to be used during neural network eval <eval data>")
            ("custom-mongo-host", boost::program_options::value<string>()->default_value(""), "mongo ipv4 address <ip> (default 127.0.0.1 outside docker and \'mongo\' inside docker)")
        ;
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")){  
            cout<<desc<<endl;
            return 0;
        }
        conflicting_options(vm,"test","run-genetic");
        conflicting_options(vm,"test","train-neural");
        conflicting_options(vm,"test","eval-neural");
        conflicting_options(vm,"run-genetic","train-neural");
        conflicting_options(vm,"run-genetic","eval-neural");
        conflicting_options(vm,"train-neural","eval-neural");
        conflicting_options(vm,"continue","just-train");
        option_dependency(vm,"test","test-function");
        option_dependency(vm,"run-genetic","simulation-id");
        option_dependency(vm,"train-neural","network-id");
        option_dependency(vm,"eval-neural","network-id");
        option_dependency(vm,"eval-neural","eval-result-id");
        option_dependency(vm,"eval-neural","eval-data");
        trap_signals=vm["trap-signals"].as<bool>();
        connect_mongo=!vm["ignore-mongo"].as<bool>();
        cache_genetic=!vm["do-not-cache-genetic"].as<bool>();
        single_thread=vm["single-thread"].as<bool>();
        set_stack_size=vm["set-stack-size"].as<bool>();
        load_weights_instead_of_training=vm["continue"].as<bool>();
        train_without_eval=vm["just-train"].as<bool>();
        verbose=vm["verbose-neural"].as<bool>();
        run_test=vm["test"].as<bool>();
        run_genetic=vm["run-genetic"].as<bool>();
        run_train_net=vm["train-neural"].as<bool>();
        run_eval_net=vm["eval-neural"].as<bool>();
        just_print_args=vm["debug-args"].as<bool>();
        test_function=vm["test-function"].as<int>();
        simulation_id=vm["simulation-id"].as<string>();
        independent_net_id=vm["network-id"].as<string>();
        result_id=vm["eval-result-id"].as<string>();
        eval_data=vm["eval-data"].as<string>();
        custom_mongo_ip=vm["custom-mongo-host"].as<string>();
        if (!run_test&&!run_genetic&&!run_train_net&&!run_eval_net&&!just_print_args){
            cout<<"No function argument found, please use --help for tips\n";
            cout<<"Nothing to do, exiting...\n";
            return 0;
        }
    }
    catch(exception& e) {
        cerr<<e.what()<<endl;
        return 2;
    }
    if (just_print_args){
        cout<<"trap_signals: "<<trap_signals<<endl;
        cout<<"connect_mongo: "<<connect_mongo<<endl;
        cout<<"set_stack_size: "<<set_stack_size<<endl;
        cout<<"cache_genetic: "<<cache_genetic<<endl;
        cout<<"single_thread: "<<single_thread<<endl;
        cout<<endl;
        cout<<"run_test: "<<run_test<<endl;
        cout<<"run_genetic: "<<run_genetic<<endl;
        cout<<"run_train_net: "<<run_train_net<<" continue: "<<load_weights_instead_of_training<<" just train: "<<train_without_eval<<endl;
        cout<<"run_eval_net: "<<run_eval_net<<endl;
        cout<<endl;
        cout<<"test_function: "<<test_function<<endl;
        cout<<"simulation_id: "<<simulation_id<<endl;
        cout<<"independent_net_id: "<<independent_net_id<<endl;
        cout<<"result_id: "<<result_id<<endl;
        cout<<"eval_data: "<<eval_data<<endl;
        return 0;
    }
    if (single_thread){
        Slide::MAX_THREADS=1;
    }
    setup();
    if (run_test){
        test(test_function);
    }else if (run_genetic){
        runGeneticSimulation(simulation_id);
    }else if (run_train_net){
        trainNeuralNetwork(independent_net_id,load_weights_instead_of_training,train_without_eval);
    }else if (run_eval_net){
        evalNeuralNetwork(independent_net_id,result_id,eval_data);
    }
    tearDown();
    return 0;
}
