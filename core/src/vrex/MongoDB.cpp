#include "MongoDB.hpp"

mongocxx::instance MongoDB::inst{}; // This should be done only once.

MongoDB::MongoDB(string host, string user, string password, int port) {
    // to fix mongo driver run the commands below if needed
    // sudo ln -s /usr/local/lib/libmogocxx.so.3.4.0 libmongocxx.so._noabi
    // sudo ln -s /usr/local/lib/libmongocxx.so._noabi /usr/local/lib/libmongocxx.so
    string conn_str="mongodb://"+user+":"+password+"@"+host+":"+to_string(port)+"/?authSource=admin";
    client=MongoDB::getClient(conn_str);
    weigths_bucket=getDB("neural_db").gridfs_bucket();
}

MongoDB::MongoDB(const MongoDB& orig) {
}

MongoDB::~MongoDB() {
    // client.~client(); // causes to crash
}

mongocxx::client MongoDB::getClient(const string &conn_str){
    mongocxx::client client{mongocxx::uri{conn_str}};
    return client;
}

mongocxx::database MongoDB::getDB(const string &db_name){
    return client[db_name];
}

mongocxx::collection MongoDB::getCollection(const mongocxx::database &db, const string &col_name){
    return db[col_name];
}

pair<string,pair<vector<int>, vector<float>>> MongoDB::bsonToDatasetEntry(const bsoncxx::v_noabi::document::view bson){
    int features_size;
    if (Utils::runningOnDockerContainer()){
        features_size=878; // I used diffferent data on production environment
    }else{
        features_size=863;
    }
    const int labels_size=8;
    
    string cve_id="";
    bsoncxx::document::element cve_el = bson["cve"];
    cve_id=getStringFromEl(cve_el);
    vector<float> features;
    bsoncxx::document::element features_el = bson["features"];
    if (features_el.type() == bsoncxx::type::k_document){
        bsoncxx::document::view features_vw=features_el.get_document().view();
        map<string,float> features_map;
        for (auto el:features_vw){
            string key=el.key().data();
            features_map[key]=getFloatFromEl(el);
        }
        if((int)features_map.size()!=features_size){
            throw runtime_error("Error features sizes for "+cve_id+" should be: "+to_string(features_size)+" but is: "+to_string(features_map.size())+"\n");
        }
        for(map<string,float>::const_iterator it=features_map.begin();it!=features_map.end();it++){
            features.push_back(it->second);
        }
    }else{
        throw runtime_error("Error unkown features type for "+cve_id+": "+bsoncxx::to_string(features_el.type())+"\n");
    }
    vector<int> labels;
    bsoncxx::document::element labels_el = bson["labels"];
    if (labels_el.type() == bsoncxx::type::k_document){
        bsoncxx::document::view labels_vw=labels_el.get_document().view();
        map<string,float> labels_map;
        for (auto el:labels_vw){
            string key=el.key().data();
            labels_map[key]=getFloatFromEl(el);
        }
        if((int)labels_map.size()!=labels_size){
            throw runtime_error("Error labels sizes for "+cve_id+" should be: "+to_string(labels_size)+" but is: "+to_string(labels_map.size())+"\n");
        }
        labels.push_back(((int) labels_map["exploits_has"]));
    }else{
        throw runtime_error("Error unkown labels type for "+cve_id+": "+bsoncxx::to_string(labels_el.type())+"\n");
    }
    return pair<string,pair<vector<int>, vector<float>>>(cve_id,pair<vector<int>, vector<float>>(labels,features));
}

pair<string,pair<vector<int>, vector<float>>> MongoDB::bsonToDatasetEntry(bsoncxx::stdx::optional<bsoncxx::document::value> opt_bson){
    if(!opt_bson) {
        throw runtime_error("Empty opt result");
    }
    return bsonToDatasetEntry(opt_bson->view());
}

pair<vector<string>,vector<pair<vector<int>, vector<float>>>> MongoDB::loadCvesFromYear(int year, int limit){
    bsoncxx::document::value query = document{}
        << "cve"<< open_document
            << "$regex" <<  "CVE-"+to_string(year)+"-.*"
        << close_document
    << finalize;
    mongocxx::options::find opts{};
    opts.sort(document{}<<"cve"<<1<<finalize);
    mongocxx::cursor cursor = getCollection(getDB("processed_data"),"dataset").find(query.view(),opts);
    int total=0;
    vector<string> cves;
    vector<pair<vector<int>, vector<float>>> data;
    for(auto&& doc:cursor) {
        pair<string,pair<vector<int>, vector<float>>> parsed_entry=bsonToDatasetEntry(doc);
        cves.push_back(parsed_entry.first);
        data.push_back(parsed_entry.second);
        if (++total>=limit && limit>0){
            break;
        }
    }
    return pair<vector<string>,vector<pair<vector<int>, vector<float>>>>(cves,data);
}

pair<vector<string>,vector<pair<vector<int>, vector<float>>>> MongoDB::loadCvesFromYears(vector<int> years, int limit){
    vector<string> cves;
    vector<pair<vector<int>, vector<float>>> data;
    for(int y:years){
        pair<vector<string>,vector<pair<vector<int>, vector<float>>>> parsed_entries=loadCvesFromYear(y,limit);
        cves.insert(cves.end(),parsed_entries.first.begin(),parsed_entries.first.end());
        data.insert(data.end(),parsed_entries.second.begin(),parsed_entries.second.end());
    }
    return pair<vector<string>,vector<pair<vector<int>, vector<float>>>>(cves,data);
}

pair<vector<string>,vector<float>> MongoDB::fetchGeneticSimulationData(string id){
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result=getCollection(getDB("genetic_db"),"simulations").find_one(document{} << "_id" << bsoncxx::oid{id} << finalize);
    string env_name;
    string train_data;
    string hall_of_fame_id;
    string population_id;
    float pop_start_size;
    float max_gens;
    float max_age;
    float max_children;
    float mutation_rate;
    float recycle_rate;
    float sex_rate;
    float max_notables;
    float cross_validation;
    float metric;
    float limit=0;
    float algorithm;
    float label_type;
    if(maybe_result) {
        bsoncxx::document::element env_name_el = maybe_result->view()["env_name"];
        env_name=getStringFromEl(env_name_el);
        bsoncxx::document::element train_data_el = maybe_result->view()["train_data"];
        train_data=getStringFromEl(train_data_el);
        bsoncxx::document::element hall_of_fame_id_el = maybe_result->view()["hall_of_fame_id"];
        hall_of_fame_id=getStringFromEl(hall_of_fame_id_el);
        bsoncxx::document::element population_id_el = maybe_result->view()["population_id"];
        population_id=getStringFromEl(population_id_el);

        bsoncxx::document::element pop_start_size_el = maybe_result->view()["pop_start_size"];
        pop_start_size=(float)getIntFromEl(pop_start_size_el);
        bsoncxx::document::element max_gens_el = maybe_result->view()["max_gens"];
        max_gens=(float)getIntFromEl(max_gens_el);
        bsoncxx::document::element max_age_el = maybe_result->view()["max_age"];
        max_age=(float)getIntFromEl(max_age_el);
        bsoncxx::document::element max_children_el = maybe_result->view()["max_children"];
        max_children=(float)getIntFromEl(max_children_el);
        bsoncxx::document::element mutation_rate_el = maybe_result->view()["mutation_rate"];
        mutation_rate=getFloatFromEl(mutation_rate_el);
        bsoncxx::document::element recycle_rate_el = maybe_result->view()["recycle_rate"];
        recycle_rate=getFloatFromEl(recycle_rate_el);
        bsoncxx::document::element sex_rate_el = maybe_result->view()["sex_rate"];
        sex_rate=getFloatFromEl(sex_rate_el);
        bsoncxx::document::element max_notables_el = maybe_result->view()["max_notables"];
        max_notables=(float)getIntFromEl(max_notables_el);
        bsoncxx::document::element cross_validation_el = maybe_result->view()["cross_validation"];
        cross_validation=(float)getIntFromEl(cross_validation_el);
        bsoncxx::document::element metric_el = maybe_result->view()["metric"];
        metric=(float)getIntFromEl(metric_el);
        bsoncxx::document::element algorithm_el = maybe_result->view()["algorithm"];
        algorithm=(float)getIntFromEl(algorithm_el);
        bsoncxx::document::element label_type_el = maybe_result->view()["label_type"];
        label_type=(float)getIntFromEl(label_type_el);
    }else{
        throw runtime_error("Unable to find simulation "+id);
    }
    string::size_type pos=train_data.find(':');
    if (pos!=string::npos){
        limit=stof(train_data.substr(pos+1,train_data.size()-pos-1));
        train_data=train_data.substr(0,pos);
    }
    vector<string> str_params;
    str_params.push_back(env_name);
    str_params.push_back(train_data);
    str_params.push_back(hall_of_fame_id);
    str_params.push_back(population_id);
    vector<float> float_params;
    float_params.push_back(pop_start_size);
    float_params.push_back(max_gens);
    float_params.push_back(max_age);
    float_params.push_back(max_children);
    float_params.push_back(mutation_rate);
    float_params.push_back(recycle_rate);
    float_params.push_back(sex_rate);
    float_params.push_back(max_notables);
    float_params.push_back(cross_validation);
    float_params.push_back(metric);
    float_params.push_back(limit);
    float_params.push_back(algorithm);
    float_params.push_back(label_type);

    return pair<vector<string>,vector<float>>(str_params,float_params);
}

void MongoDB::claimGeneticSimulation(string id,string currentDatetime, string hostname){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << "started_at" << open_document << "$ne" <<  NULL << close_document << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "started_at" <<  currentDatetime << close_document << finalize;
    getCollection(getDB("genetic_db"),"simulations").update_one(query.view(),update.view());

    query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    update=document{} << "$set" << open_document << "started_by" <<  hostname << close_document << finalize;
    getCollection(getDB("genetic_db"),"simulations").update_one(query.view(),update.view());
}

void MongoDB::finishGeneticSimulation(string id,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "finished_at" <<  currentDatetime << close_document << finalize;
    getCollection(getDB("genetic_db"),"simulations").update_one(query.view(),update.view());
}


void MongoDB::updateBestOnGeneticSimulation(string id, pair<float,int> candidate,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "updated_at" << currentDatetime << "best" <<  open_document << "output" <<  candidate.first << "at_gen" << candidate.second << close_document << close_document << finalize;
    getCollection(getDB("genetic_db"),"simulations").update_one(query.view(),update.view());   
}

void MongoDB::clearResultOnGeneticSimulation(string id){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "results" << open_array << close_array << close_document << finalize;
    getCollection(getDB("genetic_db"),"simulations").update_one(query.view(),update.view());
}

void MongoDB::appendResultOnGeneticSimulation(string id, int pop_size,int g,float best_out,long timestamp_ms){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$push" << open_document << "results" <<  open_document << "pop_size" << pop_size << "cur_gen" <<  g << "gen_best_out" << best_out << "delta_ms" << timestamp_ms << close_document << close_document << finalize;
    getCollection(getDB("genetic_db"),"simulations").update_one(query.view(),update.view());
}

SPACE_SEARCH MongoDB::fetchEnvironmentData(string name){
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result=getCollection(getDB("genetic_db"),"environments").find_one(document{} << "name" << name << finalize);
    INT_SPACE_SEARCH amount_of_layers;
    INT_SPACE_SEARCH epochs;
    INT_SPACE_SEARCH batch_size;
    INT_SPACE_SEARCH layer_size;
    INT_SPACE_SEARCH range_pow;
    INT_SPACE_SEARCH k_values;
    INT_SPACE_SEARCH l_values;
    INT_SPACE_SEARCH activation_funcs;
    FLOAT_SPACE_SEARCH alpha;
    FLOAT_SPACE_SEARCH sparcity;
    if(maybe_result) {
        bsoncxx::document::element space_search_el = maybe_result->view()["space_search"];
        if (space_search_el.type() == bsoncxx::type::k_document){
            bsoncxx::document::view space_search_vw=space_search_el.get_document().view();
            bsoncxx::document::view amount_of_layers_vw = space_search_vw["amount_of_layers"].get_document().view();
            bsoncxx::document::view epochs_vw = space_search_vw["epochs"].get_document().view();
            bsoncxx::document::view batch_size_vw = space_search_vw["batch_size"].get_document().view();
            bsoncxx::document::view layer_sizes_vw = space_search_vw["layer_sizes"].get_document().view();
            bsoncxx::document::view range_pow_vw = space_search_vw["range_pow"].get_document().view();
            bsoncxx::document::view K_vw = space_search_vw["K"].get_document().view();
            bsoncxx::document::view L_vw = space_search_vw["L"].get_document().view();
            bsoncxx::document::view activation_functions_vw = space_search_vw["activation_functions"].get_document().view();
            bsoncxx::document::view sparcity_vw = space_search_vw["sparcity"].get_document().view();
            bsoncxx::document::view alpha_vw = space_search_vw["alpha"].get_document().view();

            amount_of_layers = INT_SPACE_SEARCH(amount_of_layers_vw["min"].get_int32(),amount_of_layers_vw["max"].get_int32());
            epochs = INT_SPACE_SEARCH(epochs_vw["min"].get_int32(),epochs_vw["max"].get_int32());
            batch_size = INT_SPACE_SEARCH(batch_size_vw["min"].get_int32(),batch_size_vw["max"].get_int32());
            layer_size = INT_SPACE_SEARCH(layer_sizes_vw["min"].get_int32(),layer_sizes_vw["max"].get_int32());
            range_pow = INT_SPACE_SEARCH(range_pow_vw["min"].get_int32(),range_pow_vw["max"].get_int32());
            k_values = INT_SPACE_SEARCH(K_vw["min"].get_int32(),K_vw["max"].get_int32());
            l_values = INT_SPACE_SEARCH(L_vw["min"].get_int32(),L_vw["max"].get_int32());
            activation_funcs = INT_SPACE_SEARCH(activation_functions_vw["min"].get_int32(),activation_functions_vw["max"].get_int32());
            sparcity = FLOAT_SPACE_SEARCH(getFloatFromEl(sparcity_vw["min"]),getFloatFromEl(sparcity_vw["max"]));
            alpha = FLOAT_SPACE_SEARCH(getFloatFromEl(alpha_vw["min"]),getFloatFromEl(alpha_vw["max"]));
        }else{
            throw runtime_error("Error invalid type: "+bsoncxx::to_string(space_search_el.type())+"\n");
        }
    }
    return NeuralGenome::buildSlideNeuralNetworkSpaceSearch(amount_of_layers,epochs,alpha,batch_size,layer_size,range_pow,k_values,l_values,sparcity,activation_funcs);
}

void MongoDB::clearPopulationNeuralGenomeVector(string pop_id,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{pop_id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "updated_at" << currentDatetime << "neural_genomes" <<  open_array << close_array << close_document << finalize;
    getCollection(getDB("neural_db"),"populations").update_one(query.view(),update.view());
}

void MongoDB::addToPopulationNeuralGenomeVector(string pop_id,NeuralGenome* ng,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{pop_id} << finalize;
    bsoncxx::document::value update=document{} << "$push" << open_document << "neural_genomes" << castNeuralGenomeToBson(ng) << close_document << "$set" << open_document << "updated_at" << currentDatetime << close_document << finalize;
    getCollection(getDB("neural_db"),"populations").update_one(query.view(),update.view());
}

void MongoDB::clearHallOfFameNeuralGenomeVector(string hall_id,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{hall_id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "updated_at" << currentDatetime << "neural_genomes" <<  open_array << close_array << close_document << finalize;
    getCollection(getDB("neural_db"),"hall_of_fame").update_one(query.view(),update.view());
}

void MongoDB::addToHallOfFameNeuralGenomeVector(string hall_id,NeuralGenome* ng,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{hall_id} << finalize;
    bsoncxx::document::value update=document{} << "$push" << open_document << "neural_genomes" << castNeuralGenomeToBson(ng) << close_document << "$set" << open_document << "updated_at" << currentDatetime << close_document << finalize;
    getCollection(getDB("neural_db"),"hall_of_fame").update_one(query.view(),update.view());
}

bsoncxx::document::value MongoDB::castNeuralGenomeToBson(NeuralGenome* ng,bool store_weights){
    if (ng){
        pair<vector<int>,vector<float>> dna = ng->getDna();
        string int_dna="[ ";
        for(size_t i=0;i<dna.first.size();){
            int_dna+=to_string(dna.first[i]);
            if (++i<dna.first.size()){
                int_dna+=", ";
            }
        }
        int_dna+=" ]";
        string float_dna="[ ";
        for(size_t i=0;i<dna.second.size();){
            float_dna+=to_string(dna.second[i]);
            if (++i<dna.second.size()){
                float_dna+=", ";
            }
        }
        float_dna+=" ]";
        if (store_weights){
            bsoncxx::document::value full=document{}
            <<"int_dna"<<int_dna
            <<"float_dna"<<float_dna
            <<"output"<<ng->getOutput()
            <<"weights"<<Utils::stringToBase65(Utils::serializeWeigthsToStr(ng->getWeights()))
            << finalize;
            ng->clearWeightsIfCached();
            return full;
        }else{
            bsoncxx::document::value full=document{}
            <<"int_dna"<<int_dna
            <<"float_dna"<<float_dna
            <<"output"<<ng->getOutput()
            << finalize;
            return full;
        }
    }else{
        bsoncxx::document::value empty=document{} << finalize;
        return empty;
    }
}

pair<vector<string>,vector<int>> MongoDB::fetchNeuralNetworkTrainMetadata(string id){
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result=getCollection(getDB("neural_db"),"independent_net").find_one(document{} << "_id" << bsoncxx::oid{id} << finalize);
    string hyper_name;
    string train_data;
    string test_data;
    int epochs;
    int cross_validation;
    int train_metric;
    int test_metric;
    int train_limit=0;
    int test_limit=0;
    if(maybe_result) {
        bsoncxx::document::element hyper_name_el = maybe_result->view()["hyperparameters_name"];
        hyper_name=getStringFromEl(hyper_name_el);
        bsoncxx::document::element train_data_el = maybe_result->view()["train_data"];
        train_data=getStringFromEl(train_data_el);
        bsoncxx::document::element test_data_el = maybe_result->view()["test_data"];
        test_data=getStringFromEl(test_data_el);
        bsoncxx::document::element epochs_el = maybe_result->view()["epochs"];
        epochs=getIntFromEl(epochs_el);
        bsoncxx::document::element cross_validation_el = maybe_result->view()["cross_validation"];
        cross_validation=getIntFromEl(cross_validation_el);
        bsoncxx::document::element train_metric_el = maybe_result->view()["train_metric"];
        train_metric=getIntFromEl(train_metric_el);
        bsoncxx::document::element test_metric_el = maybe_result->view()["test_metric"];
        test_metric=getIntFromEl(test_metric_el);
    }else{
        throw runtime_error("Unable to find independent network "+id);
    }
    string::size_type pos=train_data.find(':');
    if (pos!=string::npos){
        train_limit=stof(train_data.substr(pos+1,train_data.size()-pos-1));
        train_data=train_data.substr(0,pos);
    }
    pos=test_data.find(':');
    if (pos!=string::npos){
        test_limit=stof(test_data.substr(pos+1,test_data.size()-pos-1));
        test_data=test_data.substr(0,pos);
    }

    vector<string> str_params;
    str_params.push_back(hyper_name);
    str_params.push_back(train_data);
    str_params.push_back(test_data);
    vector<int> int_params;
    int_params.push_back(epochs);
    int_params.push_back(cross_validation);
    int_params.push_back(train_metric);
    int_params.push_back(test_metric);
    int_params.push_back(train_limit);
    int_params.push_back(test_limit);

    return pair<vector<string>,vector<int>>(str_params,int_params);
}

void MongoDB::claimNeuralNetTrain(string id,string currentDatetime, string hostname){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << "started_at" << open_document << "$ne" <<  NULL << close_document << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "started_at" <<  currentDatetime << close_document << finalize;
    getCollection(getDB("neural_db"),"independent_net").update_one(query.view(),update.view());

    query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    update=document{} << "$set" << open_document << "started_by" <<  hostname << close_document << finalize;
    getCollection(getDB("neural_db"),"independent_net").update_one(query.view(),update.view());
}

void MongoDB::finishNeuralNetTrain(string id,string currentDatetime){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "finished_at" <<  currentDatetime << close_document << finalize;
    getCollection(getDB("neural_db"),"independent_net").update_one(query.view(),update.view());
}

Hyperparameters* MongoDB::fetchHyperparametersData(string name){
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result=getCollection(getDB("neural_db"),"snn_hyperparameters").find_one(document{} << "name" << name << finalize);
    if(!maybe_result) {
        throw runtime_error("Hyperparameters not found!");
    }
    int layers=maybe_result->view()["layers"].get_int32();
    Hyperparameters* hyper=new Hyperparameters(layers);
    hyper->batch_size=maybe_result->view()["batch_size"].get_int32();
    hyper->alpha=getFloatFromEl(maybe_result->view()["alpha"]);
    hyper->shuffle=maybe_result->view()["shuffle"].get_bool();
    hyper->adam=maybe_result->view()["adam"].get_bool();
    hyper->rehash=maybe_result->view()["rehash"].get_int32();
    hyper->rebuild=maybe_result->view()["rebuild"].get_int32();
    switch(maybe_result->view()["label_type"].get_int32()){
        case 0:
            hyper->label_type=SlideLabelEncoding::INT_CLASS;
            break;
        case 1:
            hyper->label_type=SlideLabelEncoding::NEURON_BY_NEURON;
            break;
        case 2:
            hyper->label_type=SlideLabelEncoding::NEURON_BY_N_LOG_LOSS;
            break;
    }
    bsoncxx::array::view layer_size_arr {maybe_result->view()["layer_sizes"].get_array().value};
    int idx=0;
    for (const bsoncxx::array::element& el:layer_size_arr){
        if(idx>=layers){
            break;
        }
        hyper->layer_sizes[idx]=el.get_int32();
        idx++;
    }
    bsoncxx::array::view range_pow_arr {maybe_result->view()["range_pow"].get_array().value};
    idx=0;
    for (const bsoncxx::array::element& el:range_pow_arr){
        if(idx>=layers){
            break;
        }
        hyper->range_pow[idx]=el.get_int32();
        idx++;
    }
    bsoncxx::array::view K_arr {maybe_result->view()["K"].get_array().value};
    idx=0;
    for (const bsoncxx::array::element& el:K_arr){
        if(idx>=layers){
            break;
        }
        hyper->K[idx]=el.get_int32();
        idx++;
    }
    bsoncxx::array::view L_arr {maybe_result->view()["L"].get_array().value};
    idx=0;
    for (const bsoncxx::array::element& el:L_arr){
        if(idx>=layers){
            break;
        }
        hyper->L[idx]=el.get_int32();
        idx++;
    }
    bsoncxx::array::view node_types_arr {maybe_result->view()["node_types"].get_array().value};
    idx=0;
    for (const bsoncxx::array::element& el:node_types_arr){
        if(idx>=layers){
            break;
        }
        switch(el.get_int32()){
            case 0:
                hyper->node_types[idx]=NodeType::ReLU;
                break;
            case 1:
                hyper->node_types[idx]=NodeType::Softmax;
                break;
            case 2:
                hyper->node_types[idx]=NodeType::Sigmoid;
                break;
        }
        idx++;
    }
    bsoncxx::array::view sparcity_arr {maybe_result->view()["sparcity"].get_array().value};
    idx=0;
    for (const bsoncxx::array::element& el:sparcity_arr){
        if(idx>=layers){
            break;
        }
        if (el.type() == bsoncxx::type::k_int32){
            hyper->sparcity[idx]=(float)el.get_int32();
        }else if (el.type() == bsoncxx::type::k_double){
            hyper->sparcity[idx]=(float)el.get_double();
        }else{
            throw runtime_error("Error invalid type: "+bsoncxx::to_string(el.type())+"\n");
        }
        idx++;
    }
    return hyper;
}

void MongoDB::appendTMetricsOnNeuralNet(string id,vector<pair<float,float>> metrics){
    string metrics_str="[ ";
    for(size_t i=0;i<metrics.size();){
        if (metrics[i].second==-666){
            metrics_str+=to_string(metrics[i].first);
        }else{
            metrics_str+="{ "+to_string(metrics[i].first)+", "+to_string(metrics[i].second)+" }";
        }
        if (++i<metrics.size()){
            metrics_str+=", ";
        }
    }
    metrics_str+=" ]";
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "train_metrics" <<  metrics_str << close_document << finalize;
    getCollection(getDB("neural_db"),"independent_net").update_one(query.view(),update.view());
}

void MongoDB::appendStatsOnNeuralNet(string id,string field_name,snn_stats stats){
    bsoncxx::builder::basic::document stats_bson_builder{};
    stats_bson_builder.append(bsoncxx::builder::basic::kvp("accuracy",stats.accuracy));
    if (stats.precision!=-1)
        stats_bson_builder.append(bsoncxx::builder::basic::kvp("precision",stats.precision));
    if (stats.recall!=-1)
        stats_bson_builder.append(bsoncxx::builder::basic::kvp("recall",stats.recall));
    if (stats.f1!=-1)
        stats_bson_builder.append(bsoncxx::builder::basic::kvp("f1",stats.f1));
    bsoncxx::document::value stats_bson = stats_bson_builder.extract();
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << field_name <<  stats_bson << close_document << finalize;
    getCollection(getDB("neural_db"),"independent_net").update_one(query.view(),update.view());
}

void MongoDB::appendWeightsOnNeuralNet(string id,const map<string, vector<float>> weights){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;
    bsoncxx::document::value update=document{} << "$set" << open_document << "weights" <<  Utils::stringToBase65(Utils::serializeWeigthsToStr(weights)) << close_document << finalize;
    getCollection(getDB("neural_db"),"independent_net").update_one(query.view(),update.view());
}

vector<pair<vector<int>, vector<float>>> MongoDB::loadCveFromId(string cve){
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result = getCollection(getDB("processed_data"),"dataset").find_one(document{} << "cve" << cve << finalize);
    vector<pair<vector<int>, vector<float>>> cve_parsed;
    cve_parsed.push_back(bsonToDatasetEntry(maybe_result).second);
    return cve_parsed;
}

map<string, vector<float>> MongoDB::loadWeightsFromNeuralNet(string id){
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result=getCollection(getDB("neural_db"),"independent_net").find_one(document{} << "_id" << bsoncxx::oid{id} << finalize);
    string weights_str="";
    if(maybe_result) {
        bsoncxx::document::element weights_el = maybe_result->view()["weights"];
        weights_str=getStringFromEl(weights_el);
    }
    map<string, vector<float>> weights;
    if (weights_str!=""){
        weights=Utils::deserializeWeigthsFromStr(Utils::base64ToString(weights_str));
    }
    return weights;
}

void MongoDB::storeEvalNeuralNetResult(string id,int correct,vector<string> cve_ids,vector<vector<pair<int,float>>> pred_labels,vector<pair<vector<int>, vector<float>>> labels,snn_stats stats){
    bsoncxx::document::value query=document{} << "_id" << bsoncxx::oid{id} << finalize;

    bsoncxx::builder::basic::document stats_bson_builder{};
    stats_bson_builder.append(bsoncxx::builder::basic::kvp("accuracy",stats.accuracy));
    if (stats.precision!=-1)
        stats_bson_builder.append(bsoncxx::builder::basic::kvp("precision",stats.precision));
    if (stats.recall!=-1)
        stats_bson_builder.append(bsoncxx::builder::basic::kvp("recall",stats.recall));
    if (stats.f1!=-1)
        stats_bson_builder.append(bsoncxx::builder::basic::kvp("f1",stats.f1));
    bsoncxx::document::value stats_bson = stats_bson_builder.extract();

    bsoncxx::builder::basic::array res_array_builder = bsoncxx::builder::basic::array{};
    for (size_t i=0;i<cve_ids.size();i++){
        pair<int,float> p_label=pred_labels[i][0];
        int d_label=labels[i].first[0];
        float chance=p_label.second*100; // transform to percent
        string res="{ "+cve_ids[i]+": { predicted_exploit: "+to_string(p_label.first)+", label: "+to_string(d_label)+", chance_of_having: "+to_string(chance)+"% } }";
        res_array_builder.append(res);
    }

    bsoncxx::document::value update=document{} << "$set" << open_document << "result_stats" << stats_bson << "total_test_cases" << (int)pred_labels.size() << "matching_preds" << correct << "results" << res_array_builder << close_document << finalize;
    getCollection(getDB("neural_db"),"eval_results").update_one(query.view(),update.view());
}

string MongoDB::getStringFromEl(bsoncxx::document::element el){
    string out;
    if (el.type() == bsoncxx::type::k_utf8){
        out=el.get_utf8().value.to_string();
    }else{
        throw runtime_error("Error invalid type: "+bsoncxx::to_string(el.type())+"\n");
    }
    return out;
}

float MongoDB::getFloatFromEl(bsoncxx::document::element el){
    float out;
    if (el.type() == bsoncxx::type::k_int32){
       out=(float)el.get_int32();
    }else if (el.type() == bsoncxx::type::k_double){
        out=(float)el.get_double();
    }else{
        throw runtime_error("Error invalid type: "+bsoncxx::to_string(el.type())+"\n");
    }
    return out;
}

int MongoDB::getIntFromEl(bsoncxx::document::element el){
    int out;
    if (el.type() == bsoncxx::type::k_int32){
       out=el.get_int32();
    }else{
        throw runtime_error("Error invalid type: "+bsoncxx::to_string(el.type())+"\n");
    }
    return out;
}

vector<string> MongoDB::bsonToFeaturesName(bsoncxx::stdx::optional<bsoncxx::document::value> opt_bson){
    if(!opt_bson) {
        throw runtime_error("Empty opt result");
    }
    return bsonToFeaturesName(opt_bson->view());
}

vector<string> MongoDB::bsonToFeaturesName(const bsoncxx::v_noabi::document::view bson){
    vector<string> features;
    bsoncxx::document::element features_el = bson["features"];
    if (features_el.type() == bsoncxx::type::k_document){
        bsoncxx::document::view features_vw=features_el.get_document().view();
        for (auto el:features_vw){
            string key=el.key().data();
            features.push_back(key);
        }
    }else{
        throw runtime_error("Error unkown features type while getting feature names \n");
    }
    return features;
}


vector<pair<vector<int>, vector<float>>> MongoDB::filterFeatures(vector<pair<vector<int>, vector<float>>> in, vector<string> to_remove){
    vector<pair<vector<int>, vector<float>>> out;
    cout<<"Features before filtering: "<<in[0].second.size()<<endl;
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result=getCollection(getDB("processed_data"),"dataset").find_one(document{} << finalize);
    vector<string> features=bsonToFeaturesName(maybe_result);
    vector<int> features_to_ignore;
    for (int i=0;i<(int)features.size();i++){
        string feature=features[i];
        for (string pattern:to_remove){
            if (feature.find(pattern)!=string::npos) {
                features_to_ignore.push_back(i);
                break;
            }
        }
    }
    for(pair<vector<int>, vector<float>> entry:in){
        vector<float> filtered;
        for(int i=0;i<(int)entry.second.size();i++){
            if (find(features_to_ignore.begin(), features_to_ignore.end(), i)==features_to_ignore.end()){
                filtered.push_back(entry.second[i]);
            }
        }
        out.push_back(pair<vector<int>, vector<float>>(entry.first,filtered));
    }
    cout<<"Features after filtering: "<<out[0].second.size()<<endl;
    return out;
}
