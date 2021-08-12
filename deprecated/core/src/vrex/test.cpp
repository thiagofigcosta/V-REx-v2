#include "test.hpp"

void testCsvRead(){
    cout << "Hello Thiago " << endl;

    vector<pair<string, vector<float>>> dataset= Utils::readLabeledCsvDataset(Utils::getResourcePath("iris.data"));
    for (pair<string, vector<float>> row : dataset) {
        cout << "------------" <<endl;
        cout << row.first <<endl;
        for (float el : row.second ){
            cout << el<< " " ;
        }
        cout <<endl;
        cout << "------------" <<endl;
    }
}

void testMongo(){
    string mongo_host;
    if (Utils::runningOnDockerContainer()){
        mongo_host="mongo";
    }else{
        mongo_host="127.0.0.1";
    }
    MongoDB mongo = MongoDB(mongo_host,"root","123456");
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result =
        mongo.getCollection(mongo.getDB("processed_data"),"dataset").find_one(document{} << "cve" << "CVE-2017-0144" << finalize);
    if(maybe_result) {
        cout << "Eternal blue" <<endl << bsoncxx::to_json(*maybe_result) << endl;
    }

    mongocxx::cursor cursor = mongo.getCollection(mongo.getDB("processed_data"),"dataset").find({});
    int total=0;
    for(auto&& doc : cursor) {
        if (total++ == 0){
            string str_doc= bsoncxx::to_json(*maybe_result);
            bsoncxx::document::element cve = doc["cve"];
            cout << "first cve: "<< cve.get_utf8().value <<endl;
        }
    }
    cout << "Total documents " << total <<endl;

    auto builder = bsoncxx::builder::stream::document{};
    bsoncxx::document::value doc_value = builder
    << "network_id" << "asd65as6d5a"
    << "layer_sizes" << bsoncxx::builder::stream::open_array
        << "500" << "2" << "3"
    << close_array
    << "weights" << bsoncxx::builder::stream::open_document
        << "layer1" <<  bsoncxx::builder::stream::open_array
            << "566" << "3453" << "53453"
        << close_array
        << "layer2" << bsoncxx::builder::stream::open_array
            << "566" << "3453" << "53453"
        << close_array
    << bsoncxx::builder::stream::close_document
    << bsoncxx::builder::stream::finalize;
    bsoncxx::document::view view = doc_value.view();

    bsoncxx::stdx::optional<mongocxx::result::insert_one> result =
    mongo.getCollection(mongo.getDB("tests"),"cxx").insert_one(view);
}

void testSlide_IntLabel(){
    cout<<"Using int label"<<endl;
    bool print_data=false;
    pair<vector<pair<int, vector<float>>>,map<string,int>> enumfied = Utils::enumfyDataset(Utils::readLabeledCsvDataset(Utils::getResourcePath("iris.data")));
    vector<pair<vector<int>, vector<float>>> dataset = Utils::encodeDatasetLabels(enumfied.first,DataEncoder::INCREMENTAL);
    enumfied.first.clear(); // free
    if (print_data){
        cout << "Raw encoded data:"<<endl;
        for(pair<vector<int>, vector<float>> data:dataset){
            for (int el : data.first){
                cout<<el;
            }
            cout<<": ";
            for (float el : data.second){
                cout<<el<<",";
            }
            cout << endl;
        }
        cout<<"----------------------------"<<endl;
    }
    dataset=Utils::normalizeDataset(dataset).second;
    if (print_data){
        cout << "Raw encoded normalized data:"<<endl;
        for(pair<vector<int>, vector<float>> data:dataset){
            for (int el : data.first){
                cout<<el;
            }
            cout<<": ";
            for (float el : data.second){
                cout<<el<<",";
            }
            cout << endl;
        }
        cout<<"----------------------------"<<endl;
    }
    map<string,int> equivalence = enumfied.second;
    dataset=Utils::shuffleDataset(dataset);
    if (print_data){
        cout << "Raw encoded normalized randomized data:"<<endl;
        for(pair<vector<int>, vector<float>> data:dataset){
            for (int el : data.first){
                cout<<el;
            }
            cout<<": ";
            for (float el : data.second){
                cout<<el<<",";
            }
            cout << endl;
        }
        cout<<"----------------------------"<<endl;
    }

    float train_percentage=.7;
    pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> dividedData=
        Utils::divideDataSet(dataset, train_percentage);
    vector<pair<vector<int>, vector<float>>> train_data=dividedData.first;
    vector<pair<vector<int>, vector<float>>> test_data=dividedData.second;
    if (print_data){
        cout << "Raw encoded splitted randomized data:"<<endl;
        cout << "Train data:"<<endl;
        for(pair<vector<int>, vector<float>> data:train_data){
            for (int el : data.first){
                cout<<el;
            }
            cout<<": ";
            for (float el : data.second){
                cout<<el<<",";
            }
            cout << endl;
        }
        cout<<endl<< "Test data:"<<endl;
        for(pair<vector<int>, vector<float>> data:test_data){
            for (int el : data.first){
                cout<<el;
            }
            cout<<": ";
            for (float el : data.second){
                cout<<el<<",";
            }
            cout << endl;
        }
        cout<<"----------------------------"<<endl;
    }

    // Too heavy for my computer :/
    // int layers=2;
    // int *layer_sizes=new int[layers]{6,(int)train_data[0].first.size()};
    // int *range_pow=new int[layers]{6,18};
    // int *K=new int[layers]{2,6};
    // int *L=new int[layers]{20,50};
    // float *sparcity=new float[layers]{1,1};

    int layers=1;
    int *layer_sizes=new int[layers]{(int)train_data[0].first.size()};
    int *range_pow=new int[layers]{6};
    int *K=new int[layers]{2};
    int *L=new int[layers]{20};
    float *sparcity=new float[layers]{1};

    int epochs=5;
    float alpha=0.01;
    int batch_size=5;
    bool adam=true;
    bool shuffle_train_data=true;
    int rehash=6400;
    int rebuild=128000;
    bool print_deltas=true;
    SlideLabelEncoding label_type=SlideLabelEncoding::INT_CLASS;
    Slide* slide=new Slide(layers, layer_sizes, Slide::getStdLayerTypes(layers), train_data[0].second.size(), train_data[0].first.size(), alpha, batch_size, adam, label_type,
    range_pow, K, L, sparcity, rehash, rebuild,SlideMetric::RAW_LOSS,SlideMetric::RAW_LOSS,shuffle_train_data, SlideCrossValidation::NONE, SlideMode::SAMPLING, SlideHashingFunction::DENSIFIED_WTA, print_deltas);
    vector<float>train_losses=slide->trainNoValidation(train_data,epochs);
    for (float loss:train_losses){
        cout<<"Train loss: "<<loss<<endl;
    }
    float test_loss=slide->evalLoss(test_data);
    cout<<"Test loss: "<<test_loss<<endl;

    pair<int,vector<vector<pair<int,float>>>> predicted = slide->evalData(test_data);
    cout<<"Test size: "<<test_data.size()<<endl;
    cout<<"Correct values: "<<predicted.first<<endl;
    if (print_data){
        cout<<"Predicted values"<<endl;
        for (size_t i=0;i<predicted.second.size();i++){
            for (pair<int,float> el : predicted.second[i]){
                cout<<el.first;
            }
            cout<<"-> ";
            for (int el : test_data[i].first){
                cout<<el;
            }
            cout << endl;
        }
    }
    Utils::printStats(Utils::statisticalAnalysis(test_data, predicted.second));
    cout<<endl<<endl;
    delete slide;
    delete[] range_pow;
    delete[] K;
    delete[] L;
    delete[] sparcity;
}

void testSlide_NeuronByNeuronLabel(){
    cout<<endl<<"Using neuron label"<<endl;
    pair<vector<pair<int, vector<float>>>,map<string,int>> enumfied = Utils::enumfyDataset(Utils::readLabeledCsvDataset(Utils::getResourcePath("iris.data")));
    vector<pair<vector<int>, vector<float>>> dataset = Utils::encodeDatasetLabels(enumfied.first,DataEncoder::BINARY);
    // vector<pair<vector<int>, vector<float>>> dataset = Utils::encodeDatasetLabels(enumfied.first,DataEncoder::SPARSE);
    enumfied.first.clear(); // free
  
    dataset=Utils::shuffleDataset(dataset);
    dataset=Utils::normalizeDataset(dataset).second;

    float train_percentage=.7;
    pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> dividedData=
        Utils::divideDataSet(dataset, train_percentage);
    vector<pair<vector<int>, vector<float>>> train_data=dividedData.first;
    vector<pair<vector<int>, vector<float>>> test_data=dividedData.second;

    int layers=1;
    int epochs=150;
    float alpha=0.1;
    int batch_size=15;
    SlideLabelEncoding label_type=SlideLabelEncoding::NEURON_BY_NEURON;
    int *layer_sizes=new int[layers]{(int)train_data[0].first.size()};
    bool adam=true;
    bool shuffle_train_data=true;
    int *range_pow=new int[layers]{6};
    int *K=new int[layers]{2};
    int *L=new int[layers]{20};
    float *sparcity=new float[layers]{1};
    int rehash=6400;
    int rebuild=128000;
    bool print_deltas=true;
    Slide* slide=new Slide(layers, layer_sizes, Slide::getStdLayerTypes(layers), train_data[0].second.size(), train_data[0].first.size(), alpha, batch_size, adam, label_type,
    range_pow, K, L, sparcity, rehash, rebuild,SlideMetric::RAW_LOSS,SlideMetric::RAW_LOSS,shuffle_train_data, SlideCrossValidation::NONE, SlideMode::SAMPLING, SlideHashingFunction::DENSIFIED_WTA, print_deltas);
    vector<float>train_losses=slide->trainNoValidation(train_data,epochs);
    float total_loss=0;
    for (float loss:train_losses){
       total_loss+=loss;
    }
    cout<<"Avg Train loss: "<<total_loss/train_losses.size()<<endl;
    // for (float loss:train_losses){
    //     cout<<"Train loss: "<<loss<<endl;
    // }
    float test_loss=slide->evalLoss(test_data);
    cout<<"Test loss: "<<test_loss<<endl;

    pair<int,vector<vector<pair<int,float>>>> predicted = slide->evalData(test_data);
    cout<<"Test size: "<<test_data.size()<<endl;
    cout<<"Correct values: "<<predicted.first<<endl;
    Utils::printStats(Utils::statisticalAnalysis(test_data, predicted.second));
    cout<<endl<<endl;
    delete slide;
    delete[] range_pow;
    delete[] K;
    delete[] L;
    delete[] sparcity;
}

void testStdGeneticsOnMath(){
    cout<<"Minimization:"<<endl;
    FLOAT_SPACE_SEARCH x = FLOAT_SPACE_SEARCH(-512,512);
    vector<FLOAT_SPACE_SEARCH> limits;
    limits.push_back(x);
    limits.push_back(x);
    SPACE_SEARCH space=SPACE_SEARCH(vector<INT_SPACE_SEARCH>(),limits);
    int population_size=300;
    int max_gens=100;
    float mutation_rate=0.2;
    float sex_rate=0.6;
    bool search_maximum=false;
    int max_notables=5;
    HallOfFame elite_min=HallOfFame(max_notables, search_maximum);
    StandardGenetic ga = StandardGenetic(mutation_rate, sex_rate);
    PopulationManager population=PopulationManager(ga, space, [](Genome *self) -> float {
        // https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
        pair<vector<int>,vector<float>> dna=self->getDna();
        return -(dna.second[1]+47)*sin(sqrt(abs(dna.second[1]+(dna.second[0]/2)+47)))-dna.second[0]*sin(sqrt(abs(dna.second[0]-(dna.second[1]+47))));}
        ,population_size, search_maximum);
    population.setHallOfFame(elite_min);
    population.naturalSelection(max_gens);
    for (Genome* individual: elite_min.getNotables()){
        cout<<individual->to_string()<<endl;
    }
    pair<float,int> min_result=elite_min.getBest();
    cout<<"Expected: y(512,404.2319) = -959.6407"<<endl;
    cout<<"Min Best ("<<min_result.second<<"): "<<min_result.first<<endl<<endl;

    cout<<"Maximization:"<<endl;
    FLOAT_SPACE_SEARCH x2 = FLOAT_SPACE_SEARCH(-100,100);
    limits.clear();
    limits.push_back(x2);
    limits.push_back(x2);
    SPACE_SEARCH space2=SPACE_SEARCH(vector<INT_SPACE_SEARCH>(),limits);
    population_size=100;
    max_gens=100;
    mutation_rate=0.1;
    sex_rate=0.7;
    search_maximum=true;
    max_notables=5;
    HallOfFame elite_max=HallOfFame(max_notables, search_maximum);
    ga = StandardGenetic(mutation_rate, sex_rate);
    PopulationManager population2=PopulationManager(ga, space2, [](Genome *self) -> float {
        // https://www.sfu.ca/~ssurjano/easom.html TIME MINUS ONE // maximum -> x1=x2=pi -> y(x1,x2)=1
        pair<vector<int>,vector<float>> dna=self->getDna();
        return -(-cos(dna.second[0])*cos(dna.second[1])*exp(-(pow(dna.second[0]-M_PI,2)+pow(dna.second[1]-M_PI,2))));}
        ,population_size, search_maximum);
    population2.setHallOfFame(elite_max);
    population2.naturalSelection(max_gens);
    for (Genome* individual: elite_max.getNotables()){
        cout<<individual->to_string()<<endl;
    }
    pair<float,int> max_result=elite_max.getBest();
    cout<<"Expected: y(3.141592,3.141592) = 1"<<endl<<"Max Best ("<<max_result.second<<"): "<<max_result.first<<endl<<endl<<endl<<endl;
}

void testEnchancedGeneticsOnMath(){
    cout<<"Enchanced vs Standard:"<<endl;
    FLOAT_SPACE_SEARCH x = FLOAT_SPACE_SEARCH(-512,512);
    vector<FLOAT_SPACE_SEARCH> limits;
    limits.push_back(x);
    limits.push_back(x);
    SPACE_SEARCH space=SPACE_SEARCH(vector<INT_SPACE_SEARCH>(),limits);
    int population_start_size=300;
    int population_start_size_std=720;
    int max_gens=100;
    int max_age=10;
    int max_children=4;
    float mutation_rate=0.1;
    float recycle_rate=0.13;
    float sex_rate=0.7;
    bool search_maximum=false;
    int max_notables=5;
    bool print_deltas=false;

    int tests=50;
    vector<pair<pair<float,int>,pair<float,int>>> results;

    for (int i=0;i<tests;i++){
        HallOfFame enchanced_elite=HallOfFame(max_notables, search_maximum);
        EnchancedGenetic en_ga = EnchancedGenetic(max_children,max_age,mutation_rate,sex_rate,recycle_rate);
        PopulationManager enchanced_population=PopulationManager(en_ga, space, [](Genome *self) -> float {
            // https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
            pair<vector<int>,vector<float>> dna=self->getDna();
            return -(dna.second[1]+47)*sin(sqrt(abs(dna.second[1]+(dna.second[0]/2)+47)))-dna.second[0]*sin(sqrt(abs(dna.second[0]-(dna.second[1]+47))));}
            ,population_start_size, search_maximum,false,print_deltas);
        enchanced_population.setHallOfFame(enchanced_elite);
        // cout<<"Enchanced: ";
        enchanced_population.naturalSelection(max_gens);
        pair<float,int> enchanced_result=enchanced_elite.getBest();

        HallOfFame std_elite=HallOfFame(max_notables, search_maximum);
        StandardGenetic std_ga = StandardGenetic(mutation_rate, sex_rate);
        PopulationManager std_population=PopulationManager(std_ga, space, [](Genome *self) -> float {
            // https://www.sfu.ca/~ssurjano/egg.html // minimum -> x1=512 | x2=404.2319 -> y(x1,x2)=-959.6407
            pair<vector<int>,vector<float>> dna=self->getDna();
            return -(dna.second[1]+47)*sin(sqrt(abs(dna.second[1]+(dna.second[0]/2)+47)))-dna.second[0]*sin(sqrt(abs(dna.second[0]-(dna.second[1]+47))));}
            ,population_start_size_std, search_maximum,false,print_deltas);
        std_population.setHallOfFame(std_elite);
        // cout<<"Standard : ";
        std_population.naturalSelection(max_gens);
        pair<float,int> std_result=std_elite.getBest();
        results.push_back(pair<pair<float,int>,pair<float,int>>(enchanced_result,std_result));
    }
    pair<float,float> enchanced_mean=pair<float,float>(0,0);
    pair<float,float> std_mean=pair<float,float>(0,0);
    for (pair<pair<float,int>,pair<float,int>> result:results){
        enchanced_mean.first+=result.first.first;
        enchanced_mean.second+=result.first.second;
        std_mean.first+=result.second.first;
        std_mean.second+=result.second.second;
        cout<<"Enchanced Best ("<<result.first.second<<"): "<<result.first.first<<" | Std Best ("<<result.second.second<<"): "<<result.second.first<<endl;
    }
    enchanced_mean.first/=tests;
    enchanced_mean.second/=tests;
    std_mean.first/=tests;
    std_mean.second/=tests;
    cout<<endl<<"Enchanced Mean ("<<enchanced_mean.second<<"): "<<enchanced_mean.first<<" | Std Mean ("<<std_mean.second<<"): "<<std_mean.first<<endl<<endl<<endl;
}

void testSlide_Validation(){
    cout<<"Testing cross validation"<<endl;
    pair<vector<pair<int, vector<float>>>,map<string,int>> enumfied = Utils::enumfyDataset(Utils::readLabeledCsvDataset(Utils::getResourcePath("iris.data")));
    vector<pair<vector<int>, vector<float>>> dataset = Utils::encodeDatasetLabels(enumfied.first,DataEncoder::INCREMENTAL);
    enumfied.first.clear(); // free
    dataset=Utils::normalizeDataset(dataset).second;
    map<string,int> equivalence = enumfied.second;
    dataset=Utils::shuffleDataset(dataset);


    float train_percentage=.7;
    pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> dividedData=
        Utils::divideDataSet(dataset, train_percentage);
    vector<pair<vector<int>, vector<float>>> train_data=dividedData.first;
    vector<pair<vector<int>, vector<float>>> test_data=dividedData.second;
   
    int layers=1;
    int *layer_sizes=new int[layers]{(int)train_data[0].first.size()};
    int *range_pow=new int[layers]{6};
    int *K=new int[layers]{2};
    int *L=new int[layers]{20};
    float *sparcity=new float[layers]{1};

    SlideCrossValidation cv=SlideCrossValidation::KFOLDS;

    int epochs=3;
    float alpha=0.01;
    int batch_size=5;
    bool adam=true;
    bool shuffle_train_data=true;
    int rehash=6400;
    int rebuild=128000;
    bool print_deltas=true;
    SlideLabelEncoding label_type=SlideLabelEncoding::INT_CLASS;
    Slide* slide=new Slide(layers, layer_sizes, Slide::getStdLayerTypes(layers), train_data[0].second.size(), train_data[0].first.size(), alpha, batch_size, adam, label_type,
    range_pow, K, L, sparcity, rehash, rebuild,SlideMetric::ACCURACY,SlideMetric::ACCURACY,shuffle_train_data, cv, SlideMode::SAMPLING, SlideHashingFunction::DENSIFIED_WTA, print_deltas);
    vector<pair<float,float>> metrics=slide->train(train_data,epochs);
    for (pair<float,float> me:metrics){
        cout<<"Train Acc: "<<me.first<<" - Val Acc: "<<me.second<<endl;
    }
    float test_loss=slide->evalLoss(test_data);
    cout<<"Test loss: "<<test_loss<<endl;

    pair<int,vector<vector<pair<int,float>>>> predicted = slide->evalData(test_data);
    cout<<"Test size: "<<test_data.size()<<endl;
    cout<<"Correct values: "<<predicted.first<<endl;
    Utils::printStats(Utils::statisticalAnalysis(test_data, predicted.second));
    cout<<endl<<endl;
    delete slide;
    delete[] range_pow;
    delete[] K;
    delete[] L;
    delete[] sparcity;
}

void testGeneticallyTunedNeuralNetwork(){
    INT_SPACE_SEARCH amount_of_layers = INT_SPACE_SEARCH(1,1); // For weaker computers
    // INT_SPACE_SEARCH amount_of_layers = INT_SPACE_SEARCH(1,2); // Too heavy for my computer :(
    // INT_SPACE_SEARCH amount_of_layers = INT_SPACE_SEARCH(1,3); // Too heavy for my computer :(
    // INT_SPACE_SEARCH amount_of_layers = INT_SPACE_SEARCH(1,4); // Too heavy for my computer :(
        
    INT_SPACE_SEARCH epochs = INT_SPACE_SEARCH(20,40); // Per generation, so it is not a good idea to use large numbers such [100,250]
    FLOAT_SPACE_SEARCH alpha = FLOAT_SPACE_SEARCH(0.0001,0.1);
    INT_SPACE_SEARCH batch_size = INT_SPACE_SEARCH(5,15);
    INT_SPACE_SEARCH layer_size = INT_SPACE_SEARCH(3,10);
    INT_SPACE_SEARCH range_pow = INT_SPACE_SEARCH(4,20);
    INT_SPACE_SEARCH k_values = INT_SPACE_SEARCH(2,10);
    INT_SPACE_SEARCH l_values = INT_SPACE_SEARCH(20,50);
    FLOAT_SPACE_SEARCH sparcity = FLOAT_SPACE_SEARCH(0.005,1);
    INT_SPACE_SEARCH activation_funcs = INT_SPACE_SEARCH(0,2);

    const float train_percentage=.7;
    vector<pair<vector<int>, vector<float>>> dataset = Utils::encodeDatasetLabels(Utils::enumfyDataset(Utils::readLabeledCsvDataset(Utils::getResourcePath("iris.data"))).first,DataEncoder::INCREMENTAL);  
    dataset=Utils::normalizeDataset(Utils::shuffleDataset(dataset)).second;
    pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> dividedData=Utils::divideDataSet(dataset, train_percentage);
    vector<pair<vector<int>, vector<float>>> train_data=dividedData.first;
    vector<pair<vector<int>, vector<float>>> test_data=dividedData.second;
    // NeuralGenome::setNeuralTrainData(train_data); // not necessary since we are using lambda [&]


    NeuralGenome::CACHE_WEIGHTS=true;


    SPACE_SEARCH space = NeuralGenome::buildSlideNeuralNetworkSpaceSearch(amount_of_layers,epochs,alpha,batch_size,
                                                layer_size,range_pow,k_values,l_values,sparcity,activation_funcs);

    int population_start_size=30; // change to 1 for fast run
    int max_gens=10; // change to 2 for fast run
    int max_age=10; // change to 2 for fast run
    int max_children=4; // change to 1 for fast run

    // int population_start_size=2; 
    // int max_gens=1;
    // int max_age=1; 
    // int max_children=1; 

    float mutation_rate=0.1;
    float recycle_rate=0.13;
    float sex_rate=0.7;
    int max_notables=3;

    SlideCrossValidation cross_validation=SlideCrossValidation::KFOLDS;
    SlideMetric metric_mode=SlideMetric::RAW_LOSS; // SlideMetric::ACCURACY or SlideMetric::RECALL.. is heavier than SlideMetric::RAW_LOSS

    const int input_size=4;
    const int output_size=1;
    const bool adam_optimizer=true;
    const bool use_neural_genome=true;
    const int rehash=6400;
    const int rebuild=128000;
    const int border_sparsity=1; // first and last layers
    const bool shuffle_train_data=true;
    const bool search_maximum=(metric_mode!=SlideMetric::RAW_LOSS);
    const SlideLabelEncoding label_encoding=SlideLabelEncoding::INT_CLASS;

    auto train_callback = [&](Genome *self) -> float {
        auto self_neural=dynamic_cast<NeuralGenome*>(self);
        if (!self_neural) {
            throw runtime_error("Error could not find an instance of NeuralGenome!\nhint: make useNeuralGenome=true on PopulationManager construtor!");
        }
        tuple<Slide*,int,function<void()>> net=self_neural->buildSlide(self->getDna(),input_size,output_size,label_encoding,rehash,rebuild,border_sparsity,metric_mode,shuffle_train_data,cross_validation,adam_optimizer);
        if (self_neural->hasWeights()){
            get<0>(net)->setWeights(self_neural->getWeights());
            self_neural->clearWeights();
        }

        vector<pair<float,float>> metric=get<0>(net)->train(train_data,get<1>(net));
        // vector<float> loss=get<0>(net)->train(self_neural->getTrainData(),get<1>(net)); // not necessary since we are using lambda [&] 
        self_neural->setWeights(get<0>(net)->getWeights());
        delete get<0>(net); // free memory
        get<2>(net)(); // free memory
        float output=0;
        for(pair<float,float> l:metric){
            output+=l.second; // use validation
        }
        output/=metric.size();
        return output;
    };
    
    HallOfFame elite=HallOfFame(max_notables, search_maximum);
    EnchancedGenetic en_ga = EnchancedGenetic(max_children,max_age,mutation_rate,sex_rate,recycle_rate);
    // StandardGenetic en_ga = StandardGenetic(mutation_rate,sex_rate);
    PopulationManager enchanced_population=PopulationManager(en_ga,space,train_callback,population_start_size,search_maximum,use_neural_genome,true);
    enchanced_population.setHallOfFame(elite);
    cout<<"Starting natural selection"<<endl;
    enchanced_population.naturalSelection(max_gens);
    cout<<"Finished natural selection"<<endl;
    cout<<"Best loss ("<<elite.getBest().second<<"): "<<elite.getBest().first<<endl;

    auto test_callback = [&](Genome *self) {
        auto self_neural=dynamic_cast<NeuralGenome*>(self);
        if (!self_neural) {
            throw runtime_error("Error could not find an instance of NeuralGenome!\nhint: make useNeuralGenome=true on PopulationManager construtor!");
        }
        tuple<Slide*,int,function<void()>> net=self_neural->buildSlide(self->getDna(),input_size,output_size,label_encoding,rehash,rebuild,border_sparsity,metric_mode,shuffle_train_data,cross_validation,adam_optimizer);
        if (self_neural->hasWeights()){
            get<0>(net)->setWeights(self_neural->getWeights());
        }

        pair<int,vector<vector<pair<int,float>>>> predicted = get<0>(net)->evalData(test_data);
        delete get<0>(net); // free memory
        get<2>(net)(); // free memory
        cout<<"Test size: "<<predicted.second.size()<<endl;
        cout<<"Correct values: "<<predicted.first<<endl;
        Utils::printStats(Utils::statisticalAnalysis(test_data, predicted.second));
    };

    test_callback(elite.getNotables()[0]); // test best of all times
    for (Genome* individual: elite.getNotables()){
        cout<<dynamic_cast<NeuralGenome*>(individual)->to_string()<<endl;
    }
}

void testMongoCveRead(){
    string mongo_host;
    if (Utils::runningOnDockerContainer()){
        mongo_host="mongo";
    }else{
        mongo_host="127.0.0.1";
    }
    MongoDB mongo = MongoDB(mongo_host,"root","123456");
    bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result =
        mongo.getCollection(mongo.getDB("processed_data"),"dataset").find_one(document{} << "cve" << "CVE-2017-0144" << finalize);
    if(maybe_result) {
        int features_size=863;
        bsoncxx::document::element index = maybe_result->view()["index"];
        bsoncxx::document::element cve = maybe_result->view()["cve"];
        bsoncxx::document::element features = maybe_result->view()["features"];
        bsoncxx::document::element labels = maybe_result->view()["labels"];
        cout<<"index type: "<<bsoncxx::to_string(index.type())<<endl;
        if (index.type() == bsoncxx::type::k_int32){
            int idx = index.get_int32();
            cout<<"index value: "<<idx<<endl;
        }
        cout<<"cve type: "<<bsoncxx::to_string(cve.type())<<endl;
        if (cve.type() == bsoncxx::type::k_utf8){
            string cve_str = cve.get_utf8().value.data();
            cout<<"cve value: "<<cve_str<<endl;
        }
        cout<<"features type: "<<bsoncxx::to_string(features.type())<<endl;
        if (features.type() == bsoncxx::type::k_document){
            bsoncxx::document::view features_vw=features.get_document().view();
            map<string,float> features;
            for (auto el:features_vw){
                string key=el.key().data();
                if (el.type() == bsoncxx::type::k_int32){
                    features[key]=(float)el.get_int32();
                }else if (el.type() == bsoncxx::type::k_double){
                    features[key]=(float)el.get_double();
                }else{
                    throw runtime_error("Error unkown type: "+bsoncxx::to_string(el.type())+"\n");
                }
            }
            cout<<"Features value (size: "+to_string(features.size())+"): "<<endl;
            if((int)features.size()!=features_size){
                throw runtime_error("Error features sizes should be: "+to_string(features_size)+" but is: "+to_string(features.size())+"\n");
            }
            for(map<string,float>::const_iterator it=features.begin();it!=features.end();it++){
                cout<<"\t"<<it->first<<": "<<it->second<<endl;
            }
        }
        cout<<"labels type: "<<bsoncxx::to_string(labels.type())<<endl;
        if (labels.type() == bsoncxx::type::k_document){
            bsoncxx::document::view labels_vw=labels.get_document().view();
            map<string,float> labels;
            for (auto el:labels_vw){
                string key=el.key().data();
                if (el.type() == bsoncxx::type::k_int32){
                    labels[key]=(float)el.get_int32();
                }else if (el.type() == bsoncxx::type::k_double){
                    labels[key]=(float)el.get_double();
                }else{
                    throw runtime_error("Error unkown type: "+bsoncxx::to_string(el.type())+"\n");
                }
            }
            cout<<"Labels value (size: "+to_string(labels.size())+"): "<<endl;
            for(map<string,float>::const_iterator it=labels.begin();it!=labels.end();it++){
                cout<<"\t"<<it->first<<": "<<it->second<<endl;
            }
        }
    }
    int year=2017;
    int limit=10;
    bsoncxx::document::value query = document{}
        << "cve"<< open_document
            << "$regex" <<  "CVE-"+to_string(year)+"-.*"
        << close_document
    << finalize;
    mongocxx::options::find opts{};
    opts.sort(document{}<<"cve"<<1<<finalize);
    mongocxx::cursor cursor = mongo.getCollection(mongo.getDB("processed_data"),"dataset").find(query.view(),opts);
    int total=0;
    for(auto&& doc : cursor) {
        cout << "cve: "<< doc["cve"].get_utf8().value <<endl;
        if (++total>=limit){
            break;
        }
    }
    cout<<"Total "<<year<<" documents: "<<total<<endl;
}

void testSmartNeuralNetwork_cveData(){
    cout<<"Testing smart neural network on CVE data"<<endl;
    string mongo_host;
    if (Utils::runningOnDockerContainer()){
        mongo_host="mongo";
    }else{
        mongo_host="127.0.0.1";
    }
    MongoDB mongo = MongoDB(mongo_host,"root","123456");
    vector<pair<vector<int>, vector<float>>> dataset=mongo.loadCvesFromYear(2020).second;
    float train_percentage=.7;
    pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> dividedData=
        Utils::divideDataSet(dataset, train_percentage);
    vector<pair<vector<int>, vector<float>>> train_data=dividedData.first;
    vector<pair<vector<int>, vector<float>>> test_data=dividedData.second;


    int layers=1;
    int *layer_sizes=new int[layers]{(int)train_data[0].first.size()};
    int *range_pow=new int[layers]{6};
    int *K=new int[layers]{2};
    int *L=new int[layers]{20};
    float *sparcity=new float[layers]{1};

    int epochs=5;
    float alpha=0.01;
    int batch_size=5;
    bool adam=true;
    bool shuffle_train_data=true;
    int rehash=6400;
    int rebuild=128000;
    bool print_deltas=true;
    SlideLabelEncoding label_type=SlideLabelEncoding::INT_CLASS;


    Slide* slide=new Slide(layers, layer_sizes, Slide::getStdLayerTypes(layers), train_data[0].second.size(), train_data[0].first.size(), alpha, batch_size, adam, label_type,
    range_pow, K, L, sparcity, rehash, rebuild,SlideMetric::RAW_LOSS,SlideMetric::RAW_LOSS,shuffle_train_data, SlideCrossValidation::NONE, SlideMode::SAMPLING, SlideHashingFunction::DENSIFIED_WTA, print_deltas);
    vector<float>train_losses=slide->trainNoValidation(train_data,epochs);
    for (float loss:train_losses){
        cout<<"Train loss: "<<loss<<endl;
    }
    float test_loss=slide->evalLoss(test_data);
    cout<<"Test loss: "<<test_loss<<endl;

    pair<int,vector<vector<pair<int,float>>>> predicted = slide->evalData(test_data);
    cout<<"Test size: "<<test_data.size()<<endl;
    cout<<"Correct values: "<<predicted.first<<endl;

    Utils::printStats(Utils::statisticalAnalysis(test_data, predicted.second));
    cout<<endl<<endl;
    delete slide;
    delete[] range_pow;
    delete[] K;
    delete[] L;
    delete[] sparcity;
}

void testGeneticallyTunedSmartNeuralNetwork_cveData(){
    cout<<"Testing genetically tuned smart neural network on CVE data"<<endl;
    string mongo_host;
    if (Utils::runningOnDockerContainer()){
        mongo_host="mongo";
    }else{
        mongo_host="127.0.0.1";
    }
    MongoDB mongo = MongoDB(mongo_host,"root","123456");
    vector<pair<vector<int>, vector<float>>> dataset=mongo.loadCvesFromYear(2020).second;
    float train_percentage=.7;
    pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> dividedData=
        Utils::divideDataSet(dataset, train_percentage);
    vector<pair<vector<int>, vector<float>>> train_data=dividedData.first;
    vector<pair<vector<int>, vector<float>>> test_data=dividedData.second;

    INT_SPACE_SEARCH amount_of_layers = INT_SPACE_SEARCH(1,1); // For weaker computers
    // INT_SPACE_SEARCH amount_of_layers = INT_SPACE_SEARCH(1,2); // Too heavy for my computer :(
        
    INT_SPACE_SEARCH epochs = INT_SPACE_SEARCH(20,40); // Per generation, so it is not a good idea to use large numbers such [100,250]
    FLOAT_SPACE_SEARCH alpha = FLOAT_SPACE_SEARCH(0.0001,0.1);
    INT_SPACE_SEARCH batch_size = INT_SPACE_SEARCH(5,15);
    INT_SPACE_SEARCH layer_size = INT_SPACE_SEARCH(5,40);
    INT_SPACE_SEARCH range_pow = INT_SPACE_SEARCH(4,20);
    INT_SPACE_SEARCH k_values = INT_SPACE_SEARCH(2,10);
    INT_SPACE_SEARCH l_values = INT_SPACE_SEARCH(20,50);
    FLOAT_SPACE_SEARCH sparcity = FLOAT_SPACE_SEARCH(0.005,1);
    INT_SPACE_SEARCH activation_funcs = INT_SPACE_SEARCH(0,2);

    NeuralGenome::CACHE_WEIGHTS=true;
    SPACE_SEARCH space = NeuralGenome::buildSlideNeuralNetworkSpaceSearch(amount_of_layers,epochs,alpha,batch_size,
                                                layer_size,range_pow,k_values,l_values,sparcity,activation_funcs);

    // int population_start_size=30; // change to 1 for fast run
    // int max_gens=10; // change to 2 for fast run
    // int max_age=10; // change to 2 for fast run
    // int max_children=4; // change to 1 for fast run

    int population_start_size=2; 
    int max_gens=1;
    int max_age=1; 
    int max_children=1; 

    float mutation_rate=0.1;
    float recycle_rate=0.13;
    float sex_rate=0.7;
    int max_notables=3;

    SlideCrossValidation cross_validation=SlideCrossValidation::ROLLING_FORECASTING_ORIGIN;
    SlideMetric metric_mode=SlideMetric::RAW_LOSS; // SlideMetric::ACCURACY or SlideMetric::RECALL.. is heavier than SlideMetric::RAW_LOSS

    const int input_size=train_data[0].second.size();
    const int output_size=train_data[0].first.size();
    const bool adam_optimizer=true;
    const bool use_neural_genome=true;
    const int rehash=6400;
    const int rebuild=128000;
    const int border_sparsity=1; // first and last layers
    const bool shuffle_train_data=true;
    const bool search_maximum=(metric_mode!=SlideMetric::RAW_LOSS);
    const SlideLabelEncoding label_encoding=SlideLabelEncoding::INT_CLASS;

    auto train_callback = [&](Genome *self) -> float {
        auto self_neural=dynamic_cast<NeuralGenome*>(self);
        if (!self_neural) {
            throw runtime_error("Error could not find an instance of NeuralGenome!\nhint: make useNeuralGenome=true on PopulationManager construtor!");
        }
        tuple<Slide*,int,function<void()>> net=self_neural->buildSlide(self->getDna(),input_size,output_size,label_encoding,rehash,rebuild,border_sparsity,metric_mode,shuffle_train_data,cross_validation,adam_optimizer);
        if (self_neural->hasWeights()){
            get<0>(net)->setWeights(self_neural->getWeights());
            self_neural->clearWeights();
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
    
    HallOfFame elite=HallOfFame(max_notables, search_maximum);
    EnchancedGenetic en_ga = EnchancedGenetic(max_children,max_age,mutation_rate,sex_rate,recycle_rate);
    // StandardGenetic en_ga = StandardGenetic(mutation_rate,sex_rate);
    PopulationManager enchanced_population=PopulationManager(en_ga,space,train_callback,population_start_size,search_maximum,use_neural_genome,true);
    enchanced_population.setHallOfFame(elite);
    cout<<"Starting natural selection"<<endl;
    enchanced_population.naturalSelection(max_gens);
    cout<<"Finished natural selection"<<endl;
    cout<<"Best loss ("<<elite.getBest().second<<"): "<<elite.getBest().first<<endl;

    auto test_callback = [&](Genome *self) {
        auto self_neural=dynamic_cast<NeuralGenome*>(self);
        if (!self_neural) {
            throw runtime_error("Error could not find an instance of NeuralGenome!\nhint: make useNeuralGenome=true on PopulationManager construtor!");
        }
        tuple<Slide*,int,function<void()>> net=self_neural->buildSlide(self->getDna(),input_size,output_size,label_encoding,rehash,rebuild,border_sparsity,metric_mode,shuffle_train_data,cross_validation,adam_optimizer);
        if (self_neural->hasWeights()){
            get<0>(net)->setWeights(self_neural->getWeights());
        }

        pair<int,vector<vector<pair<int,float>>>> predicted = get<0>(net)->evalData(test_data);
        delete get<0>(net); // free memory
        get<2>(net)(); // free memory
        cout<<"Test size: "<<predicted.second.size()<<endl;
        cout<<"Correct values: "<<predicted.first<<endl;
        Utils::printStats(Utils::statisticalAnalysis(test_data, predicted.second));
    };

    test_callback(elite.getNotables()[0]); // test best of all times
    for (Genome* individual: elite.getNotables()){
        cout<<dynamic_cast<NeuralGenome*>(individual)->to_string()<<endl;
    }
    cout<<endl<<endl;
}

void test(int func_id) {
    switch(func_id){
        default:
            cout<<"Next time use one of the following (instead of 0):\n1 - testCsvRead\n\t2 - testMongo\n\t3 - testSlide_IntLabel\n\t4 - testSlide_NeuronByNeuronLabel\n\t5 - testStdGeneticsOnMath\n\t6 - testEnchancedGeneticsOnMath\n\t7 - testSlide_Validation\n\t8 - testGeneticallyTunedNeuralNetwork\n\t9 - testMongoCveRead\n\t10 - testSmartNeuralNetwork_cveData\n\t11 - testGeneticallyTunedSmartNeuralNetwork_cveData\n";
            break;
        case 1:
            testCsvRead();
            break;
        case 2:
            testMongo();
            break;
        case 3:
            testSlide_IntLabel();
            break;
        case 4:
            testSlide_NeuronByNeuronLabel();
            break;
        case 5:
            testStdGeneticsOnMath();
            break;
        case 6:
            testEnchancedGeneticsOnMath();
            break;
        case 7:
            testSlide_Validation();
            break;
        case 8:
            testGeneticallyTunedNeuralNetwork();
            break;
        case 9:
            testMongoCveRead();
            break;
        case 10:
            testSmartNeuralNetwork_cveData();
            break;
        case 11:
            testGeneticallyTunedSmartNeuralNetwork_cveData();
            break;
    }
}
