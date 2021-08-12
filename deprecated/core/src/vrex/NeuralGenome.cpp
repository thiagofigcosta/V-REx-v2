#include "NeuralGenome.hpp"
#include "slide/Node.h"

bool NeuralGenome::CACHE_WEIGHTS=false;

const string NeuralGenome::CACHE_FOLDER="neural_genome_cache";
string NeuralGenome::last_print_str="";
vector<pair<vector<int>, vector<float>>> NeuralGenome::static_train_data=vector<pair<vector<int>, vector<float>>> ();

NeuralGenome::NeuralGenome(SPACE_SEARCH space, function<float(Genome *self)> callback)
    :Genome(space,callback){
    train_data=vector<pair<vector<int>, vector<float>>>(NeuralGenome::static_train_data);
    cache_file=genCacheFilename();
    cached=false;
}

NeuralGenome::NeuralGenome(const NeuralGenome& orig, pair<vector<int>,vector<float>> new_dna)
    :Genome(orig,new_dna){
    weights=orig.weights;
    train_data=orig.train_data;
    print_str=orig.print_str;
    cached=orig.cached;
    cache_file=genCacheFilename();
    if (cached&&NeuralGenome::CACHE_WEIGHTS){
        Utils::copyFile(orig.cache_file, cache_file);
    }
}

NeuralGenome::NeuralGenome(const NeuralGenome& orig){
    mt_dna=orig.mt_dna;
    dna=orig.dna;
    limits=orig.limits;
    fitness=orig.fitness;
    output=orig.output;
    evaluate_cb=orig.evaluate_cb;
    weights=orig.weights;
    id=orig.id;
    train_data=orig.train_data;
    print_str=orig.print_str;
    cache_file=genCacheFilename();
    cached=orig.cached;
    if (cached&&NeuralGenome::CACHE_WEIGHTS){
        Utils::copyFile(orig.cache_file, cache_file);
    }
}

NeuralGenome::~NeuralGenome(){
    weights.clear();
    train_data.clear();
    if (cached&&NeuralGenome::CACHE_WEIGHTS){
        Utils::rmFile(cache_file);
    }
}

SPACE_SEARCH NeuralGenome::buildSlideNeuralNetworkSpaceSearch(INT_SPACE_SEARCH amount_of_layers,INT_SPACE_SEARCH epochs,FLOAT_SPACE_SEARCH alpha,
                            INT_SPACE_SEARCH batch_size,INT_SPACE_SEARCH layer_size,INT_SPACE_SEARCH range_pow,INT_SPACE_SEARCH k_values,INT_SPACE_SEARCH l_values,
                            FLOAT_SPACE_SEARCH sparcity,INT_SPACE_SEARCH activation_funcs){
    vector<INT_SPACE_SEARCH> int_dna;
    vector<FLOAT_SPACE_SEARCH> float_dna;
    int_dna.push_back(epochs);
    int_dna.push_back(batch_size);
    int_dna.push_back(amount_of_layers);
    int max_layer_size=amount_of_layers.second;
    int_dna.push_back(INT_SPACE_SEARCH(max_layer_size,max_layer_size));
    for(int i=0;i<max_layer_size-1;i++){
        int_dna.push_back(layer_size);
    }
    for(int i=0;i<max_layer_size;i++){
        int_dna.push_back(range_pow);
    }
    for(int i=0;i<max_layer_size;i++){
        int_dna.push_back(k_values);
    }
    for(int i=0;i<max_layer_size;i++){
        int_dna.push_back(l_values);
    }
    for(int i=0;i<max_layer_size-1;i++){
        int_dna.push_back(activation_funcs);
    }
    float_dna.push_back(alpha);
    for(int i=0;i<max_layer_size-2;i++){
        float_dna.push_back(sparcity);
    }
    return SPACE_SEARCH(int_dna,float_dna);
}

tuple<Slide*,int,function<void()>> NeuralGenome::buildSlide(pair<vector<int>,vector<float>> dna, int input_size, int output_size, SlideLabelEncoding label_encoding, int rehash, int rebuild, int border_sparsity,SlideMetric metric,bool shuffleTrainData,SlideCrossValidation crossValidation, bool adam_optimizer){
    vector<int> int_dna=dna.first;
    vector<float> float_dna=dna.second;
    int epochs=int_dna[0];
    int batch_size=int_dna[1];
    int layers=int_dna[2];
    int max_layers=int_dna[3];

    int *layer_sizes=new int[layers];
    int *range_pow=new int[layers];
    int *K=new int[layers];
    int *L=new int[layers];
    NodeType* node_types = new NodeType[layers];
    float *sparcity=new float[layers];

    auto teardown_callback = [layer_sizes,range_pow,K,L,node_types,sparcity]() {
        delete[] range_pow;
        delete[] K;
        delete[] L;
        delete[] sparcity;
    };
    
    sparcity[0]=border_sparsity;
    sparcity[layers-1]=border_sparsity;
    layer_sizes[layers-1]=output_size;
    node_types[layers-1]=NodeType::Softmax;

    size_t maxNodes=(size_t)output_size;
    int l=4;
    int i;
    for(i=0;i<max_layers-1;l++,i++){
        if (i+1<layers){
            layer_sizes[i]=int_dna[l];
            if ((size_t)layer_sizes[i]>maxNodes){
                maxNodes=(size_t)layer_sizes[i];
            }
        }
    }
    for(i=0;i<max_layers;l++,i++){
        if (i<layers){
            range_pow[i]=int_dna[l];
        }
    }
    for(i=0;i<max_layers;l++,i++){
        if (i<layers){
            int dna_value=int_dna[l]; 
            if (dna_value>layer_sizes[i]){  // ??? K cannot be higher than layer size?
                dna_value=layer_sizes[i];
            }
            K[i]=dna_value;
        }
    }
    for(i=0;i<max_layers;l++,i++){
        if (i<layers){
            L[i]=int_dna[l];
        }
    }
    for(i=0;i<max_layers-1;l++,i++){
        if (i+1<layers){
            switch(int_dna[l]){
                case 0:
                    node_types[i]=NodeType::Softmax;
                case 1:
                    node_types[i]=NodeType::ReLU;
                case 2:
                    node_types[i]=NodeType::Sigmoid;
            }
        }
    }

    float alpha=float_dna[0];
    l=1;
    for(i=1;i<max_layers-1;l++,i++){
        if (i+1<layers){
            sparcity[i]=float_dna[l];
        }
    }

    print_str="Slide Network with:\n";
    print_str+="\tepochs: "+std::to_string(epochs)+"\n";
    print_str+="\tbatch_size: "+std::to_string(batch_size)+"\n";
    print_str+="\tlayers: "+std::to_string(layers)+"\n";
    print_str+="\talpha: "+std::to_string(alpha)+"\n";
    print_str+="\tmax_layers: "+std::to_string(max_layers)+"\n";

    for(i=0;i<layers;i++)
        print_str+="\tlayer_sizes["+std::to_string(i)+"]: "+std::to_string(layer_sizes[i])+"\n";
    for(i=0;i<layers;i++)
        print_str+="\trange_pow["+std::to_string(i)+"]: "+std::to_string(range_pow[i])+"\n";
    for(i=0;i<layers;i++)
        print_str+="\tK["+std::to_string(i)+"]: "+std::to_string(K[i])+"\n";
    for(i=0;i<layers;i++)
        print_str+="\tL["+std::to_string(i)+"]: "+std::to_string(L[i])+"\n";
    for(i=0;i<layers;i++)
        print_str+="\tnode_types["+std::to_string(i)+"]: "+std::to_string(static_cast<underlying_type<NodeType>::type>(node_types[i]))+"\n";
    for(i=0;i<layers;i++)
        print_str+="\tsparcity["+std::to_string(i)+"]: "+std::to_string(sparcity[i])+"\n";
    
    last_print_str=print_str;

    SlideMetric trainMetric=SlideMetric::RAW_LOSS;
    if (crossValidation==SlideCrossValidation::NONE){
        trainMetric=metric;
    }

    string ex_str="";
    Slide* net=nullptr;
    try{
        net=new Slide(layers, layer_sizes, node_types, input_size,output_size, alpha, batch_size, adam_optimizer, label_encoding, range_pow, K, L, sparcity, 
            rehash, rebuild,trainMetric,metric,shuffleTrainData,crossValidation, SlideMode::SAMPLING, SlideHashingFunction::DENSIFIED_WTA, false,maxNodes);
    } catch (const exception& ex) {
        ex_str=ex.what();
    } catch (const string& ex) {
        ex_str=""+ex;
    } catch (...) {
        exception_ptr p = current_exception();
        ex_str=(p ? p.__cxa_exception_type()->name() : "null");
    }

    if(!ex_str.empty()){
        throw runtime_error(ex_str+"\n\nOn "+print_str);
    }
    
    return {net,epochs,teardown_callback};
}

map<string, vector<float>> NeuralGenome::getWeights(){
    if (cached&&NeuralGenome::CACHE_WEIGHTS){
        weights=Utils::deserializeWeigths(cache_file,print_str);
    }
    return weights;
}

void NeuralGenome::clearWeightsIfCached(){
     if (cached&&NeuralGenome::CACHE_WEIGHTS){
        weights.clear();
    }
}

void NeuralGenome::forceCache(){
    if(cached&&NeuralGenome::CACHE_WEIGHTS){
        Utils::rmFile(cache_file);
        cached=false;
    }
    if (!cached&&NeuralGenome::CACHE_WEIGHTS){
        bool sucess=false;
        int tries=0;
        int max_tries=5;
        while (!sucess&&++tries<max_tries){
            try {
                Utils::serializeWeigths(weights, cache_file,print_str);
                sucess=true;
            }catch(const exception& e){
                cout<<e.what();
                if (tries==max_tries-1){
                    cache_file=genCacheFilename();
                }
            }
        }
        if (sucess){
            cached=true;
            weights.clear();
        }
    }
}

void NeuralGenome::setWeights(map<string, vector<float>> Weights){
    weights=Weights;
    forceCache();
}

void NeuralGenome::setNeuralTrainData(vector<pair<vector<int>, vector<float>>> data){
    NeuralGenome::static_train_data=data;
}

vector<pair<vector<int>, vector<float>>>& NeuralGenome::getTrainData(){
    return train_data;
}

string NeuralGenome::to_string(){
    string out=Genome::to_string()+"\n"+print_str;
    return out;
}

bool NeuralGenome::hasWeights(){
    return weights.size()>0 || (cached&&NeuralGenome::CACHE_WEIGHTS);
}

void NeuralGenome::clearWeights(){
    weights.clear();
    if (cached&&NeuralGenome::CACHE_WEIGHTS){
        Utils::rmFile(cache_file);
        cached=false;
    }
}

string NeuralGenome::getBaseFolder(){
    return Utils::getResourcePath(NeuralGenome::CACHE_FOLDER);
}

string NeuralGenome::genCacheFilename(){
    string base=getBaseFolder();
    string filename=boost::uuids::to_string(id)+Utils::genRandomUUIDStr()+".weights_cache";;
    Utils::mkdir(base);
    return Utils::joinPath(base,filename);
}