#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <memory>
#include <limits>

#define Slide_MAPLEN 2048 // was 325056
// Slide_HUGEPAGES defined in Node.h
#define Slide_USE_SMART_POINTERS 0 // 1 to use smart pointer or 0 to not. WARNING smart pointers are not thread safe 

class Bucket;
class Network;
enum class NodeType;
enum class SlideMode { TOPK_THRESHOLD=1, SAMPLING=4, UNKNOWN_MODE1=2, UNKNOWN_MODE2=3 }; // TODO find out mode names
enum class SlideHashingFunction { WTA=1, DENSIFIED_WTA=2, TOPK_MIN_HASH=3, SIMHASH=4 };
enum class SlideLabelEncoding { INT_CLASS, NEURON_BY_NEURON, NEURON_BY_N_LOG_LOSS };
enum class SlideMetric { RAW_LOSS, F1, RECALL, ACCURACY, PRECISION };
enum class SlideCrossValidation { NONE, ROLLING_FORECASTING_ORIGIN, KFOLDS, TWENTY_PERCENT };

using namespace std;

#if Slide_USE_SMART_POINTERS == 1
    typedef vector<vector<shared_ptr<Bucket>>> bucket_pointer_2d;
    typedef shared_ptr<int[]> int_array_pointer;
    typedef vector<shared_ptr<int[]>> int_array_pointer_2d;
#else
    typedef Bucket** bucket_pointer_2d;
    typedef int* int_array_pointer;
    typedef int** int_array_pointer_2d;
#endif

struct Hyperparameters{
    int batch_size;
    float alpha;
    bool shuffle;
    bool adam;
    int rehash;
    int rebuild;
    SlideLabelEncoding label_type;
    int layers;
    int *layer_sizes;
    int *range_pow;
    int *K;
    int *L;
    NodeType* node_types;
    float *sparcity;

    Hyperparameters(int amount_layers):layers(amount_layers){ 
        layer_sizes=new int[layers];
        range_pow=new int[layers];
        K=new int[layers];
        L=new int[layers];
        node_types=new NodeType[layers];
        sparcity=new float[layers];
    }
    ~Hyperparameters(){
        delete[] range_pow;
        delete[] K;
        delete[] L;
        delete[] sparcity; 
    }
    Hyperparameters* clone(){
        Hyperparameters* copy=new Hyperparameters(layers);
        copy->batch_size=batch_size;
        copy->alpha=alpha;
        copy->shuffle=shuffle;
        copy->adam=adam;
        copy->rehash=rehash;
        copy->rebuild=rebuild;
        copy->label_type=label_type;
        copy->layers=layers;
        for(int i=0;i<layers;i++){
            copy->layer_sizes[i]=layer_sizes[i];
            copy->range_pow[i]=range_pow[i];
            copy->K[i]=K[i];
            copy->L[i]=L[i];
            copy->node_types[i]=node_types[i];
            copy->sparcity[i]=sparcity[i];
        }
        return copy;
    }
};

class Slide{
    public:
        // constructors and destructor
        Slide(int numLayer, int *sizesOfLayers, NodeType* layerTypes, int InputDim, int OutputDim, float Lr, int Batchsize, bool useAdamOt, 
            SlideLabelEncoding labelType,int *RangePow, int *KValues,int *LValues,float *Sparsity, int Rehash, int Rebuild, 
            SlideMetric trainMetric,SlideMetric valMetric, bool shuffleTrainData, SlideCrossValidation crossValidation,
            SlideMode Mode=SlideMode::SAMPLING, SlideHashingFunction HashFunc=SlideHashingFunction::DENSIFIED_WTA, bool printDeltas=false, size_t maxLayerS = numeric_limits<size_t>::max());
        Slide(const Slide& orig);
        virtual ~Slide();

        // methods
        void setWeights(map<string, vector<float>> loadedData);
        map<string, vector<float>> getWeights();
        static NodeType* getStdLayerTypes(const int amount_layers);
        vector<float> trainNoValidation(vector<pair<vector<int>, vector<float>>> &train_data,int epochs);
        vector<pair<float,float>> train(vector<pair<vector<int>, vector<float>>> &train_data,int epochs);
        float evalLoss(vector<pair<vector<int>, vector<float>>> &eval_data);
        pair<int,vector<vector<pair<int,float>>>> evalData(vector<pair<vector<int>, vector<float>>> &test_data);
        void allocAndCastDatasetToSlide(vector<pair<vector<int>, vector<float>>> &data,float **&values, int *&sizes, int **&records, int **&labels, int *&labelsize);
        void deallocSlideDataset(float **values, int *sizes, int **records, int **labels, int *labelsize);
        void eagerInit();
        void flushNetwork();
        string toString();

        // variables
        static int MAX_THREADS; // 0 = max allowed 
        static constexpr float SINGLE_CLASS_THRESHOLD=.59;
        #pragma omp threadprivate(SINGLE_CLASS_THRESHOLD)
        static const bool MEAN_ERROR_INSTEAD_OF_GRADS_SUM=true;
        #pragma omp threadprivate(MEAN_ERROR_INSTEAD_OF_GRADS_SUM)
        static constexpr float ADAM_OT_BETA1=0.9;
        #pragma omp threadprivate(ADAM_OT_BETA1)
        static constexpr float ADAM_OT_BETA2=0.999;
        #pragma omp threadprivate(ADAM_OT_BETA2)
        static constexpr float ADAM_OT_EPSILON=0.00000001;
        #pragma omp threadprivate(ADAM_OT_EPSILON)
        static const int BUCKET_SIZE=32; // was 128
        #pragma omp threadprivate(BUCKET_SIZE)
        static const int TOPK_HASH_TOPK=30;
        #pragma omp threadprivate(TOPK_HASH_TOPK)
        static const int SIMHASH_RATIO=3;
        #pragma omp threadprivate(SIMHASH_RATIO)
        static const int WTA_BIN_SIZE=8; // binsize is the number of times the range is larger than the total number of hashes we need.
        #pragma omp threadprivate(WTA_BIN_SIZE)
        static const int TOPK_THRESHOLD_SECONDS=2;
        #pragma omp threadprivate(TOPK_THRESHOLD_SECONDS)
        static const bool FIFO_INSTEAD_OF_RESERVOIR_SAMPLING=true;
        #pragma omp threadprivate(FIFO_INSTEAD_OF_RESERVOIR_SAMPLING)
        static constexpr float SOFTMAX_LINEAR_CONSTANT=0.0000001;
        #pragma omp threadprivate(SOFTMAX_LINEAR_CONSTANT)
        static constexpr float RAND_WEIGHT_START=0;
        #pragma omp threadprivate(RAND_WEIGHT_START)
        static constexpr float RAND_WEIGHT_END=.33;
        #pragma omp threadprivate(RAND_WEIGHT_END)
        static const int K_FOLDS=10;
        #pragma omp threadprivate(K_FOLDS)
        static constexpr float ROLLING_FORECASTING_ORIGIN_MIN=.5;
        #pragma omp threadprivate(ROLLING_FORECASTING_ORIGIN_MIN)
        
    private:
        // methods
        pair<float,float> trainEpoch(vector<pair<vector<int>, vector<float>>> &train_data,vector<pair<vector<int>, vector<float>>> &validation_data, int cur_epoch);

        // variables
        Network *slide_network;
        NodeType* layer_types;
        int *range_pow;
        int *K;
        int *L;
        float *sparsity;
        int batch_size;
        int rehash;
        int rebuild;
        int input_dim;
        int output_dim;
        float learning_rate;
        int epochs;
        int *layer_sizes;
        int amount_layers;
        SlideMode mode;
        SlideHashingFunction hash_function;
        bool print_deltas;
        bool use_adam;
        SlideLabelEncoding label_type;
        SlideMetric train_metric;
        SlideMetric val_metric;
        bool shuffle_train_data;
        SlideCrossValidation cross_validation;
        size_t size_max_for_layer;

};
