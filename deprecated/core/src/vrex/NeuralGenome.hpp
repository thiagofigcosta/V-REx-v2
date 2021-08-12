#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <functional>
#include <tuple>
#include <exception>
#include <typeinfo>
#include <stdexcept>

#include "Genome.hpp"
#include "PopulationManager.hpp"
#include "Utils.hpp"
#include "Slide.hpp"

using namespace std;

class NeuralGenome : public Genome{
    public:
        // constructors and destructor
        NeuralGenome(SPACE_SEARCH space, function<float(Genome *self)> callback);
        NeuralGenome(const NeuralGenome& orig, pair<vector<int>,vector<float>> new_dna);
        NeuralGenome(const NeuralGenome& orig);
        virtual ~NeuralGenome();

        // methods
        void clearWeights();
        bool hasWeights();
        map<string, vector<float>> getWeights();
        void setWeights(map<string, vector<float>> Weights);
        vector<pair<vector<int>, vector<float>>>& getTrainData();
        static SPACE_SEARCH buildSlideNeuralNetworkSpaceSearch(INT_SPACE_SEARCH amount_of_layers,INT_SPACE_SEARCH epochs,FLOAT_SPACE_SEARCH alpha,
                            INT_SPACE_SEARCH batch_size,INT_SPACE_SEARCH layer_size,INT_SPACE_SEARCH range_pow,INT_SPACE_SEARCH k_values,INT_SPACE_SEARCH l_values,
                            FLOAT_SPACE_SEARCH sparcity,INT_SPACE_SEARCH activation_funcs);
        tuple<Slide*,int,function<void()>> buildSlide(pair<vector<int>,vector<float>> dna, int input_size, int output_size, SlideLabelEncoding label_encoding, int rehash, int rebuild, int border_sparsity,SlideMetric metric,bool shuffleTrainData,SlideCrossValidation crossValidation,  bool adam_optimizer=true);
        static void setNeuralTrainData(vector<pair<vector<int>, vector<float>>> data);
        string to_string();
        static string getBaseFolder();
        void forceCache();
        void clearWeightsIfCached();
        
        // variables
        static string last_print_str;
        static bool CACHE_WEIGHTS;

    private:
        // methods
        string genCacheFilename();
        // variables
        static const string CACHE_FOLDER;
        string cache_file;
        bool cached;
        map<string, vector<float>> weights;
        static vector<pair<vector<int>, vector<float>>> static_train_data;
        vector<pair<vector<int>, vector<float>>> train_data;
        string print_str;
};