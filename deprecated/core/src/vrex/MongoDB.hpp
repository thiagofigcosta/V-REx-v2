#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <map>
#include <utility> // std::pair
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/stdx.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/instance.hpp>
#include <bsoncxx/builder/stream/helpers.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/builder/stream/array.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>

using bsoncxx::builder::stream::close_array;
using bsoncxx::builder::stream::close_document;
using bsoncxx::builder::stream::document;
using bsoncxx::builder::stream::finalize;
using bsoncxx::builder::stream::open_array;
using bsoncxx::builder::stream::open_document;
using namespace std;

#include "Utils.hpp"
#include "Slide.hpp"
#include "NeuralGenome.hpp"
#include "GeneticAlgorithm.hpp"
#include "slide/Node.h"

class MongoDB{
    public:
        // constructors and destructor
        MongoDB(string host, string user, string password, int port=27017);
        MongoDB(const MongoDB& orig);
        virtual ~MongoDB();

        // methods
        mongocxx::database getDB(const string &db_name);
        mongocxx::collection getCollection(const mongocxx::database &db, const string &col_name);
        static pair<string,pair<vector<int>, vector<float>>> bsonToDatasetEntry(const bsoncxx::v_noabi::document::view bson);
        static pair<string,pair<vector<int>, vector<float>>> bsonToDatasetEntry(bsoncxx::stdx::optional<bsoncxx::document::value> opt_bson);
        pair<vector<string>,vector<pair<vector<int>, vector<float>>>> loadCvesFromYear(int year, int limit=0);
        pair<vector<string>,vector<pair<vector<int>, vector<float>>>> loadCvesFromYears(vector<int> years, int limit=0);
        pair<vector<string>,vector<float>> fetchGeneticSimulationData(string id);
        void claimGeneticSimulation(string id,string currentDatetime, string hostname);
        void finishGeneticSimulation(string id,string currentDatetime);
        void updateBestOnGeneticSimulation(string id, pair<float,int> candidate,string currentDatetime);
        void clearResultOnGeneticSimulation(string id);
        void appendResultOnGeneticSimulation(string id,int pop_size,int g,float best_out,long timestamp_ms);
        SPACE_SEARCH fetchEnvironmentData(string name);
        void clearPopulationNeuralGenomeVector(string pop_id,string currentDatetime);
        void addToPopulationNeuralGenomeVector(string pop_id,NeuralGenome* ng,string currentDatetime);
        void clearHallOfFameNeuralGenomeVector(string hall_id,string currentDatetime);
        void addToHallOfFameNeuralGenomeVector(string hall_id,NeuralGenome* ng,string currentDatetime);
        pair<vector<string>,vector<int>> fetchNeuralNetworkTrainMetadata(string id);
        void claimNeuralNetTrain(string id,string currentDatetime, string hostname);
        void finishNeuralNetTrain(string id,string currentDatetime);
        Hyperparameters* fetchHyperparametersData(string name);
        void appendTMetricsOnNeuralNet(string id,vector<pair<float,float>> metrics);
        void appendStatsOnNeuralNet(string id,string field_name,snn_stats stats);
        void appendWeightsOnNeuralNet(string id,const map<string, vector<float>> weights);
        vector<pair<vector<int>, vector<float>>> loadCveFromId(string cve);
        map<string, vector<float>> loadWeightsFromNeuralNet(string id);
        void storeEvalNeuralNetResult(string id,int correct,vector<string> cve_ids,vector<vector<pair<int,float>>> pred_labels,vector<pair<vector<int>, vector<float>>> labels,snn_stats stats);
	vector<pair<vector<int>, vector<float>>> filterFeatures(vector<pair<vector<int>, vector<float>>> in, vector<string> to_remove);
	vector<string> bsonToFeaturesName(const bsoncxx::v_noabi::document::view bson);
	vector<string> bsonToFeaturesName(bsoncxx::stdx::optional<bsoncxx::document::value> opt_bson);

    private:
        // methods
        static mongocxx::client getClient(const string &conn_str);
        bsoncxx::document::value castNeuralGenomeToBson(NeuralGenome* ng,bool store_weights=true);
        static string getStringFromEl(bsoncxx::document::element el);
        static float getFloatFromEl(bsoncxx::document::element el);
        static int getIntFromEl(bsoncxx::document::element el);
        // variables
        static mongocxx::instance inst;
        mongocxx::client client;
        mongocxx::v_noabi::gridfs::bucket weigths_bucket; // https://github.com/mongodb/mongo-cxx-driver/blob/releases/stable/examples/mongocxx/gridfs.cpp | https://stackoverflow.com/questions/60896129/delete-files-and-chunk-with-gridfsbucket
}; 
