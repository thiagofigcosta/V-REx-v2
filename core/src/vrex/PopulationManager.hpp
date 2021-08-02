#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <functional>
#include <chrono>

using namespace std;
class Genome;
class HallOfFame;

#include "GeneticAlgorithm.hpp"
#include "EnchancedGenetic.hpp"


class PopulationManager{
    public:
        // constructors and destructor
        PopulationManager(GeneticAlgorithm &galg, SPACE_SEARCH space, function<float(Genome *self)> callback, int startPopulationSize, bool searchHighestFitness, bool useNeuralGenome=false, bool printDeltas=false,function<void(int pop_size,int g,float best_out,long timestamp_ms,vector<Genome*> population,HallOfFame *hall_of_fame)> afterGen_cb=nullptr);
        PopulationManager(GeneticAlgorithm* galg, SPACE_SEARCH space, function<float(Genome *self)> callback, int startPopulationSize, bool searchHighestFitness, bool useNeuralGenome=false, bool printDeltas=false,function<void(int pop_size,int g,float best_out,long timestamp_ms,vector<Genome*> population,HallOfFame *hall_of_fame)> afterGen_cb=nullptr);
        PopulationManager(const PopulationManager& orig);
        virtual ~PopulationManager();

        // methods
        void setHallOfFame(HallOfFame &hallOfFame);
        void setHallOfFame(HallOfFame *hallOfFame);
        void naturalSelection(int gens,bool verbose=false);
        vector<Genome*> getPopulation();

    private:
        // variables
        unique_ptr<GeneticAlgorithm> ga;
        vector<Genome*> population;
        bool looking_highest_fitness;
        HallOfFame *hall_of_fame;
        bool print_deltas;
        static const int mt_dna_validity; // TODO change to %
        function<void(int pop_size,int g,float best_out,long timestamp_ms,vector<Genome*> population,HallOfFame *hall_of_fame)> after_gen_cb;
};