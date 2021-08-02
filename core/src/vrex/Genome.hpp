#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <functional>
#include <omp.h>

#include "PopulationManager.hpp"
#include "Utils.hpp"

using namespace std;

class Genome{
    public:
        // constructors and destructor
        Genome(){};
        Genome(SPACE_SEARCH space, function<float(Genome *self)> callback);
        Genome(const Genome& orig, pair<vector<int>,vector<float>> new_dna);
        Genome(const Genome& orig);
        virtual ~Genome();
        void evaluate();

        // methods
        float getFitness();
        float getOutput();
        void setFitness(float nFit);
        void checkLimits();
        bool operator< (Genome& o);
        static bool compare (Genome* l, Genome* r);
        pair<vector<int>,vector<float>> getDna();
        void setDna(pair<vector<int>,vector<float>> new_dna);
        string to_string();
        boost::uuids::uuid getMtDna();
        void resetMtDna();
        boost::uuids::uuid getId();

    protected:
        // variables
        pair<vector<int>,vector<float>> dna;
        boost::uuids::uuid mt_dna;
        boost::uuids::uuid id;
        SPACE_SEARCH limits;
        float fitness;
        float output;
        function<float(Genome *self)> evaluate_cb; //float (*evaluate_cb)(pair<vector<int>,vector<float>> dna)
};