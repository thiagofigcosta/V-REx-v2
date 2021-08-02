#pragma once

#include <iostream>
#include <utility>
#include <vector>

using namespace std;

#include "Genome.hpp"

class HallOfFame{
    public:
        // constructors and destructor
        HallOfFame(int maxNotables, bool lookingHighestFitness);
        HallOfFame(const HallOfFame& orig);
        virtual ~HallOfFame();

        // methods
        void update(vector<Genome*> candidates, int gen=-1);
        vector<Genome*> getNotables();
        static vector<pair<Genome*,string>> joinGenomeVector(vector<Genome*> candidates, vector<string> extraCandidatesArguments);
        static vector<Genome*> splitGenomeVector(vector<pair<Genome*,string>> genomeVec);
        pair<float,int> getBest();
    private:
        // variables
        vector<Genome*> notables;
        pair<float,int> best;
        bool looking_highest_fitness;
        int max_notables;
};