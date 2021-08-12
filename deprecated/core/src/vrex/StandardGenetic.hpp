#pragma once

#include <iostream>
#include <utility>
#include <vector>

using namespace std;
class Genome;
enum class StdGeneticRankType { RELATIVE, ABSOLUTE };

#include "GeneticAlgorithm.hpp"
#include "Genome.hpp"
#include "Utils.hpp"

class StandardGenetic : public GeneticAlgorithm{ 
    public:
        // constructors and destructor
        StandardGenetic(float mutationRate, float sexRate, StdGeneticRankType rankType=StdGeneticRankType::RELATIVE);
        StandardGenetic(const StandardGenetic& orig);
        virtual ~StandardGenetic();

        //methods override
        void select(vector<Genome*> &currentGen);
        void fit(vector<Genome*> &currentGen);
        vector<Genome*> sex(Genome* father, Genome* mother);
        void mutate(vector<Genome*> &individuals);
        unique_ptr<GeneticAlgorithm> clone();

    private:
        // variables 
        float mutation_rate;
        float sex_rate;
        StdGeneticRankType rank_type;

        //methods 
        float genRandomFactor();
};