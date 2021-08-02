#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <set>
#include <chrono>

using namespace std;
class Genome;

#include "GeneticAlgorithm.hpp"
#include "Genome.hpp"
#include "Utils.hpp"

class EnchancedGenetic : public GeneticAlgorithm{ 
    public:
        // constructors and destructor
        EnchancedGenetic(int maxChildren, int maxAge, float mutationRate, float sexRate, float recycleRate);
        EnchancedGenetic(const EnchancedGenetic& orig);
        virtual ~EnchancedGenetic();

        //methods
        void setMaxPopulation(int maxPopulation);

        //methods override
        void select(vector<Genome*> &currentGen);
        void fit(vector<Genome*> &currentGen);
        vector<Genome*> sex(Genome* father, Genome* mother);
        void mutate(vector<Genome*> &individuals);
        unique_ptr<GeneticAlgorithm> clone();
        SPACE_SEARCH enrichSpace(SPACE_SEARCH &space);

    private:
        // variables 
        size_t index_age;
        size_t index_max_age;
        size_t index_max_children;
        int max_population;
        int max_age;
        int max_children;
        float mutation_rate;
        float sex_rate;
        float recycle_rate;
        int current_population_size;
        static const float will_of_D_percent;
        static const float recycle_threshold_percent;
        static const float cutoff_population_limit;

        //methods 
        Genome* age(Genome *individual, int cur_population_size);
        void mutate(Genome *individual);
        void forceMutate(Genome *individual);
        bool isRelative(Genome *father, Genome *mother);
        float randomize();
        int getLifeLeft(Genome *individual);
        bool recycleBadIndividuals(vector<Genome*> &individuals);
        float calcBirthRate(int cur_population_size);

};
