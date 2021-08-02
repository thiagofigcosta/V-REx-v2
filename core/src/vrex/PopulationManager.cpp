#include "PopulationManager.hpp"
#include "NeuralGenome.hpp"
#include "HallOfFame.hpp"

PopulationManager::PopulationManager(GeneticAlgorithm &galg, SPACE_SEARCH space, function<float(Genome *self)> callback,int startPopulationSize, bool searchHighestFitness, bool useNeuralGenome, bool printDeltas,function<void(int pop_size,int g,float best_out,long timestamp_ms,vector<Genome*> population,HallOfFame *hall_of_fame)> afterGen_cb){
    ga=galg.clone();
    looking_highest_fitness=searchHighestFitness;
    if (typeid(*ga) == typeid(EnchancedGenetic)){
        EnchancedGenetic* casted_ga=dynamic_cast<EnchancedGenetic*>(ga.get());
        casted_ga->setMaxPopulation(startPopulationSize*2);
        casted_ga->enrichSpace(space);
    }else{
        space=ga->enrichSpace(space);
    }
    for (int i=0;i<startPopulationSize;i++){
        if (useNeuralGenome){
            population.push_back(new NeuralGenome(space,callback));
        }else{
            population.push_back(new Genome(space,callback));
        }
    }
    print_deltas=printDeltas;
    hall_of_fame=nullptr;
    after_gen_cb=afterGen_cb;
}

PopulationManager::PopulationManager(GeneticAlgorithm* galg, SPACE_SEARCH space, function<float(Genome *self)> callback,int startPopulationSize, bool searchHighestFitness, bool useNeuralGenome, bool printDeltas,function<void(int pop_size,int g,float best_out,long timestamp_ms,vector<Genome*> population,HallOfFame *hall_of_fame)> afterGen_cb){
    ga=unique_ptr<GeneticAlgorithm>(move(galg)); // ga.release(galg);
    looking_highest_fitness=searchHighestFitness;
    if (typeid(*ga) == typeid(EnchancedGenetic)){
        EnchancedGenetic* casted_ga=dynamic_cast<EnchancedGenetic*>(ga.get());
        casted_ga->setMaxPopulation(startPopulationSize*2);
        casted_ga->enrichSpace(space);
    }else{
        space=ga->enrichSpace(space);
    }
    for (int i=0;i<startPopulationSize;i++){
        if (useNeuralGenome){
            population.push_back(new NeuralGenome(space,callback));
        }else{
            population.push_back(new Genome(space,callback));
        }
    }
    print_deltas=printDeltas;
    hall_of_fame=nullptr;
    after_gen_cb=afterGen_cb;
}

PopulationManager::PopulationManager(const PopulationManager& orig) {
}

PopulationManager::~PopulationManager() {
    GeneticAlgorithm* ptr=ga.release();
    ga.get_deleter() ( ptr );
    for (Genome* g:population){
        delete g;
    }
}

void PopulationManager::setHallOfFame(HallOfFame &hallOfFame){
    hall_of_fame=&hallOfFame;
}

void PopulationManager::setHallOfFame(HallOfFame *hallOfFame){
    hall_of_fame=hallOfFame;
}

const int PopulationManager::mt_dna_validity=15;

void PopulationManager::naturalSelection(int gens,bool verbose){
    ga->setLookingHighestFitness(looking_highest_fitness);
    chrono::high_resolution_clock::time_point t1,t2;
    float best_out;
    const int PRINT_REL_FREQUENCY=10;
    for (int g=0;g<gens;){
        t1 = chrono::high_resolution_clock::now();
        if (looking_highest_fitness){
            best_out=numeric_limits<float>::min();
        }else{
            best_out=numeric_limits<float>::max();
        }
        if(verbose)
            cout<<"\tEvaluating individuals...\n";
        int p=0;
        for(Genome *individual:population){ // evaluate output
            individual->evaluate();
            float ind_out=individual->getOutput();
            if (looking_highest_fitness){
                if (ind_out>best_out)
                    best_out=ind_out;
            }else{
                if (ind_out<best_out)
                    best_out=ind_out;
            }
            if(verbose){
                float percent=++p/(float)population.size()*100;
                if (int(percent)%PRINT_REL_FREQUENCY==0){
                    cout<<"\t\tprogress: "<<percent<<"%\n";
                }
            }
        }
        if(verbose){
            cout<<"\tEvaluated individuals...OK\n";
            cout<<"\tCalculating fitness...\n";
        }
        ga->fit(population); // calculate fitness
        if(verbose)
            cout<<"\tCalculated fitness...OK\n";
        if (hall_of_fame){
            if(verbose)
                cout<<"\tSetting hall of fame...\n";
            hall_of_fame->update(population,g+1); // store best on hall of fame
            if(verbose)
                cout<<"\tSetted hall of fame...OK\n";
        }
        if (++g<gens){
            if(verbose)
                cout<<"\tSelecting and breeding individuals...\n";
            ga->select(population); // selection + sex
            if(verbose){
                cout<<"\tSelected and breed individuals...OK\n";
                cout<<"\tMutating (and aging if Enchanced) individuals...\n";
            }
            ga->mutate(population);  // mutation + age if aged algorithm
            if (g>0&&g%mt_dna_validity==0){
                for(Genome *individual:population){
                    individual->resetMtDna();
                }
            }
            if(verbose)
                cout<<"\tMutated individuals...OK\n";
        }else{
            sort(population.begin(),population.end(),Genome::compare);
        }
        t2 = chrono::high_resolution_clock::now();
        long delta = chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        if(after_gen_cb){
            after_gen_cb((int)population.size(),g,best_out,delta,population,hall_of_fame);
        }
        if (print_deltas) {
            cout<<"Generation "<<g<<" of "<<gens<<", size: "<<population.size()<<" takes: "<<Utils::msToHumanReadable(delta)<<endl;
        }
    }
}

vector<Genome*> PopulationManager::getPopulation(){
    return population;
}
