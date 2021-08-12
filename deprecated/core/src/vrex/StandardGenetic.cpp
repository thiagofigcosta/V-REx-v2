#include "StandardGenetic.hpp"
#include "NeuralGenome.hpp"


StandardGenetic::StandardGenetic(float mutationRate, float sexRate, StdGeneticRankType rankType){
    mutation_rate=mutationRate;
    sex_rate=sexRate;
    rank_type=rankType;
}

StandardGenetic::StandardGenetic(const StandardGenetic& orig){
    mutation_rate=orig.mutation_rate;
    sex_rate=orig.sex_rate;
    rank_type=orig.rank_type;
    looking_highest_fitness=orig.looking_highest_fitness;
}

StandardGenetic::~StandardGenetic(){
}

void StandardGenetic::select(vector<Genome*> &currentGen){
    // roulette wheel
    sort(currentGen.begin(),currentGen.end(),Genome::compare);
    float min=currentGen[0]->getFitness();
    float offset=0;
    float fitness_sum=0;
    if (min<=0){
        offset=abs(min);
    }
    for (Genome *g:currentGen){
        fitness_sum+=g->getFitness()+offset;
    }
    vector<Genome*> nxt_gen;
    uniform_real_distribution<float> roulette(0,fitness_sum);
    for (size_t i=0;i<currentGen.size()/2;i++){
        vector<Genome*> parents;
        for (int c=0;c<2;c++){
            float sorted=roulette(Utils::RNG);
            float roulette_val=0;
            for(Genome *g:currentGen){
                roulette_val+=(g->getFitness()+offset);
                if (sorted<=roulette_val){
                    parents.push_back(g);
                    break;
                }
            }
        }
        vector<Genome*> children=sex(parents[0],parents[1]);
        nxt_gen.insert(nxt_gen.end(),children.begin(),children.end());
    }
    for (Genome *g:currentGen){
        delete g;
    }
    currentGen.clear();
    currentGen.insert(currentGen.end(),nxt_gen.begin(),nxt_gen.end());
    nxt_gen.clear();
}

void StandardGenetic::fit(vector<Genome*> &currentGen){
    int signal=1;
    if (!looking_highest_fitness){
        signal=-1;
    }
    for (Genome *g:currentGen){
        switch(rank_type){
            case StdGeneticRankType::ABSOLUTE:
            case StdGeneticRankType::RELATIVE:
                g->setFitness(g->getOutput()*signal);
                break;
        }
    }
    if (rank_type==StdGeneticRankType::RELATIVE){
        sort(currentGen.begin(),currentGen.end(),Genome::compare);
        for (size_t i=0;i<currentGen.size();i++){
            currentGen[i]->setFitness(100.0/(currentGen.size()-i+2));
        }
    }
}

vector<Genome*> StandardGenetic::sex(Genome* father, Genome* mother){
    vector<Genome*> children;
    if (Utils::getRandomBetweenZeroAndOne()<sex_rate){
        pair<vector<int>,vector<float>> father_dna=father->getDna();
        pair<vector<int>,vector<float>> mother_dna=mother->getDna();
        pair<vector<int>,vector<float>> child_1=pair<vector<int>,vector<float>>(vector<int>(),vector<float>());
        pair<vector<int>,vector<float>> child_2=pair<vector<int>,vector<float>>(vector<int>(),vector<float>());

        for(size_t i=0;i<father_dna.first.size();i++){
            float gene_share=Utils::getRandomBetweenZeroAndOne();
            child_1.first.push_back(gene_share*father_dna.first[i]+(1-gene_share)*mother_dna.first[i]);
            child_2.first.push_back((1-gene_share)*father_dna.first[i]+gene_share*mother_dna.first[i]);
        }
        for(size_t i=0;i<father_dna.second.size();i++){
            float gene_share=Utils::getRandomBetweenZeroAndOne();
            child_1.second.push_back(gene_share*father_dna.second[i]+(1-gene_share)*mother_dna.second[i]);
            child_2.second.push_back((1-gene_share)*father_dna.second[i]+gene_share*mother_dna.second[i]);
        }
        if (dynamic_cast<NeuralGenome*>(mother)){
            children.push_back(new NeuralGenome(*((NeuralGenome*)mother),child_1));
            children.push_back(new NeuralGenome(*((NeuralGenome*)mother),child_2));
        }else{
            children.push_back(new Genome(*mother,child_1));
            children.push_back(new Genome(*mother,child_2));
        }
    }else{
        if (dynamic_cast<NeuralGenome*>(mother)){
            children.push_back(new NeuralGenome(*((NeuralGenome*)mother)));
            children.push_back(new NeuralGenome(*((NeuralGenome*)father)));
        }else{
            children.push_back(new Genome(*mother));
            children.push_back(new Genome(*father));
        }
    }
    return children;
}

void StandardGenetic::mutate(vector<Genome*> &individuals){
    for (Genome *g:individuals){
        pair<vector<int>,vector<float>> dna=g->getDna();
        for(size_t i=0;i<dna.first.size();i++){
            if (Utils::getRandomBetweenZeroAndOne()<mutation_rate){
                dna.first[i]*=genRandomFactor();
            }
        }
        for(size_t i=0;i<dna.second.size();i++){
            if (Utils::getRandomBetweenZeroAndOne()<mutation_rate){
                dna.second[i]*=genRandomFactor();
            }
        }
        g->setDna(dna);
        g->checkLimits();
    }
}

unique_ptr<GeneticAlgorithm> StandardGenetic::clone(){
    return make_unique<StandardGenetic>(*this); 
}

float StandardGenetic::genRandomFactor(){
    float r=Utils::getRandomBetweenZeroAndOne();
    if (r<=0.3){
        uniform_real_distribution<float> random(0,0.06);
        r=random(Utils::RNG);
    }else if (r<=0.8){
        uniform_real_distribution<float> random(0,0.11);
        r=random(Utils::RNG);
    }else if (r<=0.9){
        uniform_real_distribution<float> random(0.09,0.16);
        r=random(Utils::RNG);
    }else if (r<=0.97){
        uniform_real_distribution<float> random(0.15,0.23);
        r=random(Utils::RNG);
    }else{
        uniform_real_distribution<float> random(0.333,0.666);
        r=random(Utils::RNG);
    }
    if (Utils::getRandomBetweenZeroAndOne()>0.5){
        r=-(1+r);
    }else{
        r=(1+r);
    }
    return r;
}