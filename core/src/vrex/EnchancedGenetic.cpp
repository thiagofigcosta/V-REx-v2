#include "EnchancedGenetic.hpp"
#include "NeuralGenome.hpp"

const float EnchancedGenetic::will_of_D_percent=0.07;
const float EnchancedGenetic::recycle_threshold_percent=0.03;
const float EnchancedGenetic::cutoff_population_limit=1.3;

EnchancedGenetic::EnchancedGenetic(int maxChildren, int maxAge, float mutationRate, float sexRate, float recycleRate){
    mutation_rate=mutationRate;
    sex_rate=sexRate;
    recycle_rate=recycleRate;
    max_children=maxChildren;
    max_children=maxChildren;
    max_age=maxAge;
    max_population=0;
}

EnchancedGenetic::EnchancedGenetic(const EnchancedGenetic& orig){
    index_age=orig.index_age;
    index_max_age=orig.index_max_age;
    index_max_children=orig.index_max_children;
    max_population=orig.max_population;
    max_age=orig.max_age;
    max_children=orig.max_children;
    mutation_rate=orig.mutation_rate;
    sex_rate=orig.sex_rate;
    recycle_rate=orig.recycle_rate;
    looking_highest_fitness=orig.looking_highest_fitness;
}

EnchancedGenetic::~EnchancedGenetic(){
}

void EnchancedGenetic::setMaxPopulation(int maxPopulation){
    max_population=maxPopulation;
}

void EnchancedGenetic::select(vector<Genome*> &currentGen){
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
    uniform_real_distribution<float> roulette(0,fitness_sum);
    vector<vector<Genome*>> all_parents;
    set<Genome*> useful_beings;
    for (size_t i=0;i<currentGen.size()/2;i++){
        vector<Genome*> parents;
        Genome* backup=nullptr;
        for (int c=0;c<2;c++){
            float sorted=roulette(Utils::RNG);
            float roulette_val=0;
            for(size_t g=0;g<currentGen.size();g++){
                roulette_val+=(currentGen[g]->getFitness()+offset);
                if (sorted<=roulette_val){
                    if ( parents.size()<1 || !isRelative(parents[0],currentGen[g])){
                        parents.push_back(currentGen[g]);
                        break;
                    }else if (!backup){
                        backup=currentGen[g];
                    }
                }
            }
        }
        if (parents.size()!=2){
            parents.push_back(backup);
        }
        useful_beings.insert(parents[0]);
        useful_beings.insert(parents[1]);
        all_parents.push_back(parents);
    }
    vector<Genome*> useful_beings_vec;
    copy(useful_beings.begin(), useful_beings.end(), back_inserter(useful_beings_vec));
    useful_beings.clear();
    vector<Genome*> useless_beings_vec = Utils::subtractVectors(currentGen,useful_beings_vec);
    for (Genome *g:useless_beings_vec){
        delete g;
    }
    useless_beings_vec.clear();
    current_population_size=currentGen.size();
    vector<Genome*> nxt_gen;
    for(vector<Genome*> parents:all_parents){
        vector<Genome*> children=sex(parents[0],parents[1]);
        nxt_gen.insert(nxt_gen.end(),children.begin(),children.end());
        current_population_size+=children.size()-2;
    }
    for (Genome *g:useful_beings_vec){
        delete g;
    }
    useful_beings_vec.clear();
    currentGen.clear();
    currentGen.insert(currentGen.end(),nxt_gen.begin(),nxt_gen.end());
    nxt_gen.clear();
}

void EnchancedGenetic::fit(vector<Genome*> &currentGen){
    int signal=1;
    if (!looking_highest_fitness){
        signal=-1;
    }
    int i=0;
    bool recycled=false;
    while (i==0||recycled){
        for (Genome *g:currentGen){
            g->setFitness(g->getOutput()*signal);
        }
        sort(currentGen.begin(),currentGen.end(),Genome::compare);
        for (size_t i=0;i<currentGen.size();i++){
            currentGen[i]->setFitness(i+1);
        }
        if (i++<1){
            recycled=recycleBadIndividuals(currentGen);
        }else{
            recycled=false;
            int cutout_limit=max_population*cutoff_population_limit;
            int cutout_diff=(currentGen.size()-cutout_limit);
            if(cutout_diff>0){
                for (size_t e=0;e<(size_t)cutout_diff;e++){
                    delete currentGen[e];
                }
                currentGen.erase(currentGen.begin(),currentGen.begin()+cutout_diff);
            }
        }
    }
}

vector<Genome*> EnchancedGenetic::sex(Genome* father, Genome* mother){
    vector<Genome*> family;
    if (Utils::getRandomBetweenZeroAndOne()<sex_rate){
        pair<vector<int>,vector<float>> father_dna=father->getDna();
        pair<vector<int>,vector<float>> mother_dna=mother->getDna();
        float child_rnd=Utils::getRandomBetweenZeroAndOne();
        int childs=child_rnd*father_dna.first[index_max_children]+(1-child_rnd)*mother_dna.first[index_max_children];
        if (childs<1)
            childs=1;
        if (childs>max_children)
            childs=max_children;
        childs=ceil(childs*calcBirthRate(current_population_size));
        for (int c=0;c<childs;c++){
            pair<vector<int>,vector<float>> child=pair<vector<int>,vector<float>>(vector<int>(),vector<float>());
            for(size_t i=0;i<father_dna.first.size();i++){
                if (i != index_age){
                    float gene_share=Utils::getRandomBetweenZeroAndOne();
                    bool heritage_mother=Utils::getRandomBetweenZeroAndOne()>0.5;
                    if (heritage_mother){
                        child.first.push_back((1-gene_share)*father_dna.first[i]+gene_share*mother_dna.first[i]);
                    }else{
                        child.first.push_back(gene_share*father_dna.first[i]+(1-gene_share)*mother_dna.first[i]);
                    }
                }else{
                    child.first.push_back(-1); //age
                }
            }
            for(size_t i=0;i<father_dna.second.size();i++){
                float gene_share=Utils::getRandomBetweenZeroAndOne();
                bool heritage_mother=Utils::getRandomBetweenZeroAndOne()>0.5;
                if (heritage_mother){
                    child.second.push_back((1-gene_share)*father_dna.second[i]+gene_share*mother_dna.second[i]);
                }else{
                    child.second.push_back(gene_share*father_dna.second[i]+(1-gene_share)*mother_dna.second[i]);
                }
            }
            if (dynamic_cast<NeuralGenome*>(mother)){
                family.push_back(new NeuralGenome(*((NeuralGenome*)mother),child));
            }else{
                family.push_back(new Genome(*mother,child));
            }
        }
    }
    if (dynamic_cast<NeuralGenome*>(mother)){
        family.push_back(new NeuralGenome(*((NeuralGenome*)mother)));
        family.push_back(new NeuralGenome(*((NeuralGenome*)father)));
    }else{
        family.push_back(new Genome(*mother));
        family.push_back(new Genome(*father));
    }
    return family;
}

Genome* EnchancedGenetic::age(Genome* individual, int cur_population_size){
    pair<vector<int>,vector<float>> dna=individual->getDna();
    dna.first[index_age]++;
    individual->setDna(dna);
    if (dna.first[index_age]>dna.first[index_max_age]){
        if (individual->getFitness()<=(1-will_of_D_percent)*cur_population_size){
            delete individual;
            return nullptr; // dead
        }else{
            individual->resetMtDna();
        }
    }
    return individual;
}

void EnchancedGenetic::mutate(vector<Genome*> &individuals){
    vector<Genome*> nxt_gen;
    int cur_pop_size=individuals.size();
    for (Genome *g:individuals){
        Genome* out=age(g,cur_pop_size);
        if (out){
            mutate(out);
            nxt_gen.push_back(out);
        }
    }
    individuals.clear();
    individuals.insert(individuals.end(),nxt_gen.begin(),nxt_gen.end());
    nxt_gen.clear();
}

unique_ptr<GeneticAlgorithm> EnchancedGenetic::clone(){
    return make_unique<EnchancedGenetic>(*this); 
}

float EnchancedGenetic::randomize(){
    float r=Utils::getRandomBetweenZeroAndOne();
    if (r<=0.3){
        uniform_real_distribution<float> random(0,0.07);
        r=random(Utils::RNG);
    }else if (r<=0.5){
        uniform_real_distribution<float> random(0,0.11);
        r=random(Utils::RNG);
    }else if (r<=0.6){
        uniform_real_distribution<float> random(0.03,0.13);
        r=random(Utils::RNG);
    }else if (r<=0.7){
        uniform_real_distribution<float> random(0.06,0.15);
        r=random(Utils::RNG);
    }else if (r<=0.8){
        uniform_real_distribution<float> random(0.08,0.24);
        r=random(Utils::RNG);
    }else if (r<=0.9){
        uniform_real_distribution<float> random(0.1,0.27);
        r=random(Utils::RNG);
    }else if (r<=0.97){
        uniform_real_distribution<float> random(0.23,0.30);
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

void EnchancedGenetic::mutate(Genome *individual){
    pair<vector<int>,vector<float>> xmen_dna=individual->getDna();
    for(size_t i=0;i<xmen_dna.first.size();i++){
        if (i != index_age && i!= index_max_age && i != index_max_children) {
            if (Utils::getRandomBetweenZeroAndOne()<mutation_rate){
                xmen_dna.first[i]*=randomize();
            }
        }
    }
    for(size_t i=0;i<xmen_dna.second.size();i++){
        if (Utils::getRandomBetweenZeroAndOne()<mutation_rate){
            xmen_dna.second[i]*=randomize();
        }
    }
    individual->setDna(xmen_dna);

    int age=xmen_dna.first[index_age];
    individual->checkLimits();

    xmen_dna=individual->getDna();
    xmen_dna.first[index_age]=age;
    individual->setDna(xmen_dna);
}

void EnchancedGenetic::forceMutate(Genome *individual){
    pair<vector<int>,vector<float>> xmen_dna=individual->getDna();
    for(size_t i=0;i<xmen_dna.first.size();i++){
        if (i != index_age && i!= index_max_age && i != index_max_children) {
            xmen_dna.first[i]*=randomize();
        }
    }
    for(size_t i=0;i<xmen_dna.second.size();i++){
        xmen_dna.second[i]*=randomize();
    }
    individual->setDna(xmen_dna);

    int age=xmen_dna.first[index_age];
    individual->checkLimits();

    xmen_dna=individual->getDna();
    xmen_dna.first[index_age]=age;
    individual->setDna(xmen_dna);
}

bool EnchancedGenetic::isRelative(Genome *father, Genome *mother){
    return father->getMtDna()==mother->getMtDna();
}

int EnchancedGenetic::getLifeLeft(Genome *individual){
    return individual->getDna().first[index_max_age]-individual->getDna().first[index_age];
}

bool EnchancedGenetic::recycleBadIndividuals(vector<Genome*> &individuals){
    int threshold=(int)individuals.size()*recycle_threshold_percent+1;
    bool recycled=false;
    for (size_t t=0;t<individuals.size();t++){
        if (individuals[t]->getFitness()<threshold){
            if (Utils::getRandomBetweenZeroAndOne()<recycle_rate){
                delete individuals[t];
                int idx_mirror=individuals.size()-(will_of_D_percent*individuals.size()*Utils::getRandomBetweenZeroAndOne())-1; // exploit
                pair<vector<int>,vector<float>> dna=individuals[idx_mirror]->getDna();
                if (dynamic_cast<NeuralGenome*>(individuals[idx_mirror])){
                    individuals[t]=new NeuralGenome(*((NeuralGenome*)individuals[idx_mirror]),dna);
                }else{
                    individuals[t]=new Genome(*individuals[idx_mirror],dna);
                }
                dna.first[index_age]=-1;
                individuals[t]->setDna(dna);
                forceMutate(individuals[t]); // explore
                individuals[t]->evaluate();
                recycled=true;
            }
        }else{
            break;
        }
    }
    return recycled;
}


float EnchancedGenetic::calcBirthRate(int cur_population_size){
    float out=1-(cur_population_size/max_population)*0.22;
    return out;
}

SPACE_SEARCH EnchancedGenetic::enrichSpace(SPACE_SEARCH &space){
    index_age=space.first.size();
    space.first.push_back(pair<int,int>(0,0));
    index_max_age=space.first.size();
    space.first.push_back(pair<int,int>(max_age/2,max_age*abs(randomize())));
    index_max_children=space.first.size();
    space.first.push_back(pair<int,int>(1,max_children*abs(randomize())));
    return space;
}
