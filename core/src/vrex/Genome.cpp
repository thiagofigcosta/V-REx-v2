#include "Genome.hpp"

Genome::Genome(SPACE_SEARCH space, function<float(Genome *self)> callback) {
    for (pair<int,int> limit:space.first){
        uniform_int_distribution<int> random(limit.first,limit.second);
        dna.first.push_back(random(Utils::RNG));
    }
    for (pair<float,float> limit:space.second){
        uniform_real_distribution<float> random(limit.first,limit.second);
        dna.second.push_back(random(Utils::RNG));
    }
    fitness=0;
    output=0;
    limits=space;
    evaluate_cb=callback;
    resetMtDna();
    id=Utils::genRandomUUID();
}

Genome::Genome(const Genome& orig, pair<vector<int>,vector<float>> new_dna) {
    mt_dna=orig.mt_dna;
    dna=new_dna;
    limits=orig.limits;
    fitness=0;
    output=0;
    evaluate_cb=orig.evaluate_cb;
    id=Utils::genRandomUUID();
}

Genome::Genome(const Genome& orig){
    mt_dna=orig.mt_dna;
    dna=orig.dna;
    limits=orig.limits;
    fitness=orig.fitness;
    output=orig.output;
    evaluate_cb=orig.evaluate_cb;
    id=orig.id;
}

Genome::~Genome() {
}

void Genome::resetMtDna(){
    mt_dna=Utils::genRandomUUID();
}

float Genome::getFitness(){
    return fitness;
}

void Genome::evaluate(){
    try {
        output=evaluate_cb(this);
    } catch (const exception& ex) {
        string print="Exception at Thread: ";
        int tid = omp_get_thread_num();
        print+=std::to_string(tid)+"\n";
        string what = ex.what();
        print+=what+"\n";
        cout<<print;
    }
}

bool Genome::operator< (Genome& o){
    return fitness < o.fitness;
}

bool Genome::compare (Genome* l, Genome* r){
    return l->fitness < r->fitness;
}

void Genome::setFitness(float nFit){
    fitness=nFit;
}

float Genome::getOutput(){
    return output;
}

pair<vector<int>,vector<float>> Genome::getDna(){
    return dna;
}

void Genome::checkLimits(){
    for(size_t i=0;i<limits.first.size();i++){
        if (dna.first[i]<limits.first[i].first){
            dna.first[i]=limits.first[i].first;
        }else if (dna.first[i]>limits.first[i].second){
            dna.first[i]=limits.first[i].second;
        }
    }
    for(size_t i=0;i<limits.second.size();i++){
        if (dna.second[i]<limits.second[i].first){
            dna.second[i]=limits.second[i].first;
        }else if (dna.second[i]>limits.second[i].second){
            dna.second[i]=limits.second[i].second;
        }
    }
}

void Genome::setDna(pair<vector<int>,vector<float>> new_dna){
    dna=new_dna;
}

string Genome::to_string(){
    string out="Output: "+std::to_string(output)+" Fitness: "+std::to_string(fitness);
    out+=" -- Genes{ Int: ";
    for(size_t i=0;i<dna.first.size();i++){
        out+=std::to_string(dna.first[i])+" ";
    }
    out+="Float: ";
    for(size_t i=0;i<dna.second.size();i++){
        out+=std::to_string(dna.second[i])+" ";
    }
    out+="}";
    return out;
}

boost::uuids::uuid Genome::getMtDna(){
    return mt_dna;
}

boost::uuids::uuid Genome::getId(){
    return id;
}