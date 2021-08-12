#include "HallOfFame.hpp"
#include "NeuralGenome.hpp"

HallOfFame::HallOfFame(int maxNotables, bool lookingHighestFitness) {
    looking_highest_fitness=lookingHighestFitness;
    max_notables=maxNotables;
    float starting_point;
    if (looking_highest_fitness){
        starting_point=numeric_limits<float>::min();
    }else{
        starting_point=numeric_limits<float>::max();
    }
    best=pair<float,int>(starting_point,-1);
}

HallOfFame::HallOfFame(const HallOfFame& orig) {
}

HallOfFame::~HallOfFame() {
    for (Genome* g:notables){
        delete g;
    }
}

void HallOfFame::update(vector<Genome*> candidates, int gen){
    vector<pair<float,Genome*>> notables_to_select;
    for (Genome *entry:notables){
        notables_to_select.push_back(pair<float,Genome*>(entry->getOutput(),entry));
    }
    for (Genome *entry:candidates){
        notables_to_select.push_back(pair<float,Genome*>(entry->getOutput(),entry));
    }
    sort(notables_to_select.begin(),notables_to_select.end(),[&](pair<float,Genome*> &lhs, pair<float,Genome*> &rhs){
        if (looking_highest_fitness){
            return lhs.first > rhs.first; // descending
        }else{
            return lhs.first < rhs.first; // ascending
        }
    });
    while(notables_to_select.size()>(size_t)max_notables){
        pair<float,Genome*> entry=notables_to_select.back();
        if(find(notables.begin(), notables.end(), entry.second) != notables.end()) {
            delete entry.second;
        }
        notables_to_select.pop_back();
    }
    vector<Genome*> new_notables;
    for (pair<float,Genome*> entry:notables_to_select){
        if(find(notables.begin(), notables.end(), entry.second) != notables.end()) {
            new_notables.push_back(entry.second);
        }else{
            if (dynamic_cast<NeuralGenome*>(entry.second)){
                new_notables.push_back(new NeuralGenome(*((NeuralGenome*)entry.second)));
            }else{
                new_notables.push_back(new Genome(*entry.second));
            }
        }
    }
    notables.clear();
    notables=new_notables;
    if ((looking_highest_fitness && notables[0]->getOutput()>best.first) || (!looking_highest_fitness && notables[0]->getOutput()<best.first)){
        best.first=notables[0]->getOutput();
        best.second=gen;
    }
}

vector<Genome*> HallOfFame::getNotables(){
    return notables;
}

vector<Genome*> HallOfFame::splitGenomeVector(vector<pair<Genome*,string>> genomeVec){
    vector<Genome*> out;
    for (pair<Genome*,string> notable:genomeVec){
        out.push_back(notable.first);
    }
    return out;
}

pair<float,int> HallOfFame::getBest(){
    return best;
}

vector<pair<Genome*,string>> HallOfFame::joinGenomeVector(vector<Genome*> candidates, vector<string> extraCandidatesArguments){
    vector<pair<Genome*,string>> out;
    for (size_t i=0;i<candidates.size();i++) {
        Genome *gen=candidates[i];
        string extra="";
        if (i<extraCandidatesArguments.size()){
            extra=extraCandidatesArguments[i];
        }
        if (dynamic_cast<NeuralGenome*>(gen)){
            out.push_back(pair<Genome*,string>(new NeuralGenome(*((NeuralGenome*)gen)),extra));
        }else{
            out.push_back(pair<Genome*,string>(new Genome(*gen),extra));
        }
    }
    return out;
}