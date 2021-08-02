#include "Utils.hpp"


#ifdef _WIN32 // windows
  const string Utils::FILE_SEPARATOR="\\";
#elif __APPLE__ // mac 
  const string Utils::FILE_SEPARATOR="/";
#else // linux <3
  const string Utils::FILE_SEPARATOR="/";
#endif
string Utils::RESOURCES_FOLDER="../../res";
mt19937_64 Utils::RNG = Utils::getRandomGenerator();
uniform_real_distribution<float> Utils::dist_zero_one=uniform_real_distribution<float>(0,nextafter(1, numeric_limits<float>::max()));

Utils::Utils() {
}

Utils::Utils(const Utils& orig) {
}

Utils::~Utils() {
}

mt19937_64 Utils::getRandomGenerator(){
    random_device rd;
    mt19937_64 mt(rd());
    return mt;
}

vector<string> Utils::splitString(string str, string token){
    vector<string> result;
    while(str.size()){
        size_t index = str.find(token);
        if(index!=string::npos){
            result.push_back(str.substr(0,index));
            str = str.substr(index+token.size());
            if(str.size()==0)result.push_back(str);
        }else{
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

vector<pair<string, vector<float>>> Utils::readLabeledCsvDataset(const string filename){
    vector<pair<string, vector<float>>> result;
    ifstream file(filename);
    if(!file.is_open()) throw runtime_error("Could not open file: "+filename);
    string line;
    while (getline(file, line)){
        if (!line.empty()){
            pair<string, vector<float>> parsed;
            vector<string> fields = Utils::splitString(line,",");
            parsed.first=fields.back();
            parsed.second={};
            for (vector<string>::size_type i=0;i<fields.size()-1; i++){
                parsed.second.push_back(stof(fields[i]));
            }
            result.push_back(parsed);
        }
    }
    file.close();
    return result;
}

void Utils::ltrimInPlace(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !isspace(ch);
    }));
}

void Utils::rtrimInPlace(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !isspace(ch);
    }).base(), s.end());
}

void Utils::trimInPlace(string &s) {
    ltrimInPlace(s);
    rtrimInPlace(s);
}

string Utils::ltrim(string s) {
    ltrimInPlace(s);
    return s;
}

string Utils::rtrim(string s) {
    rtrimInPlace(s);
    return s;
}

string Utils::trim(string s) {
    trimInPlace(s);
    return s;
}

string Utils::getResourcesFolder(){
    if (Utils::runningOnDockerContainer()){ 
        Utils::RESOURCES_FOLDER="/vrex/res";
    }
    return Utils::RESOURCES_FOLDER;
}

string Utils::getResourcePath(string filename){
    return Utils::joinPath(Utils::getResourcesFolder(),filename);
}

string Utils::joinPath(string base,string sufix){
    trimInPlace(base);
    trimInPlace(sufix);
    if (!Utils::endsWith(base,Utils::FILE_SEPARATOR)){
        base+=Utils::FILE_SEPARATOR;
    }
    return base+sufix;
}

bool Utils::endsWith(const string &str, const string &ending){
    if (ending.size() > str.size()) return false;
    return equal(ending.rbegin(), ending.rend(), str.rbegin());
}

bool Utils::runningOnDockerContainer(){
    string path = "/proc/self/cgroup";
    return Utils::checkIfFileExists("/.dockerenv") ||
        (Utils::checkIfIsFile(path) && Utils::checkIfFileContainsString(path,"docker"));
}

bool Utils::checkIfFileExists(const string &path){
    struct stat buffer;   
    return (stat (path.c_str(), &buffer) == 0); 
}

bool Utils::checkIfIsFile(const string &path){
    struct stat buffer;
    if (stat (path.c_str(), &buffer) == 0){
        if (buffer.st_mode & S_IFDIR){
            return false;
        }else if (buffer.st_mode & S_IFREG){
            return true;
        }else{
            throw runtime_error("File "+path+" is a special file!");
        }
    }else{
        throw runtime_error("File "+path+" does not exists!");
    }
}

bool Utils::checkIfFileContainsString(const string &path,const string &str){
    bool found=false;
    ifstream file(path);
    if(!file.is_open()) {
        throw runtime_error("Could not open file");
    }
    string line;
    while(getline(file, line) && !found){
        if(line.find(str) != string::npos){
            found = true;
            break;
        }
    }
    file.close();
    return found;
}

vector<pair<vector<int>, vector<float>>> Utils::extractSubVector(const vector<pair<vector<int>, vector<float>>> &vec, int start, int size){
    if ((size_t)start>=vec.size()){
        throw runtime_error("Start position ("+to_string(start)+") bigger than vector size ("+to_string(vec.size())+")");
    }
    int diff=vec.size()-(start+size);
    if (diff<0){
        size+=diff;
    }
    return {vec.begin()+start, vec.begin()+start+size };
}

pair<vector<pair<int, vector<float>>>,map<string,int>> Utils::enumfyDataset(const vector<pair<string, vector<float>>> &vec){
    set<string> label;
    for (pair<string, vector<float>> entry:vec){
        label.insert(entry.first);
    }
    map<string,int> equivalence;
    int i=0;
    for (string l:label){
        equivalence[l]=i++;
    }
    vector<pair<int, vector<float>>> enumfied_data;
    for (pair<string, vector<float>> entry:vec){
        enumfied_data.push_back(pair<int, vector<float>>(equivalence[entry.first],entry.second));
    } 
    return pair<vector<pair<int, vector<float>>>,map<string,int>> (enumfied_data,equivalence);
}

vector<pair<vector<int>, vector<float>>> Utils::shuffleDataset(const vector<pair<vector<int>, vector<float>>> &vec){
    vector<int> indexes(vec.size());
    iota (indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), Utils::RNG);
    vector<pair<vector<int>, vector<float>>> out;
    for(int i:indexes){
        out.push_back(vec[i]);
    }
    return out;
}

pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> Utils::divideDataSet(const vector<pair<vector<int>, vector<float>>> &vec, float percentageOfFirst){
    size_t fristSize=vec.size()*percentageOfFirst;
    return pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> (Utils::extractSubVector(vec, 0, fristSize),Utils::extractSubVector(vec, fristSize, vec.size()-fristSize));
}

vector<pair<vector<int>, vector<float>>> Utils::encodeDatasetLabelsUsingFirst(const vector<pair<vector<int>, vector<float>>> &vec, DataEncoder enc, int ext_max){
    vector<pair<int, vector<float>>> simplified;
    for(pair<vector<int>, vector<float>> entry:vec){
        simplified.push_back(pair<int, vector<float>>(entry.first[0],entry.second));
    }
    return Utils::encodeDatasetLabels(simplified,enc,ext_max);
}

vector<pair<vector<int>, vector<float>>> Utils::encodeDatasetLabels(const vector<pair<int, vector<float>>> &vec, DataEncoder enc, int ext_max){
    int max=numeric_limits<int>::min();
    if (ext_max>0){
        max=ext_max;
    }
    set<int> values;
    for (pair<int, vector<float>> entry:vec){
        values.insert(entry.first);
        if (entry.first > max){
            max = entry.first;
        }
    }

    int output_neurons=0;

    switch(enc){
        case DataEncoder::BINARY:
        case DataEncoder::BINARY_PLUS_ONE:
            output_neurons=ceil(log2(max+1));
            break;
        case DataEncoder::SPARSE:
        case DataEncoder::DISTINCT_SPARSE:
        case DataEncoder::DISTINCT_SPARSE_PLUS_ONE:
            output_neurons=max+1;
            break;
        case DataEncoder::INCREMENTAL:
        case DataEncoder::INCREMENTAL_PLUS_ONE:
        case DataEncoder::EXPONENTIAL:
            output_neurons=1;
            break;
    }

    map<int,vector<int>> equivalence;
    for (int l:values){
        vector<int> array;
        for (int i = 0; i < output_neurons; ++i) {
            switch(enc){
                case DataEncoder::BINARY:
                    array.push_back((l >> i) & 1);
                    break;
                case DataEncoder::SPARSE:
                    array.push_back((int) (i==l));
                    break;
                case DataEncoder::DISTINCT_SPARSE:
                    array.push_back((int) (i==l)*(l));
                    break;
                case DataEncoder::DISTINCT_SPARSE_PLUS_ONE:
                    array.push_back((int) (i==l)*(l+1));
                    break;
                case DataEncoder::INCREMENTAL:
                    array.push_back(l);
                    break;
                case DataEncoder::INCREMENTAL_PLUS_ONE:
                    array.push_back(l+1);
                    break;
                case DataEncoder::EXPONENTIAL:
                    array.push_back(pow(2,l+1));
                    break;
                case DataEncoder::BINARY_PLUS_ONE:
                    array.push_back(((l+1) >> i) & 1);
                    break;
            }
        }
        equivalence[l]=array;
    }
    vector<pair<vector<int>, vector<float>>> encoded_data;
    for (pair<int, vector<float>> entry:vec){
        encoded_data.push_back(pair<vector<int>, vector<float>>(equivalence[entry.first],entry.second));
    }
    return encoded_data;
}

pair<vector<pair<float,float>>,vector<pair<vector<int>, vector<float>>>> Utils::normalizeDataset(const vector<pair<vector<int>, vector<float>>> &vec){
    vector<pair<vector<int>, vector<float>>> normalized;
    vector<pair<float,float>> scale;
    vector<float> min;
    vector<float> max;
    for (auto& dummy:vec[0].second){
        (void) dummy;
        min.push_back(numeric_limits<float>::max());
        max.push_back(numeric_limits<float>::min());
    }
    for(pair<vector<int>, vector<float>> entry:vec){
        for (size_t i=0;i<entry.second.size();i++){
            if (entry.second[i]<min[i]){
                min[i]=entry.second[i];
            }
            if (entry.second[i]>max[i]){
                max[i]=entry.second[i];
            }
        }
    }
    for (size_t i=0;i<vec[0].second.size();i++){
        scale.push_back(pair<float,float>(min[i],(max[i]-min[i])));
    }
    
    for(pair<vector<int>, vector<float>> entry:vec){
        vector<float> normalized_entry_labels;
        for (size_t i=0;i<entry.second.size();i++){
            normalized_entry_labels.push_back( (entry.second[i]-scale[i].first)/scale[i].second );
        }
        normalized.push_back(pair<vector<int>, vector<float>>(entry.first,normalized_entry_labels));
    }
    return pair<vector<pair<float,float>>,vector<pair<vector<int>, vector<float>>>>(scale,normalized);
}

float Utils::getRandomBetweenZeroAndOne(){
    return dist_zero_one(Utils::RNG);
}

boost::uuids::uuid Utils::genRandomUUID(){
    return boost::uuids::random_generator()();
}

string Utils::genRandomUUIDStr(){
    return boost::uuids::to_string(Utils::genRandomUUID());
}

string Utils::msToHumanReadable(long timestamp){
    int D=int(timestamp/1000/60/60/24);
    int H=int(timestamp/1000/60/60%24);
    int M=int(timestamp/1000/60%60);
    int S=int(timestamp/1000%60);
    int MS=int(timestamp%1000);

    string out="";
    if (timestamp <= 0)
        out="0 ms";
    if (D > 0)
        out+=to_string(D)+" days ";
    if (D > 0 and MS == 0 and S == 0 and M == 0 and H > 0)
        out+="and ";
    if (H > 0)
        out+=to_string(H)+" hours ";
    if ((D > 0 or H > 0) and MS == 0 and S == 0 and M == 0)
        out+="and ";
    if (M > 0)
        out+=to_string(M)+" minutes ";
    if ((D > 0 or H > 0 or M > 0) and MS == 0 and S == 0)
        out+="and ";
    if (S > 0)
        out+=to_string(S)+" seconds ";
    if ((D > 0 or H > 0 or M > 0 or S > 0) and MS == 0)
        out+="and ";
    if (MS > 0)
        out+=to_string(MS)+" milliseconds ";
    return out;
}

snn_stats Utils::statisticalAnalysis(vector<pair<vector<int>, vector<float>>> correct, vector<vector<pair<int,float>>> pred){
    vector<vector<int>> c;
    for(pair<vector<int>, vector<float>> entry: correct){
        c.push_back(entry.first);
    }
    vector<vector<int>> p;
    for(vector<pair<int,float>> entry: pred){
        vector<int> p_tmp;
        for(pair<int,float> val: entry){
            p_tmp.push_back(val.first);
        }
        p.push_back(p_tmp);
    }
    return statisticalAnalysis(c,p);
}

snn_stats Utils::statisticalAnalysis(vector<vector<int>> correct, vector<vector<int>> pred){
    // if (correct.size()!=pred.size())
    //     throw runtime_error("Mismatching sizes on statisticalAnalysis! "+to_string(correct.size())+" - "+to_string(pred.size()));
    size_t size = pred.size();
    if(correct.size()<size){
        size=correct.size();
    }
    int total=0;
    int hits=0;
    int true_negative=0;
    int true_positive=0;
    int false_negative=0;
    int false_positive=0;
    int wrong=0;
    int pos_count=0;
    for (size_t i=0;i<size;i++){
        vector<int> cur_pred=pred[i];
        vector<int> cur_correct=correct[i];
        bool equal=true;
        bool positive=true;
        for(size_t j=0;j<cur_correct.size();j++){
            if(cur_pred[j]!=cur_correct[j]){
                equal=false;
            }
            if(cur_correct[j]==0){
                positive=false;
            }
        }
        total++;
        if(equal){
            hits++;
            if(positive){
                true_positive++;
                pos_count++;
            }else{
                true_negative++;
            }
        }else{
            wrong++;
            if(positive){
                false_negative++;
                pos_count++;
            }else{
                false_positive++;
            }
        }
    }
    snn_stats stats;
    stats.accuracy=(float)hits/total;
    if (correct[0].size()==1&&pos_count>0){
        stats.precision=(float)true_positive/(true_positive+false_positive);
        stats.recall=(float)true_positive/(true_positive+false_negative);
        stats.f1=2*(stats.precision*stats.recall)/(stats.precision+stats.recall);
    }
    return stats;
}

void Utils::printStats(snn_stats stats){
    cout<<"Stats:"<<"\taccuracy: "<<stats.accuracy<<"\n\tprecision: "<<stats.precision<<"\n\trecall: "<<stats.recall<<"\n\tf1: "<<stats.f1<<endl;
}

void Utils::copyFile(string source, string dst){
    if (Utils::checkIfFileExists(source))
        boost::filesystem::copy_file(source, dst, boost::filesystem::copy_option::overwrite_if_exists);
}

bool Utils::rmFile(string file){
    return boost::filesystem::remove_all(file)==0;
}

void Utils::serializeWeigths(map<string, vector<float>> weights, string filename,string except_str){
    if(weights.size()==0){
        return;
    }
    ofstream ofs (filename,ofstream::binary);
    if (ofs.is_open()){
        hps::to_stream(weights,ofs);
        ofs.close();
    }else{
        throw runtime_error("Could not serialize at: "+filename+"\n"+except_str);
    }
}

map<string, vector<float>> Utils::deserializeWeigths(string filename,string except_str){
    ifstream ifs (filename, ifstream::binary);
    map<string, vector<float>> weights;
    if(ifs.is_open()){
        weights=hps::from_stream<map<string, vector<float>>>(ifs);
    }
    // else{
    //     throw runtime_error("Could not serialize\n"+except_str);
    // }
    ifs.close();
    return weights;
}

bool Utils::mkdir(string path){
    boost::filesystem::create_directory(path);
    return true;
}

string Utils::getHostname(){
    int buff_size=150;
    string name="";
    #ifdef WIN32
        TCHAR infoBuf[buff_size];
        DWORD bufCharCount = buff_size;
        if(GetComputerName(infoBuf,&bufCharCount)){
            for(int i=0; i<buff_size; i++){
                if (infoBuf[i]=='\0')
                    break;
                name+=(char) infoBuf[i];
            }
        }else{
            name="Unknown_Host_Name";
        }
    #else
        char buff[buff_size];
        gethostname(buff, buff_size);
        name=string(buff);
    #endif
    return name;
}

string Utils::getStrNow(string format){
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),format.c_str(),timeinfo);
    string str(buffer);
    return str;
}

string Utils::serializeWeigthsToStr(map<string, vector<float>> weights){
    return hps::to_string(weights);
}

map<string, vector<float>> Utils::deserializeWeigthsFromStr(string serialized_str){
    return hps::from_string<map<string, vector<float>>>(serialized_str);
}

void Utils::compareAndPrintLabel(vector<vector<int>> correct, vector<vector<int>> pred){
    for(size_t i=0;i<pred.size();i++){
        vector<int> c=correct[i];
        vector<int> p=pred[i];
        cout<<"Reference label:";
        for(size_t j=0;j<c.size();j++){
            cout<<" "<<c[j];
        }
        cout<<" | Predicted label:";
        for(size_t j=0;j<p.size();j++){
            cout<<" "<<p[j];
        }
        cout<<endl;
    }
}

void Utils::compareAndPrintLabel(vector<pair<vector<int>, vector<float>>> correct, vector<vector<pair<int,float>>> pred){
    vector<vector<int>> c;
    for(pair<vector<int>, vector<float>> entry: correct){
        c.push_back(entry.first);
    }
    vector<vector<int>> p;
    for(vector<pair<int,float>> entry: pred){
        vector<int> p_tmp;
        for(pair<int,float> val: entry){
            p_tmp.push_back(val.first);
        }
        p.push_back(p_tmp);
    }
    Utils::compareAndPrintLabel(c,p);
}

string Utils::compressBase64(const string &in_64){
    string out_65;
    u_char last_c='&';
    int count=1;
    for (u_char c : in_64) {
        if (last_c=='&'){
            last_c=c;
            count=1;
        }else{
            if (last_c!=c){
                if (count>3){
                    out_65.push_back(last_c);
                    out_65.push_back('*');
                    out_65+=std::to_string(count);
                    out_65.push_back('*');
                }else{
                    string decompressed(count, last_c);
                    out_65+=decompressed;
                }
                count=0;
            }
            last_c=c;
            count++;
        }
    }
    if (count>3){
        out_65.push_back(last_c);
        out_65.push_back('*');
        out_65+=std::to_string(count);
        out_65.push_back('*');
    }else{
        string decompressed(count, last_c);
        out_65+=decompressed;
    }
    return out_65;
}

string Utils::decompressBase64(const string &in_65){
    string out_64;
    int amount=-666;
    u_char last_c=0;
    for (u_char c : in_65) {
        if (c=='*'){
            if (amount==-666){
                amount=-1;
            }else{
                string decompressed(amount-1, last_c);
                out_64+=decompressed;
                amount=-666;
            }
        }
        if (amount==-666){
            if (c!='*'){
                out_64.push_back(c);
                last_c=c;
            }
        }else{
            if (c!='*'){
                if (amount==-1){
                    amount=c-'0';
                }else{
                    amount*=10; 
                    amount+=c-'0';
                }
            }
        }
    }
    return out_64;
}

string Utils::base64ToString(const string &in_65){
    string out;
    string in=decompressBase64(in_65);
    vector<int> T(256,-1);
    for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val=0, valb=-8;
    for (u_char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val>>valb)&0xFF));
            valb -= 8;
        }
    }
    return out;
}

string Utils::stringToBase64(const string &in){
    string out;
    int val = 0, valb = -6;
    for (u_char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val>>valb)&0x3F]);
            valb -= 6;
        }
    }
    if (valb>-6) out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val<<8)>>(valb+8))&0x3F]);
    while (out.size()%4) out.push_back('=');
    return out;
}

string Utils::stringToBase65(const string &in){
    return compressBase64(stringToBase64(in));
}

vector<pair<vector<int>, vector<float>>> Utils::balanceSingleLabelDataset(const vector<pair<vector<int>, vector<float>>> &vec){
    vector<pair<vector<int>, vector<float>>> pos;
    vector<pair<vector<int>, vector<float>>> neg;
    for (pair<vector<int>, vector<float>> entry : vec){
        if(entry.first[0]==0){
            neg.push_back(entry);
        }else{
            pos.push_back(entry);
        }
    }
    if (pos.size()>neg.size()){
        pos=shuffleDataset(pos);
        pos.erase(pos.begin()+neg.size(),pos.end());
        pos.insert(pos.end(),neg.begin(),neg.end());
        pos=shuffleDataset(pos);
        return pos;
    } else if (pos.size()<neg.size()){
        neg=shuffleDataset(neg);
        neg.erase(neg.begin()+pos.size(),neg.end());
        neg.insert(neg.end(),pos.begin(),pos.end());
        neg=shuffleDataset(neg);
        return neg;
    }else{
        return vec;
    }
}

