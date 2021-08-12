// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#include "Layer.h"
#include "Node.h"
#include "../Slide.hpp"

using namespace std;


Layer::Layer(size_t noOfNodes, int previousLayerNumOfNodes, size_t maxNodes, int layerID, NodeType type, int batchsize,  int K, int L, int RangePow, float Sparsity,SlideMode Mode, SlideHashingFunction hashFunc, bool useAdamOt,SlideLabelEncoding labelType, float* weights, float* bias, float *adamAvgMom, float *adamAvgVel) {
    _layerID = layerID;
    _noOfNodes = noOfNodes;
    if (maxNodes==numeric_limits<size_t>::max()){
        size_2d=_noOfNodes * previousLayerNumOfNodes;
        size_1d=_noOfNodes;
    }else{
        if (layerID!=0){
            size_2d=maxNodes*maxNodes;
        }else{
            size_2d=maxNodes*previousLayerNumOfNodes;
        }
        size_1d=maxNodes;
    }
    use_adam=useAdamOt;
    label_type=labelType;
    #if Slide_HUGEPAGES == 1
        _Nodes = Node::createNodeArray(_noOfNodes,use_adam,label_type);
    #else
        _Nodes = new Node *[noOfNodes];
    #endif
    _type = type;
    _noOfActive = floor(_noOfNodes * Sparsity);
    _K = K;
    _L = L;
    _batchsize = batchsize;
    _RangeRow = RangePow;
    _previousLayerNumOfNodes = previousLayerNumOfNodes;
    hash_func=hashFunc;
    mode=Mode;

// create a list of random nodes just in case not enough nodes from hashtable for active nodes.
    _randNode = new int[_noOfNodes];
    for (size_t n = 0; n < _noOfNodes; n++) {
        _randNode[n] = n;
    }

    std::random_shuffle(_randNode, _randNode + _noOfNodes);

//TODO: Initialize Hash Tables and add the nodes. Done by Beidi
    _hashTables = new LSH(_K, _L, RangePow,hash_func);

    if (hash_func == SlideHashingFunction::WTA) {
        _wtaHasher = new WtaHash(_K * _L, previousLayerNumOfNodes);
    } else if (hash_func == SlideHashingFunction::DENSIFIED_WTA) {
        _binids = new int[previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, previousLayerNumOfNodes);
    } else if (hash_func == SlideHashingFunction::TOPK_MIN_HASH) {
        _binids = new int[previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, previousLayerNumOfNodes);
        _MinHasher->getMap(previousLayerNumOfNodes, _binids);
    } else if (hash_func == SlideHashingFunction::SIMHASH) {
        _srp = new SparseRandomProjection(previousLayerNumOfNodes, _K * _L, Slide::SIMHASH_RATIO);
    }

    if ((weights != nullptr && bias != nullptr) || (weights != NULL && bias != NULL) ) {
        _weights = weights;
        _bias = bias;

        if (use_adam){
            _adamAvgMom = adamAvgMom;
            _adamAvgVel = adamAvgVel;
        }

    }else{
        _weights = new float[size_2d](); 
        _bias = new float[size_1d]; 
        random_device rd;
        default_random_engine dre(rd());
        normal_distribution<float> distribution(Slide::RAND_WEIGHT_START, Slide::RAND_WEIGHT_END);

        generate(_weights, _weights + size_2d, [&] () { return distribution(dre); }); // 
        generate(_bias, _bias + size_1d, [&] () { return distribution(dre); });  // 


        if (use_adam)
        {
            _adamAvgMom = new float[size_2d]();  
            _adamAvgVel = new float[size_2d]();

        }
    }

    _train_array = new train[_noOfNodes*_batchsize];

    // create nodes for this layer
#pragma omp parallel for
    for (size_t i = 0; i < _noOfNodes; i++)
    {
        #if Slide_HUGEPAGES == 1
            _Nodes[i].Update(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i, _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i, _train_array);
            addtoHashTable(_Nodes[i]._weights, previousLayerNumOfNodes, _Nodes[i]._bias, i);
        #else
            _Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, type,label_type, batchsize, use_adam, _weights+previousLayerNumOfNodes*i, _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i, _train_array);
            // _Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i, _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i);
             addtoHashTable(_Nodes[i]->_weights, previousLayerNumOfNodes, _Nodes[i]->_bias, i);
        #endif
    }

    if (type == NodeType::Softmax)
    {
        _normalizationConstants = new float[batchsize]();
    }
}


void Layer::updateTable()
{

    if (hash_func == SlideHashingFunction::WTA) {
         delete _wtaHasher;
        _wtaHasher = new WtaHash(_K * _L, _previousLayerNumOfNodes);
    } else if (hash_func == SlideHashingFunction::DENSIFIED_WTA) {
        delete _dwtaHasher;
        delete _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, _previousLayerNumOfNodes);
    } else if (hash_func == SlideHashingFunction::TOPK_MIN_HASH) {
        delete _MinHasher;
        delete  _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, _previousLayerNumOfNodes);
        _MinHasher->getMap(_previousLayerNumOfNodes, _binids);
    } else if (hash_func == SlideHashingFunction::SIMHASH) {
        delete _srp;
        _srp = new SparseRandomProjection(_previousLayerNumOfNodes, _K * _L, Slide::SIMHASH_RATIO);

    }
}


void Layer::updateRandomNodes()
{
    std::random_shuffle(_randNode, _randNode + _noOfNodes);
}


void Layer::addtoHashTable(float* weights, int length, float bias, int ID)
{
    //LSH logic
    int *hashes;
    if(hash_func==SlideHashingFunction::WTA) {
        hashes = _wtaHasher->getHash(weights);
    }else if (hash_func==SlideHashingFunction::DENSIFIED_WTA) {
        hashes = _dwtaHasher->getHashEasy(weights, length, Slide::TOPK_HASH_TOPK);
    }else if (hash_func== SlideHashingFunction::TOPK_MIN_HASH) {
        hashes = _MinHasher->getHashEasy(_binids, weights, length, Slide::TOPK_HASH_TOPK);
    }else if (hash_func==SlideHashingFunction::SIMHASH) {
        hashes = _srp->getHash(weights, length);
    }

    int * hashIndices = _hashTables->hashesToIndex(hashes);
    int * bucketIndices = _hashTables->add(hashIndices, ID+1);

    #if Slide_HUGEPAGES == 1
        _Nodes[ID]._indicesInTables = hashIndices;
        _Nodes[ID]._indicesInBuckets = bucketIndices;
    #else
        _Nodes[ID]->_indicesInTables = hashIndices;
        _Nodes[ID]->_indicesInBuckets = bucketIndices;
    #endif

    #pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    delete [] hashes;
    #pragma GCC diagnostic pop 

}


Node* Layer::getNodebyID(size_t nodeID)
{
    #pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
    assert(("nodeID less than _noOfNodes" , nodeID < _noOfNodes));
    #pragma GCC diagnostic pop
    #if Slide_HUGEPAGES == 1
        return &_Nodes[nodeID];
    #else
        return _Nodes[nodeID];
    #endif
}


node_array Layer::getAllNodes()
{
    return _Nodes;
}

int Layer::getNodeCount()
{
    return _noOfNodes;
}

float Layer::getNomalizationConstant(int inputID)
{
    #pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
    assert(("Error Call to Normalization Constant for non - softmax layer", _type == NodeType::Softmax));
    #pragma GCC diagnostic pop 
    return _normalizationConstants[inputID];
}


float innerproduct(int* index1, float* value1, int len1, float* value2){
    float total = 0;
    for (int i=0; i<len1; i++){
        total+=value1[i]*value2[index1[i]];
    }
    return total;
}


float collision(int* hashes, int* table_hashes, int k, int l){
    int cp = 0;
    for (int i=0; i<l; i=i+k){
        int tmp = 0;
        for (int j=i; j< i+k;j++){
            if(hashes[j]==table_hashes[j]){
                tmp++;
            }
        }
        if (tmp==k){
            cp++;
        }
    }
    return cp*1.0/(l/k);
}


int Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int* label, int labelsize, float Sparsity, int iter)
{
    //LSH QueryLogic

    //Beidi. Query out all the candidate nodes
    #pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    int len;
    #pragma GCC diagnostic pop 
    int in = 0;

    if(Sparsity == 1.0){
        len = _noOfNodes;
        lengths[layerIndex + 1] = len;
        activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
        GarbageCollector::get()->addInt1d(activenodesperlayer[layerIndex + 1]);
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[layerIndex + 1][i] = i;
        }
    }
    else
    {
        if (mode==SlideMode::TOPK_THRESHOLD) {
            int *hashes;
            if (hash_func == SlideHashingFunction::WTA) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (hash_func == SlideHashingFunction::DENSIFIED_WTA) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (hash_func == SlideHashingFunction::TOPK_MIN_HASH) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], Slide::TOPK_HASH_TOPK);
            } else if (hash_func == SlideHashingFunction::SIMHASH) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int_array_pointer_2d actives = _hashTables->retrieveRaw(hashIndices);
            

            // Get candidates from hashtable
            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }

            for (int i = 0; i < _L; i++) {
                if (actives[i] == nullptr) {
                    continue;
                } else {
                    for (int j = 0; j < Slide::BUCKET_SIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            //thresholding
            vector<int> vect;
            for (auto &&x : counts){
                if (x.second>(unsigned int)Slide::TOPK_THRESHOLD_SECONDS){
                    vect.push_back(x.first);
                }
            }

            len = vect.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = vect[i];
            }
            in = len;

            #pragma GCC diagnostic push 
            #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            delete[] hashes;
            #pragma GCC diagnostic pop 
            delete[] hashIndices;
            #if Slide_USE_SMART_POINTERS == 0
                delete[] actives;
            #endif
        }
        if (mode==SlideMode::SAMPLING) {
            int *hashes;
            if (hash_func == SlideHashingFunction::WTA) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (hash_func == SlideHashingFunction::DENSIFIED_WTA) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (hash_func == SlideHashingFunction::TOPK_MIN_HASH) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], Slide::TOPK_HASH_TOPK);
            } else if (hash_func == SlideHashingFunction::SIMHASH) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int_array_pointer_2d actives = _hashTables->retrieveRaw(hashIndices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashtable
            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax && labelsize > 0) {
                for (int i = 0; i < labelsize ;i++){
                    counts[label[i]] = _L;
                }
            }

            for (int i = 0; i < _L; i++) {
                if (actives[i] == nullptr) {
                    continue;
                } else {
                    // copy sparse array into (dense) map
                    for (int j = 0; j < Slide::BUCKET_SIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            in = counts.size();
            if (counts.size()<1500){
                srand(time(NULL));
                size_t start = rand() % _noOfNodes;
                for (size_t i = start; i < _noOfNodes; i++) {
                    if (counts.size() >= 1000) {
                        break;
                    }
                    if (counts.count(_randNode[i]) == 0) {
                        counts[_randNode[i]] = 0;
                    }
                }

                if (counts.size() < 1000) {
                    for (size_t i = 0; i < _noOfNodes; i++) {
                        if (counts.size() >= 1000) {
                            break;
                        }
                        if (counts.count(_randNode[i]) == 0) {
                            counts[_randNode[i]] = 0;
                        }
                    }
                }
            }

            len = counts.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            // copy map into new array
            int i=0;
            for (auto &&x : counts) {
                activenodesperlayer[layerIndex + 1][i] = x.first;
                i++;
            }

            #pragma GCC diagnostic push 
            #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            delete[] hashes;
            #pragma GCC diagnostic pop 
            delete[] hashIndices;
             #if Slide_USE_SMART_POINTERS == 0
                delete[] actives;
            #endif

        }
        else if ((mode == SlideMode::UNKNOWN_MODE1) & (_type== NodeType::Softmax)) {
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            bitset <Slide_MAPLEN> bs; // boost/dynamic_bitset can be slower than bitset
            int tmpsize = 0;
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        activenodesperlayer[layerIndex + 1][i] = label[i];
                        bs[label[i]] = 1;
                    }
                    tmpsize = labelsize;
                }
            }

            while(tmpsize<len){
                int v = rand() % _noOfNodes;
                if(bs[v]==0) {
                    activenodesperlayer[layerIndex + 1][tmpsize] = v;
                    bs[v]=1;
                    tmpsize++;
                }
            }
        }else if ((mode == SlideMode::UNKNOWN_MODE2) & (_type== NodeType::Softmax)){
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];
            vector<pair<float, int> > sortW;
            int what = 0;

            for (size_t s = 0; s < _noOfNodes; s++) {
                #if Slide_HUGEPAGES == 1
                    float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                            lengths[layerIndex], _Nodes[s]._weights);
                    tmp += _Nodes[s]._bias;
                #else
                    float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                            lengths[layerIndex], _Nodes[s]->_weights);
                    tmp += _Nodes[s]->_bias;
                #endif
                if (find(label, label + labelsize, s) != label + labelsize) {
                    sortW.push_back(make_pair(-1000000000, s));
                    what++;
                }
                else{
                    sortW.push_back(make_pair(-tmp, s));
                }
            }

            std::sort(begin(sortW), end(sortW));

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = sortW[i].second;
                if (find (label, label+labelsize, sortW[i].second)!= label+labelsize){
                    in=1;
                }
            }
        }
    }

    //***********************************
    #pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
    GarbageCollector::get()->addFloat1d(activeValuesperlayer[layerIndex + 1]);
    #pragma GCC diagnostic pop 
    float maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    // find activation for all ACTIVE nodes in layer
    for (int i = 0; i < len; i++)
    {
        #if Slide_HUGEPAGES == 1
            activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]].getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        #else
            activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]]->getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        #endif
        if(_type == NodeType::Softmax && activeValuesperlayer[layerIndex + 1][i] > maxValue){
            maxValue = activeValuesperlayer[layerIndex + 1][i];
        }
    }

    if(_type == NodeType::Softmax) {
        for (int i = 0; i < len; i++) {
            float realActivation;
            if (_noOfNodes>1){
                realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            }else{
                realActivation = activeValuesperlayer[layerIndex + 1][i];
            }
            activeValuesperlayer[layerIndex + 1][i] = realActivation;
            #if Slide_HUGEPAGES == 1
                _Nodes[activenodesperlayer[layerIndex + 1][i]].SetlastActivation(inputID, realActivation);
            #else
                _Nodes[activenodesperlayer[layerIndex + 1][i]]->SetlastActivation(inputID, realActivation);
            #endif
            _normalizationConstants[inputID] += realActivation;
        }
    }

    return in;
}

map<string, vector<float>> Layer::mapfyWeights()
{
    map<string, vector<float>> arr;
    if (_layerID==0) {
        arr["w_layer_0"]=vector<float>(_weights, _weights + size_2d);
        arr["b_layer_0"]=vector<float>(_bias, _bias + size_1d);
        if (use_adam){
            arr["am_layer_0"]=vector<float>(_adamAvgMom, _adamAvgMom + size_2d);
            arr["av_layer_0"]=vector<float>(_adamAvgVel, _adamAvgVel + size_2d);
        }
    }else{
        arr["w_layer_"+ to_string(_layerID)]=vector<float>(_weights, _weights + size_2d);
        arr["b_layer_"+ to_string(_layerID)]=vector<float>(_bias, _bias + size_1d);
        if (use_adam){
            arr["am_layer_"+ to_string(_layerID)]=vector<float>(_adamAvgMom, _adamAvgMom + size_2d);
            arr["av_layer_"+ to_string(_layerID)]=vector<float>(_adamAvgVel, _adamAvgVel + size_2d);
        }
    }
    // cout<<"save for layer "<<to_string(_layerID)<<endl;
    return arr;
}

void Layer::flushTable(){
    _hashTables->clear();
}

Layer::~Layer()
{
    delete _hashTables;
    if (_type == NodeType::Softmax){
        delete[] _normalizationConstants;
        delete[] _adamAvgMom;
        delete[] _adamAvgVel;
    }
    #if Slide_HUGEPAGES == 1
        for (size_t d=0;d<_noOfNodes;d++){
            delete &_Nodes[d];
        }
    #else
        for (size_t d=0;d<_noOfNodes;d++){
            delete _Nodes[d];
        }
        delete [] _Nodes;
    #endif
    delete [] _weights;
    delete [] _bias;
    delete [] _binids;
    if (hash_func == SlideHashingFunction::WTA) {
        delete _wtaHasher;
    } else if (hash_func == SlideHashingFunction::DENSIFIED_WTA) {
        delete _dwtaHasher;
    } else if (hash_func == SlideHashingFunction::TOPK_MIN_HASH) {
        delete _MinHasher;
    } else if (hash_func == SlideHashingFunction::SIMHASH) {
        delete _srp;
    }
    delete [] _randNode;
    if (Node::HUGEPAGES){
        for (size_t d=0;d<_noOfNodes*_batchsize;d++){
            delete &_train_array[d];
        }
    }
    delete[] _train_array;
}
