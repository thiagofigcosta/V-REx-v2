// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#pragma once

#include <sys/mman.h>
#include <map>
#include <bitset>
#include <fstream>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <climits>
#include <limits>

class Slide;
class LSH;
class Node;

#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"

using namespace std;

#include "../GarbageCollector.hpp"

class Layer
{
private:
	NodeType _type;
	node_array _Nodes;
	int * _randNode;
	float* _normalizationConstants;
    int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;
    train* _train_array;
	SlideMode mode;
	SlideHashingFunction hash_func;
	bool use_adam;
	SlideLabelEncoding label_type;

public:
	int _layerID, _noOfActive;
	size_t size_2d;
    size_t size_1d;
	size_t _noOfNodes;
	float* _weights;
	float* _adamAvgMom;
	float* _adamAvgVel;
	float* _bias;
	LSH *_hashTables;
	WtaHash *_wtaHasher;
    DensifiedMinhash *_MinHasher;
    SparseRandomProjection *_srp;
    DensifiedWtaHash *_dwtaHasher;
	int * _binids;
	Layer(size_t noOfNodes, int previousLayerNumOfNodes, size_t maxNodes,int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity,SlideMode Mode,SlideHashingFunction hashFunc, bool useAdamOt,SlideLabelEncoding labelType, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
	Node* getNodebyID(size_t nodeID);
	node_array getAllNodes();
	int getNodeCount();
	void addtoHashTable(float* weights, int length, float bias, int id);
	float getNomalizationConstant(int inputID);
	int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int queryActiveNodes(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeSoftmax(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
	map<string, vector<float>> mapfyWeights();
	void updateTable();
	void updateRandomNodes();
	void flushTable();

	~Layer();
	#if Slide_HUGEPAGES == 1
		void * operator new(size_t size){
			void* ptr = mmap(NULL, size,
				PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
				-1, 0);
			if (ptr == MAP_FAILED){
				ptr = mmap(NULL, size,
					PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
					-1, 0);
			}
			if (ptr == MAP_FAILED)
				std::cout << "mmap fail! No new layer!" << std::endl;
			return ptr;};
		void operator delete(void * pointer){munmap(pointer, sizeof(Layer));};
	#endif
};
