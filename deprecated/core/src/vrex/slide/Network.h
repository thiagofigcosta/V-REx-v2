// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#pragma once

#include <chrono>
#include <sys/mman.h>
#include <map>
#include <omp.h>
#include <iostream>
#include <math.h>
#include <algorithm>

class Layer;

#include "Layer.h"
#include "Node.h"
#include "LSH.h"
#include "../Slide.hpp"
#include "../GarbageCollector.hpp"

using namespace std;

class Network
{
private:
	Layer** _hiddenlayers;
	float _learningRate;
	int _numberOfLayers;
	int* _sizesOfLayers;
	NodeType* _layersTypes;
	float * _Sparsity;
	//int* _inputIDs;
	int  _currentBatchSize;
	int _inputDim;
	int _outputDim;
	int * _K;
	int * _L;
	int * _RangePow;
	SlideHashingFunction hash_func;
	SlideMode mode;
	bool use_adam;
	SlideLabelEncoding label_type;
	bool init;
	size_t size_max_for_layer;
	float **_weight, **_bias, **_adamAvgMom, **_adamAvgVel; 

public:
	Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int outputdim, int* K, int* L, int* RangePow, float* Sparsity,SlideMode Mode,SlideHashingFunction hashFunc, bool useAdamOt,SlideLabelEncoding labelType, size_t maxLayerS);
	void setWeights(map<string, vector<float>> loadedData);
	Layer* getLayer(int LayerID);
	pair<int,vector<vector<pair<int,float>>>> predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
	float ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** labels, int *labelsize, int iter, bool rehash, bool rebuild);
	pair<float,vector<vector<pair<int,float>>>> ProcessInputAndRetrieveClass(int** inputIndices, float** inputValues, int* lengths, int ** labels, int *labelsize, int iter, bool rehash, bool rebuild);
	float evalInput(int** inputIndices, float** inputValues, int* lengths, int ** labels, int *labelsize);
	map<string, vector<float>> mapfyWeights();
	void lateInit();
	void flushLayers();
	~Network();
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
			if (ptr == MAP_FAILED){
				std::cout << "mmap failed at Network." << std::endl;
			}
			return ptr;
		}
		void operator delete(void * pointer){munmap(pointer, sizeof(Network));};
	#endif
};

