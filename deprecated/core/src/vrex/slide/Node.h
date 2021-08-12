// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#pragma once

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <linux/mman.h>
#include <sys/mman.h>
#include <asm-generic/mman-common.h>
#include <random>
#include <math.h>
#include <time.h>
#include <chrono>
#include <algorithm>

enum class NodeType { ReLU, Softmax, Sigmoid };

#define Slide_HUGEPAGES 0 // 1 to use huge pages or 0 to not 

struct train_with_huge_pages {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;

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
            std::cout << "mmap failed at train." << std::endl;
        }
        return ptr;
    }
    void* operator new (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new (std::size_t size, void* ptr){return operator new (size);};
    void* operator new[] (std::size_t size){
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap fail! No train array!" << std::endl;
        }
        return ptr;
    }
    void* operator new[] (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new[] (std::size_t size, void* ptr){return operator new (size);};

    void operator delete(void * ptr){munmap(ptr, sizeof(train_with_huge_pages));};
    void operator delete (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(train_with_huge_pages));};
    void operator delete (void* ptr, void* voidptr2){};
    // TODO: The size to be munmap'd should be the entire array, not just a single object
    void operator delete[](void * ptr){munmap(ptr, sizeof(train_with_huge_pages));};
    void operator delete[] (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(train_with_huge_pages));};
    void operator delete[] (void* ptr, void* voidptr2){};

} __attribute__ ((aligned (64)));

struct train_without_huge_pages {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;
};

class Node;
#if Slide_HUGEPAGES == 1
    typedef train_with_huge_pages train;
    typedef Node* node_array;
#else
    typedef train_without_huge_pages train;
    typedef Node** node_array;
#endif

#include "../Slide.hpp"

using namespace std;

class Node
{
private:
	int _activeInputs;
    NodeType _type;
	bool use_adam;
    SlideLabelEncoding label_type;
	Node(){};

public:
	train* _train;
    int _currentBatchsize;
    size_t _dim, _layerNum, _IDinLayer;
	int* _indicesInTables;
	int* _indicesInBuckets;
	float* _weights;
	float* _mirrorWeights;
	float* _adamAvgMom;
	float* _adamAvgVel;
	float* _t; //for adam
	int* _update;
	float _bias =0;
	float _tbias = 0;
	float _adamAvgMombias=0;
	float _adamAvgVelbias=0;
	float _mirrorbias =0;
    #if Slide_HUGEPAGES == 1
        static const bool HUGEPAGES=true;
    #else
        static const bool HUGEPAGES=false;
    #endif

	Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, bool useAdamOt, float *weights, float bias, float *adamAvgMom, float *adamAvgVel);
	Node(int dim, int nodeID, int layerID, NodeType type,SlideLabelEncoding labelType, int batchsize, bool useAdamOt, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob);
	void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob);
	void updateWeights(float* newWeights, float newbias);
	float getLastActivation(int inputID);
	void incrementDelta(int inputID, float incrementValue);
	float getActivation(int* indices, float* values, int length, int inputID);
	bool getInputActive(int inputID);
	bool getActiveInputs(void);
	void SetlastActivation(int inputID, float realActivation);
	float ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize);
	float backPropagate(node_array previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
	float backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
    float calcBackPropagateGrad(int previousLayerActiveNodeSize, int inputID);
    static Node* createNodeArray(int size, bool useAdamOt,SlideLabelEncoding labelType);
	~Node();

    #if Slide_HUGEPAGES == 1
        void * operator new(size_t size){
            std::cout << "new Node" << std::endl;
            void* ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                -1, 0);
            if (ptr == MAP_FAILED){
                ptr = mmap(NULL, size,
                    PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                    -1, 0);
            }
            if (ptr == MAP_FAILED){
                std::cout << "mmap failed at Node." << std::endl;
            }
            return ptr;
        }
        void* operator new (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
        void* operator new (std::size_t size, void* ptr){return operator new (size);};
        void* operator new[] (std::size_t size){
            void* ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                -1, 0);
            if (ptr == MAP_FAILED){
                ptr = mmap(NULL, size,
                    PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                    -1, 0);
            }
            if (ptr == MAP_FAILED){
                std::cout << "mmap failed at Node array." << std::endl;
            }
            return ptr;
        }
        void* operator new[] (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
        void* operator new[] (std::size_t size, void* ptr){return operator new (size);};

        void operator delete(void * ptr){munmap(ptr, sizeof(Node));};
        void operator delete (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(Node));};
        void operator delete (void* ptr, void* voidptr2){};
        // TODO: should munmap the size of the entire array, not a single Node
        void operator delete[](void * ptr){munmap(ptr, sizeof(Node));};
        void operator delete[] (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(Node));};
        void operator delete[] (void* ptr, void* voidptr2){};
    #endif

	//only for debugging
	float purturbWeight(int weightid, float delta);
	float getGradient(int weightid, int inputID, float InputVal);
};
