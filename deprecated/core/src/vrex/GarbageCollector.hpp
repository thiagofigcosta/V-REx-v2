#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

using namespace std;

#include "Utils.hpp"
#include "NeuralGenome.hpp"

class GarbageCollector{
    private:
        // variables
        static GarbageCollector* singleton;
        #pragma omp threadprivate(singleton)
        vector<int*> int_1_pointers;
        vector<float*> float_1_pointers;
        vector<int**> int_2_pointers;
        vector<float**> float_2_pointers;

        vector<int*> int_1_cleaned;
        vector<float*> float_1_cleaned;
        vector<int**> int_2_cleaned;
        vector<float**> float_2_cleaned;

    public:
        // constructors and destructor
        GarbageCollector(){}

        // methods
        static GarbageCollector* get(){
            if(singleton==nullptr)
                singleton = new GarbageCollector();
            return singleton;
        }
        static void erase(){
            if(singleton!=nullptr){
                singleton->flush();
                delete singleton;
                singleton=nullptr;
            }
        }
    
        void addInt1d(int* obj);
        void addFloat1d(float* obj);
        void addInt2d(int** obj);
        void addFloat2d(float** obj);
        void flush();
        void rmInt1d(int* obj);
        void rmFloat1d(float* obj);
        void rmInt2d(int** obj);
        void rmFloat2d(float** obj);
};