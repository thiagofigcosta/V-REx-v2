#include "GarbageCollector.hpp"
#include "Utils.hpp"

GarbageCollector* GarbageCollector::singleton=nullptr;

void GarbageCollector::addInt1d(int* obj){
    int_1_pointers.push_back(obj);
}

void GarbageCollector::addFloat1d(float* obj){
    float_1_pointers.push_back(obj);
}

void GarbageCollector::addInt2d(int** obj){
    int_2_pointers.push_back(obj);
}

void GarbageCollector::addFloat2d(float** obj){
    float_2_pointers.push_back(obj);
}

void GarbageCollector::rmInt1d(int* obj){
    int_1_cleaned.push_back(obj);
}

void GarbageCollector::rmFloat1d(float* obj){
    float_1_cleaned.push_back(obj);
}

void GarbageCollector::rmInt2d(int** obj){
    int_2_cleaned.push_back(obj);
}

void GarbageCollector::rmFloat2d(float** obj){
    float_2_cleaned.push_back(obj);
}

void GarbageCollector::flush(){
    int_1_pointers=Utils::subtractVectors(int_1_pointers, int_1_cleaned);
    int_1_cleaned.clear();
    float_1_pointers=Utils::subtractVectors(float_1_pointers, float_1_cleaned);
    float_1_cleaned.clear();
    int_2_pointers=Utils::subtractVectors(int_2_pointers, int_2_cleaned);
    int_2_cleaned.clear();
    float_2_pointers=Utils::subtractVectors(float_2_pointers, float_2_cleaned);
    float_2_cleaned.clear();

    while(int_1_pointers.size()>0){
        delete[] int_1_pointers.back();
        int_1_pointers.pop_back();
    }
    while(float_1_pointers.size()>0){
        delete[] float_1_pointers.back();
        float_1_pointers.pop_back();
    }
    while(int_2_pointers.size()>0){
        delete[] int_2_pointers.back();
        int_2_pointers.pop_back();
    }
    while(float_2_pointers.size()>0){
        delete[] float_2_pointers.back();
        float_2_pointers.pop_back();
    }
}