// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#include "Bucket.h"
#include "../Slide.hpp"

Bucket::Bucket()
{
    isInit = -1;
    #if Slide_USE_SMART_POINTERS == 1
        arr = shared_ptr<int[]>(new int[Slide::BUCKET_SIZE]);
    #else
        arr = new int[Slide::BUCKET_SIZE]();
    #endif
}


Bucket::~Bucket()
{
    #if Slide_USE_SMART_POINTERS == 0
        delete[] arr;
    #endif
}


int Bucket::getTotalCounts()
{
    return _counts;
}


int Bucket::getSize()
{
    return _counts;
}


int Bucket::add(int id) {

    //FIFO
    if (Slide::FIFO_INSTEAD_OF_RESERVOIR_SAMPLING) {
        isInit += 1;
        index = _counts & (Slide::BUCKET_SIZE - 1);
        _counts++;
        arr[index] = id;
        return index;
    }
    //Reservoir Sampling
    else {
        _counts++;
        if (index == Slide::BUCKET_SIZE) {
            int randnum = rand() % (_counts) + 1;
            if (randnum == 2) {
                int randind = rand() % Slide::BUCKET_SIZE;
                arr[randind] = id;
                return randind;
            } else {
                return -1;
            }
        } else {
            arr[index] = id;
            int returnIndex = index;
            index++;
            return returnIndex;
        }
    }
}


int Bucket::retrieve(int indice)
{
    if (indice >= Slide::BUCKET_SIZE)
        return -1;
    return arr[indice];
}



int_array_pointer Bucket::getAll(){
    if (isInit == -1)
        return nullptr;
    if(_counts<Slide::BUCKET_SIZE){
        arr[_counts]=-1;
    }
    return arr;
}
