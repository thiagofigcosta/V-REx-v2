// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#pragma once

#include <climits>
#include <chrono>
#include <unordered_map>
#include <iostream>
#include <random>
#include <memory>
#include <vector>

using namespace std;
class Bucket;

#include "../Slide.hpp"

class LSH {
private:
	bucket_pointer_2d _bucket;
	int _K;
	int _L;
	int _RangePow;
	int *rand1;
	SlideHashingFunction hash_func;


public:
	LSH(int K, int L, int RangePow,SlideHashingFunction hashFunc);
	void clear();
	int* add(int *indices, int id);
	int add(int indices, int tableId, int id);
	int * hashesToIndex(int * hashes);
	int_array_pointer_2d retrieveRaw(int *indices);
	int retrieve(int table, int indices, int bucket);
	void count();
	~LSH();
};
