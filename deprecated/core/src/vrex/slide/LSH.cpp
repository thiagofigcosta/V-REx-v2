// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#include "LSH.h"
#include "Bucket.h"

using namespace std;

LSH::LSH(int K, int L, int RangePow,SlideHashingFunction hashFunc)
{
	_K = K;
	_L = L;
	_RangePow = RangePow;
	#if Slide_USE_SMART_POINTERS == 1
		_bucket=vector<vector<shared_ptr<Bucket>>>(_L);
	#else
		_bucket = new Bucket*[L];
	#endif
	hash_func=hashFunc;

//#pragma omp parallel for
	for (int i = 0; i < _L; i++){
		#if Slide_USE_SMART_POINTERS == 1
			_bucket[i] = vector<shared_ptr<Bucket>>(1 << _RangePow);
			for(size_t z=0;z<_bucket[i].size();z++){
				_bucket[i][z]=make_shared<Bucket>();
			}
		#else
			_bucket[i] = new Bucket[1 << _RangePow];
		#endif
	}

	rand1 = new int[_K*_L];

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, INT_MAX);

//#pragma omp parallel for
	for (int i = 0; i < _K*_L; i++)
	{
		rand1[i] = dis(gen);
		if (rand1[i] % 2 == 0)
			rand1[i]++;
	}
}


void LSH::clear()
{
    for (int i = 0; i < _L; i++){
		#if Slide_USE_SMART_POINTERS == 1
			_bucket[i].clear();
			_bucket[i] = vector<shared_ptr<Bucket>>(1 << _RangePow);
    		for(size_t z=0;z<_bucket[i].size();z++){
				_bucket[i][z]=make_shared<Bucket>();
			}
		#else
			delete[] _bucket[i];
			_bucket[i] = new Bucket[1 << _RangePow];
		#endif
		

    }
}


void LSH::count()
{
	for (int j=0; j<_L;j++) {
		int total = 0;
		for (int i = 0; i < 1 << _RangePow; i++) {
			#if Slide_USE_SMART_POINTERS == 1
				if (_bucket[j][i]->getSize()!=0) {
					cout <<_bucket[j][i]->getSize() << " ";
				}
				total += _bucket[j][i]->getSize();
			#else
				if (_bucket[j][i].getSize()!=0) {
					cout <<_bucket[j][i].getSize() << " ";
				}
				total += _bucket[j][i].getSize();
			#endif
		}
		cout << endl;
		cout <<"TABLE "<< j << "Total "<< total << endl;
	}
}


int* LSH::hashesToIndex(int * hashes)
{

	int * indices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		unsigned int index = 0;

		for (int j = 0; j < _K; j++)
		{

			if (hash_func==SlideHashingFunction::SIMHASH){
				unsigned int h = hashes[_K*i + j];
				index += h<<(_K-1-j);
			}else if ((hash_func==SlideHashingFunction::WTA) | (hash_func==SlideHashingFunction::DENSIFIED_WTA)){
                unsigned int h = hashes[_K*i + j];
                index += h<<((_K-1-j)*(int)floor(log(Slide::WTA_BIN_SIZE)));

            }else {
                unsigned int h = rand1[_K*i + j];
                h *= rand1[_K * i + j];
                h ^= h >> 13;
                h ^= rand1[_K * i + j];
                index += h * hashes[_K * i + j];
            }
		}
		if (hash_func==SlideHashingFunction::TOPK_MIN_HASH) {
			index = index&((1<<_RangePow)-1);
		}
		indices[i] = index;
	}

	return indices;
}


int* LSH::add(int *indices, int id)
{
	int * secondIndices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		#if Slide_USE_SMART_POINTERS == 1
			if(!_bucket[i][indices[i]]){
				cout<<"nullref\n";
				_bucket[i][indices[i]]=make_shared<Bucket>();
			}
			secondIndices[i] = _bucket[i][indices[i]]->add(id);
		#else
			secondIndices[i] = _bucket[i][indices[i]].add(id);
		#endif
	}

	return secondIndices;
}


int LSH::add(int tableId, int indices, int id)
{
	#if Slide_USE_SMART_POINTERS == 1
		int secondIndices = _bucket[tableId][indices]->add(id);
	#else
		int secondIndices = _bucket[tableId][indices].add(id);
	#endif
	return secondIndices;
}


/*
* Returns all the buckets
*/

int_array_pointer_2d LSH::retrieveRaw(int *indices){
	int_array_pointer_2d rawResults;
	#if Slide_USE_SMART_POINTERS == 1
		rawResults = vector<shared_ptr<int[]>>(_L);
	#else
		rawResults = new int*[_L];
	#endif

	for (int i = 0; i < _L; i++)
	{
		#if Slide_USE_SMART_POINTERS == 1
			rawResults[i] = _bucket[i][indices[i]]->getAll();
		#else
			rawResults[i] = _bucket[i][indices[i]].getAll();
		#endif
	}
	return rawResults;
}


int LSH::retrieve(int table, int indices, int bucket)
{
	#if Slide_USE_SMART_POINTERS == 1
		return _bucket[table][indices]->retrieve(bucket);
	#else
		return _bucket[table][indices].retrieve(bucket);
	#endif
}

LSH::~LSH(){
	delete [] rand1;
	#if Slide_USE_SMART_POINTERS == 1
		for (int i = 0; i < _L; i++){
			_bucket[i].clear();
		}
		_bucket.clear();
	#else
		for (int i = 0; i < _L; i++){
			delete[] _bucket[i];
		}
		delete[] _bucket;
	#endif	
}
