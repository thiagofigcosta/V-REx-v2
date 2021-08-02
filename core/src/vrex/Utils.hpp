#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <algorithm> 
#include <cctype>
#include <locale>
#include <sys/stat.h>
#include <stdexcept>
#include <stdio.h>
#include <set>
#include <random>
#include <map>
#include <limits>
#include <cmath>
#include <memory>
#include <boost/uuid/uuid.hpp>            
#include <boost/uuid/uuid_generators.hpp> 
#include <boost/uuid/uuid_io.hpp>
#include <boost/filesystem.hpp>
#ifdef WIN32
    #include <Windows.h>
    #include <tchar.h>
#else
    #include <unistd.h>
#endif
#include <ctime>

#include "hps/hps.h"

using namespace std;

enum class DataEncoder { BINARY, SPARSE, INCREMENTAL, INCREMENTAL_PLUS_ONE, EXPONENTIAL, DISTINCT_SPARSE, DISTINCT_SPARSE_PLUS_ONE, BINARY_PLUS_ONE };

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args){ // this is from C++14
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

typedef std::basic_string<char> string;
typedef struct snn_stats{
  float accuracy=-1;
  float precision=-1;
  float recall=-1;
  float f1=-1;
} snn_stats;

class Utils{
    public:
        // constructors and destructor
        Utils();
        Utils(const Utils& orig);
        virtual ~Utils();

        // methods
        static vector<string> splitString(string str, string token);
        static void copyFile(string source, string dst);
        static bool rmFile(string file);
        static vector<pair<string, vector<float>>> readLabeledCsvDataset(const string filename);
        static string getResourcesFolder();
        static string getResourcePath(string filename);
        static string joinPath(string base,string sufix);
        static void ltrimInPlace(string &s);
        static void rtrimInPlace(string &s);
        static void trimInPlace(string &s);
        static string ltrim(string s);
        static string rtrim(string s);
        static string trim(string s);
        static bool endsWith(const string &str, const string &ending);
        static bool runningOnDockerContainer();
        static bool checkIfFileExists(const string &path);
        static bool checkIfIsFile(const string &path);
        static bool checkIfFileContainsString(const string &path,const string &str);
        static pair<vector<pair<int, vector<float>>>,map<string,int>> enumfyDataset(const vector<pair<string, vector<float>>> &vec);
        static vector<pair<vector<int>, vector<float>>> extractSubVector(const vector<pair<vector<int>, vector<float>>> &vec, int start, int size);
        static vector<pair<vector<int>, vector<float>>> encodeDatasetLabels(const vector<pair<int, vector<float>>> &vec, DataEncoder enc, int ext_max=-1);
        static vector<pair<vector<int>, vector<float>>> encodeDatasetLabelsUsingFirst(const vector<pair<vector<int>, vector<float>>> &vec, DataEncoder enc, int ext_max=-1);
        static vector<pair<vector<int>, vector<float>>> shuffleDataset(const vector<pair<vector<int>, vector<float>>> &vec);
        static pair<vector<pair<vector<int>, vector<float>>>,vector<pair<vector<int>, vector<float>>>> divideDataSet(const vector<pair<vector<int>, vector<float>>> &vec, float percentageOfFirst);
        static pair<vector<pair<float,float>>,vector<pair<vector<int>, vector<float>>>> normalizeDataset(const vector<pair<vector<int>, vector<float>>> &vec);
        static float getRandomBetweenZeroAndOne();
        static boost::uuids::uuid genRandomUUID();
        static string genRandomUUIDStr();
        static string msToHumanReadable(long timestamp);
        static snn_stats statisticalAnalysis(vector<vector<int>> correct, vector<vector<int>> pred);
        static snn_stats statisticalAnalysis(vector<pair<vector<int>, vector<float>>> correct, vector<vector<pair<int,float>>> pred);
        static void printStats(snn_stats stats);
        template<typename T>
        static vector<T> subtractVectors(vector<T> a, vector<T> b){
            vector<T> out;
            remove_copy_if(a.begin(), a.end(), back_inserter(out),
                [&b](const T& arg)
                {return (find(b.begin(), b.end(), arg) != b.end());});
            return out;
        }
        static void serializeWeigths(map<string, vector<float>> weights, string filename,string except_str="");
        static map<string, vector<float>> deserializeWeigths(string filename,string except_str="");
        static bool mkdir(string path);
        static string getHostname();
        static string getStrNow(string format="%d-%m-%Y %H:%M:%S");
        static string serializeWeigthsToStr(map<string, vector<float>> weights);
        static map<string, vector<float>> deserializeWeigthsFromStr(string serialized_str);
        static void compareAndPrintLabel(vector<vector<int>> correct, vector<vector<int>> pred);
        static void compareAndPrintLabel(vector<pair<vector<int>, vector<float>>> correct, vector<vector<pair<int,float>>> pred);
        static string compressBase64(const string &in_64);
        static string decompressBase64(const string &in_65);
        static string base64ToString(const string &in_65);
        static string stringToBase64(const string &in);
        static string stringToBase65(const string &in);
        static vector<pair<vector<int>, vector<float>>> balanceSingleLabelDataset(const vector<pair<vector<int>, vector<float>>> &vec);

        // variables
        static mt19937_64 RNG;

    private:
        // methods
        static mt19937_64 getRandomGenerator();
        static uniform_real_distribution<float> dist_zero_one;

        // variables
        static string RESOURCES_FOLDER;
        static const string FILE_SEPARATOR;
};
