#pragma once

#include <iostream>
#include <math.h>

#include "Slide.hpp"
#include "Utils.hpp"
#include "MongoDB.hpp"
#include "Slide.hpp"
#include "PopulationManager.hpp"
#include "HallOfFame.hpp"
#include "StandardGenetic.hpp"
#include "NeuralGenome.hpp"
#include "GarbageCollector.hpp"

using namespace std;

void test(int func_id);
void testCsvRead();
void testMongo();
void testSlide_IntLabel();
void testSlide_NeuronByNeuronLabel();
void testStdGeneticsOnMath();
void testEnchancedGeneticsOnMath();
void testSlide_Validation();
void testGeneticallyTunedNeuralNetwork();
void testMongoCveRead();
void testSmartNeuralNetwork_cveData();
void testGeneticallyTunedSmartNeuralNetwork_cveData();