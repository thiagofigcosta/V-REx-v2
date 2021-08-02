// SLIDE: https://github.com/keroro824/HashingDeepLearning 
#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define MURMURHASH_VERSION "0.0.3"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns a murmur hash of `key' based on `seed'
 * using the MurmurHash3 algorithm
 */

uint32_t MurmurHash (const char *, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif