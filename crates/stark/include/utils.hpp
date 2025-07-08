#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);    \
            fprintf(stderr, "Failing CUDA call: %s\n", #call);            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)


#define CHECK_KERNEL() \
    { cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } }