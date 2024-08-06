#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include "matrix.h"
#include <cfloat>

// NVIDIA A30 L2 cache size in bytes
#define L2_CACHE_SIZE 25165824

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result);

// Functions to fill device matrix with random float32 values
void gpu_fill_rand(matrix mat, const int TOT_SIZE);
__global__ void scale_and_shift(matrix mat, const int SIZE, const matrix_element VAL_MIN, const matrix_element VAL_RANGE);

// Kernel which performs some random oparation to call before other kernels
__global__ void warm_up_gpu();

#endif


