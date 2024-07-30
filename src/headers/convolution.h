#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "cudaUtils.h"
#include "convolution.h"

void cpu_convolution(int image_dim_x, int image_dim_y, int* image, int K_dim, float* K, int* output);
__global__ void gpu_convolution(int image_dim_x, int image_dim_y, int* image, int K_dim, float* K, int* output);

#endif