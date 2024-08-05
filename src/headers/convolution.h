#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "matrix.h"
#include "pngUtils.h"
#include "common.h"
#include "cudaUtils.h"
#include "convolution.h"

__global__ void gpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM );
void cpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM);

// Fill kernel with mean values
void fill_mean_kernel (matrix K, const int K_DIM);


#endif
