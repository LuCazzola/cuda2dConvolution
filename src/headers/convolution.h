#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "matrix.h"
#include "pngUtils.h"
#include "common.h"
#include "cudaUtils.h"
#include "convolution.h"


// convolution implementations
void cpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int K_DIM);
__global__ void gpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int K_DIM );
__global__ void gpu_convolution_shared(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int K_DIM );
__global__ void gpu_convolution_shared_constk(matrix image, matrix output, const int W, const int H, const int C, const int K_DIM);
__global__ void gpu_convolution_shared_constk_cached(matrix image, matrix output, const int W, const int H, const int C, const int K_DIM);

// set kernel as average kernel
void fill_mean_kernel (matrix K, const int K_DIM);
// wrapper function to fill array in constant memory
void fill_const_kernel (matrix h_k, const int TOT_K_DIM);
// Get the number of bytes read and written by the convolution kernel
long int get_conv_bytes_read_write(const int W, const int H, const int C, const int K_DIM);
// Get the number of FLOPs performed by the convolution kernel
long int get_conv_flops(const int W, const int H, const int C, const int K_DIM);

#endif
