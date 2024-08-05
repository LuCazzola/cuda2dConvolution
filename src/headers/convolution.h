#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "matrix.h"
#include "pngUtils.h"
#include "common.h"

#include "cudaUtils.h"
#include "convolution.h"

__global__ void gpu_convolution_naive(PngImage* image, int K_dim, matrix K, PngImage* output);
void cpu_convolution_naive(PngImage* image, int K_dim, matrix K, PngImage* output);

#endif
