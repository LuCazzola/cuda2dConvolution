#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "matrix.h"
#include "pngUtils.h"
#include "common.h"

#include "cudaUtils.h"
#include "convolution.h"

__global__ void gpu_convolution(int image_dim_x, int image_dim_y, int* image, int K_dim, float* K, int* output);
void cpu_convolution(PngImage* image, int K_dim, matrix K, PngImage* output);

#endif
