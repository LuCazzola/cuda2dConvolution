#include "headers/convolution.h"

void fill_mean_kernel (matrix K, const int K_DIM){
    matrix_element mean = 1.0 / (K_DIM * K_DIM);

    for(int i = 0; i < K_DIM; i++){
        for(int j = 0; j < K_DIM; j++){
            K[i*K_DIM + j] = mean;
        }
    }
}

__global__
void gpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM){
    int u = blockIdx.x * blockDim.x + threadIdx.x; // image pixel (u) (on which conv. is currently computed)
    int v = blockIdx.y * blockDim.y + threadIdx.y; // image pixel (v) (on which conv. is currently computed)
    
    int i,j;                // kernel indeces
    int patch_u, patch_v;   // image pixel currently evaluated by kernel
    int c;                  // channel index
    matrix_element sum = 0.0;
    const int K_CENTER = K_DIM / 2;

    if(u >= PAD && v >= PAD && u < W+PAD && && v < H+PAD){
        sum = 0.0;
        for(c = 0; c < C; c++){
            for(i = 0; i < K_DIM; i++){
                for(j = 0; j < K_DIM; j++){
                        patch_u = u - K_CENTER + i;
                        patch_v = v - K_CENTER + j;
                        sum += K[i*K_DIM + j] * image[patch_u * (W + 2*PAD) * C +patch_v*C + c];
                    }
                }
            output[(u*W*C) + (v*C) + c] = sum;
        }
    }
}

void cpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM){
    int u,v;               // image pixel indeces (on which conv. is currently computed)
    int i,j;               // kernel indeces
    int patch_u, patch_v;  // image pixel currently evaluated by kernel
    int c;                 // channel index
    
    matrix_element sum = 0.0;
    const int K_CENTER = K_DIM / 2;

    for(u = PAD; u < (H + PAD); u++){
        for(v = PAD; v < (W + PAD); v++){
	        for(c = 0; c < C; c++){
                sum = 0.0;
                for(i = 0; i < K_DIM; i++){
                    for(j = 0; j < K_DIM; j++){
                        patch_u = u - K_CENTER + i;
                        patch_v = v - K_CENTER + j;
                    
                        // Ensure patch_u and patch_v are within bounds
                        if (patch_u >= 0 && patch_u < (H + 2*PAD) && patch_v >= 0 && patch_v < (W + 2*PAD)) {
                            sum += K[i*K_DIM + j] * image[patch_u * (W + 2*PAD) * C + patch_v * C + c];
                        }
                    }   
                }
                // Ensure u and v are within bounds for output
                if (u >= PAD && u < (H + PAD) && v >= PAD && v < (W + PAD)) {
                    output[(u - PAD) * W * C + (v - PAD) * C + c] = sum;
                }
            }
        }
     }
}
