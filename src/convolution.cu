#include "headers/convolution.h"

// set kernel as average kernel
void fill_mean_kernel (matrix K, const int K_DIM){
    matrix_element mean = 1.0 / (K_DIM * K_DIM);

    for(int i = 0; i < K_DIM; i++){
        for(int j = 0; j < K_DIM; j++){
            K[i*K_DIM + j] = mean;
        }
    }
}
// Get the number of bytes read and written by the convolution kernel
long int get_conv_bytes_read_write(const int W, const int H, const int C, const int PAD, const int K_DIM){
    const int TOT_SIZE_NOPAD = W * H * C; 
    const int TOT_K_DIM = K_DIM*K_DIM;
    
    const long int BR = sizeof(matrix_element) * ((TOT_K_DIM + TOT_K_DIM) * TOT_SIZE_NOPAD); 
    const long int BW = sizeof(matrix_element) * TOT_SIZE_NOPAD;

    return BR + BW;
}
// Get the number of FLOPs performed by the convolution kernel
long int get_conv_flops(const int W, const int H, const int C, const int PAD, const int K_DIM){
    const int TOT_SIZE_NOPAD = W * H * C; 
    const int TOT_K_DIM = K_DIM*K_DIM;
    
    const long int FLOP = (TOT_K_DIM + TOT_K_DIM) * TOT_SIZE_NOPAD;

    return FLOP;
}


// ================================================================================================================================= //
// ================================================== CONVOLUTION IMPLEMENTATIONS ================================================== //
// ================================================================================================================================= //


void cpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM){
    int u,v;               // image pixel indeces (on which conv. is currently computed)
    int i,j;               // kernel indeces
    int patch_u, patch_v;  // image pixel currently evaluated by kernel
    int c;                 // channel index
    const int K_CENTER = (int)(K_DIM / 2);
    matrix_element sum;

    for(u = PAD; u < (H + PAD); u++){
        for(v = PAD; v < (W + PAD); v++){
	        for(c = 0; c < C; c++){
                sum = 0.0;
                for(i = 0; i < K_DIM; i++){
                    for(j = 0; j < K_DIM; j++){
                        patch_u = u - K_CENTER + i;
                        patch_v = v - K_CENTER + j;
                    
                        sum += K[i*K_DIM + j] * image[patch_u*(W + 2*PAD)*C + patch_v*C + c];
                    }   
                }
                output[(u-PAD)*W*C + (v-PAD)*C + c] = sum;
            }
        }
     }
}


__global__
void gpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM){
    int u = blockIdx.x*blockDim.x + threadIdx.x; // image pixel (u) (on which conv. is currently computed)
    int v = blockIdx.y*blockDim.y + threadIdx.y; // image pixel (v) (on which conv. is currently computed)
    
    if(u >= PAD && v >= PAD && u < W+PAD && v < H+PAD){
        int i,j;                // kernel indeces
        int patch_u, patch_v;   // image pixel currently evaluated by kernel
        int c;                  // channel index
        const int K_CENTER = (int)(K_DIM / 2);
        matrix_element sum;

        for(c = 0; c < C; c++){
            sum = 0.0;
            for(i = 0; i < K_DIM; i++){
                for(j = 0; j < K_DIM; j++){
                        patch_u = u - K_CENTER + i;
                        patch_v = v - K_CENTER + j;

                        sum += K[i*K_DIM + j] * image[patch_u*(W + 2*PAD)*C + patch_v*C + c];
                    }
                }
            output[(u-PAD)*W*C + (v-PAD)*C + c] = sum;
        }
    }
}

__global__ 
void gpu_convolution_shared(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int PAD, const int K_DIM) {
    int u = blockIdx.x*blockDim.x + threadIdx.x; // image pixel (u) (on which conv. is currently computed)
    int v = blockIdx.y*blockDim.y + threadIdx.y; // image pixel (v) (on which conv. is currently computed)

    extern __shared__ matrix_element shared_image[];

    for(int c = 0; c < C; c++){
        shared_image[(threadIdx.x*blockDim.x)*C + threadIdx.y*C + c] = image[u*(W + 2*PAD)*C + v*C + c]; //Maybe remove padding
    }

   __syncthreads();

    if(u >= PAD && v >= PAD && u < W+PAD && v < H+PAD){
        int i,j;                // kernel indeces
        int patch_u, patch_v;   // image pixel currently evaluated by kernel
        int c;                  // channel index
        const int K_CENTER = (int)(K_DIM / 2);
        matrix_element sum;

        for(c = 0; c < C; c++){
            sum = 0.0;
            for(i = 0; i < K_DIM; i++){
                for(j = 0; j < K_DIM; j++){
                        sum += K[i*K_DIM + j] * shared_image[(threadIdx.x - K_CENTER + i)*(blockDim.x + 2*PAD)*C + (threadIdx.y - K_CENTER + j)*C + c];
                    }
                }
            output[(u-PAD)*W*C + (v-PAD)*C + c] = sum;
        }
    }

    __syncthreads();
}
