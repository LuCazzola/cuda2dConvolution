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
long int get_conv_bytes_read_write(const int W, const int H, const int C, const int K_DIM){
    long int TOT_SIZE = W * H * C; 
    long int TOT_K_DIM = K_DIM*K_DIM;
    
    long int BR = sizeof(matrix_element) * ((TOT_K_DIM + TOT_K_DIM) * TOT_SIZE); 
    long int BW = sizeof(matrix_element) * TOT_SIZE;

    return BR + BW;
}
// Get the number of FLOPs performed by the convolution kernel
long int get_conv_flops(const int W, const int H, const int C, const int K_DIM){
    long int TOT_SIZE = W * H * C; 
    long int TOT_K_DIM = K_DIM*K_DIM;
    
    long int FLOP = (TOT_K_DIM + TOT_K_DIM) * TOT_SIZE;

    return FLOP;
}


// ================================================================================================================================= //
// ================================================== CONVOLUTION IMPLEMENTATIONS ================================================== //
// ================================================================================================================================= //


void cpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int K_DIM){
    int u,v;               // image pixel indeces (on which conv. is currently computed)
    int i,j;               // kernel indeces
    int patch_u, patch_v;  // image pixel currently evaluated by kernel
    int c;                 // channel index
    const int K_CENTER = (int)(K_DIM / 2);
    matrix_element sum;

    for(u = 0; u < H; u++){
        for(v = 0; v < W; v++){
	        for(c = 0; c < C; c++){
                sum = 0.0;
                for(i = 0; i < K_DIM; i++){
                    for(j = 0; j < K_DIM; j++){
                        patch_u = u - K_CENTER + i;
                        patch_v = v - K_CENTER + j;
                        if (patch_u >= 0 && patch_v >= 0 && patch_u < W && patch_v < H){
                            sum += K[i*K_DIM + j] * image[patch_u*W*C + patch_v*C + c];
                        }
                    }   
                }
                output[u*W*C + v*C + c] = sum;
            }
        }
     }
}


__global__
void gpu_convolution_naive(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int K_DIM){
    int u = blockIdx.x*blockDim.x + threadIdx.x; // image pixel (u) (on which conv. is currently computed)
    int v = blockIdx.y*blockDim.y + threadIdx.y; // image pixel (v) (on which conv. is currently computed)
    
    if(u >= 0 && v >= 0 && u < W && v < H){
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
                        if (patch_u >= 0 && patch_v >= 0 && patch_u < W && patch_v < H){
                            sum += K[i*K_DIM + j] * image[patch_u*W*C + patch_v*C + c];
                        }
                    }
                }
            output[u*W*C + v*C + c] = sum;
        }
    }
}

__global__ 
void gpu_convolution_shared(matrix image, matrix K, matrix output, const int W, const int H, const int C, const int K_DIM) {
    
    const int PAD = (int) (K_DIM / 2);
    
    int in_tile_dim = blockDim.x;
    int out_tile_dim = in_tile_dim - 2*PAD;
    
    int col = blockIdx.x*out_tile_dim + threadIdx.x - PAD;
    int row = blockIdx.y*out_tile_dim + threadIdx.y - PAD; 

    // Loading input tile
    extern __shared__ matrix_element shared_image[];
    K = &shared_image[(blockDim.x+2*PAD)*(blockDim.y+2*PAD)*C];

    for(int c = 0; c < C; c++){
        if(row >= 0 && row < H && col >= 0 && col < W){
            shared_image[threadIdx.y*blockDim.x*C + threadIdx.x*C + c] = image[row*W*C + col*C + c];
        }
        else {
            shared_image[threadIdx.y*blockDim.x*C + threadIdx.x*C + c] = 0.0;
        }
    }

    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - PAD;
    int tileRow = threadIdx.y - PAD;
    // Turning off threads at the edges of the block
    if(col >= 0 && col < W && row >= 0 && row < H){
        if(tileCol >= 0 && tileCol < out_tile_dim && tileRow >= 0 && tileRow < out_tile_dim){
            for(int c = 0; c < C; c++){
                float sum = 0.0f;
                for(int i = 0; i < K_DIM; i++){
                    for(int j = 0; j < K_DIM; j++){
                        sum += K[i*K_DIM + j] * shared_image[(tileRow+i)*blockDim.x*C + (tileCol+j)*C + c];
                    }
                }
                output[row*W*C + col*C + c] = sum;
            }
        }
    }
}
