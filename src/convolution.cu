#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IMAGE_DIM 5

void generate_image(int dim, int* image){
    // Generate image for testing, use loaded image later
    for(int i = 0; i < dim * dim; i++){
        image[i] = rand() % 100;
    }
}

__global__
void gpu_convolution(int image_dim, int* image, int K_dim, float* K, int* output){
    if(K_dim % 2 == 0){
        perror("Method 'apply_convolution' takes only filters with odd dimensions. Got even dimension.");
        exit(EXIT_FAILURE);
    }

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;
    int K_center = K_dim / 2;

    if(thread_x >= 0 && thread_x < image_dim && thread_y >= 0 && thread_y < image_dim){
        sum = 0.0;
        for(int i = 0; i < K_dim; i++){
            for(int j = 0; j < K_dim; j++){
                int patch_i = thread_x-K_center+i;
                int patch_j = thread_y-K_center+j;
                if(patch_i < 0 || patch_i >= image_dim || patch_j < 0 || patch_j >= image_dim){
                    sum += 0;
                }
                else {
                    sum += K[i*K_dim+j] * (float) image[patch_i*image_dim+patch_j];
                }
            }
        }
        output[thred_x*image_dim+thread_y] = (int) sum;
    }
}

void cpu_convolution(int image_dim, int* image, int K_dim, float* K, int* output){
    if(filter_dim % 2 == 0){
        perror("Method 'apply_convolution' takes only filters with odd dimensions. Got even dimension.");
        exit(EXIT_FAILURE);
    }
    
    float sum = 0.0;
    int K_center = K_dim / 2;

    for(int u = 0; u < image_dim; u++){
        for(int v = 0; v < image_dim; v++){
            sum = 0.0;
            for(int i = 0; i < K_dim; i++){
                for(int j = 0; j < K_dim; j++){
                    int patch_u = u-K_center+i;
                    int patch_v = v-K_center+j;
                    if(patch_u < 0 || patch_u >= image_dim || patch_v < 0 || patch_v >= image_dim){
                        sum += 0;
                    }
                    else {
                        sum += K[i*K_dim+j] * (float) image[patch_u*image_dim+patch_v];
                    }
                }
            }
            output[u*image_dim+v] = (int) sum;
        }
    }
}