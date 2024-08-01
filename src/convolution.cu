#include "headers/convolution.h"

__global__
void gpu_convolution(int image_dim_x, int image_dim_y, int* image, int K_dim, float* K, int* output){
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.0;
    int K_center = K_dim / 2;

    if(thread_x >= 0 && thread_x < image_dim_x && thread_y >= 0 && thread_y < image_dim_y){
        sum = 0.0;
        for(int i = 0; i < K_dim; i++){
            for(int j = 0; j < K_dim; j++){
                int patch_i = thread_x-K_center+i;
                int patch_j = thread_y-K_center+j;
                if(patch_i < 0 || patch_i >= image_dim_x || patch_j < 0 || patch_j >= image_dim_y){
                    sum += 0.0;
                }
                else {
                    sum += K[i*K_dim+j] * (float) image[patch_i*image_dim_x+patch_j];
                }
            }
        }
        output[thread_x*image_dim_x+thread_y] = (int) sum;
    }
}

void cpu_convolution(PngImage* image, int K_dim, matrix K, matrix output){
    if(K_dim % 2 == 0){
        perror("Method 'apply_convolution' takes only filters with odd dimensions. Got even dimension.");
        exit(EXIT_FAILURE);
    }
    
    unsigned int u,v;               // image pixel indeces (on which conv. is currently computed)
    unsigned int i,j;               // kernel indeces
    unsigned int patch_u, patch_v;  // image pixel currently evaluated by kernel

    matrix_element sum = 0.0;
    unsigned int K_center = K_dim / 2;

    for(u = image->PAD; u < (image->H + image->PAD); u++){
        for(v = image->PAD; v < (image->W + image->PAD); v++){
            sum = 0.0;
            for(i = 0; i < K_dim; i++){
                for(j = 0; j < K_dim; j++){
                    patch_u = u-K_center+i;
                    patch_v = v-K_center+j;
                    
                    sum += K[i*K_dim+j] * image->val[patch_u * (image->W + 2*image->PAD) +patch_v];
                }
            }
            output[u*image->W + v] = sum;
        }
    }
}
