#include "headers/convolution.h"

__global__
void gpu_convolution_naive(PngImage* image, int K_dim, matrix K, PngImage* output){

    unsigned int thread_u, thread_v; // image pixel indeces (on which conv. is currently computed)
    unsigned int i,j;                // kernel indeces
    unsigned int patch_u, patch_v;  // image pixel currently evaluated by kernel
    unsigned int c;                  // channel index

    thread_u = blockIdx.x * blockDim.x + threadIdx.x;
    thread_v = blockIdx.y * blockDim.y + threadIdx.y;
    
    matrix_element sum = 0.0;
    unsigned int K_center = K_dim / 2;

    if(thread_u >= image->PAD && thread_u < image->W+image->PAD && thread_v >= image->PAD && thread_v < image->H+image->PAD){
        sum = 0.0;
        for(i = 0; i < K_dim; i++){
            for(j = 0; j < K_dim; j++){
                for(c = 0; c < image->C; c++){
                    patch_u = thread_u-K_center+i;
                    patch_v = thread_v-K_center+j;

                    sum += K[i*K_dim+j] * image->val[patch_u * (image->W + 2*image->PAD) * image->C +patch_v*image->C + c];
                }
            }
        }
        output->val[thread_u*image->W*image->C + thread_v*image->C + c] = sum;
    }
}

void cpu_convolution_naive(PngImage* image, int K_dim, matrix K, PngImage* output){
    unsigned int u,v;               // image pixel indeces (on which conv. is currently computed)
    unsigned int i,j;               // kernel indeces
    unsigned int patch_u, patch_v;  // image pixel currently evaluated by kernel
    unsigned int c;                 // channel index
    
    matrix_element sum = 0.0;
    unsigned int K_center = K_dim / 2;

    for(u = image->PAD; u < (image->H + image->PAD); u++){
        for(v = image->PAD; v < (image->W + image->PAD); v++){
	        for(c = 0; c < image->C; c++){
                sum = 0.0;
                for(i = 0; i < K_dim; i++){
                    for(j = 0; j < K_dim; j++){
                        patch_u = u-K_center+i;
                        patch_v = v-K_center+j;
                    
                        sum += K[i*K_dim+j] * image->val[patch_u * (image->W + 2*image->PAD) * image->C +patch_v*image->C + c];
                    }   
                }
                output->val[u*image->W*image->C + v*image->C + c] = sum;
            }
        }
     }
}
