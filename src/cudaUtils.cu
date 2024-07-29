#include "headers/cudaUtils.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result){
  #if defined(DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
  #endif

  return result;
}

void gpu_fill_rand(matrix mat, const int SIZE){
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, mat, SIZE*SIZE);
}

// Function to fill device matrix with random values faster
__global__ void generate_in_a_b(matrix mat, const float A, const float B, const int SIZE, const int BLK_SIZE, const int TOT_SIZE) {

    const unsigned int x = blockIdx.x * BLK_SIZE + threadIdx.x;
    const unsigned int y = blockIdx.y * BLK_SIZE + threadIdx.y;
    unsigned int i, j, idx;

    for (j = 0; j < BLK_SIZE; j += blockDim.y){
        for (i = 0; i < BLK_SIZE; i += blockDim.x){
            idx = (y+j)*SIZE + x+i;

            if (idx < TOT_SIZE){
                mat[idx] = (B-A) * mat[idx] + A;
            }
        }
    }
}

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}