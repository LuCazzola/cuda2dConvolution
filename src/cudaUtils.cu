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

void gpu_fill_rand(matrix mat, const int TOT_SIZE) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Generate uniform random numbers in the range [0, 1)
    curandGenerateUniform(prng, mat, TOT_SIZE);

    // define random range of values according to the datatype
    char datatype [] = {VALUE(MATRIX_ELEM_DTYPE)};
    bool is_int = strcmp(datatype, "int") == 0 ? true : false;
    srand(time(NULL));

    const float A = is_int ? (int)(rand() - rand()) : (float)((float)(rand()) / (float)(rand())-(float)(rand()) / (float)(rand()));
    const float B = is_int ? (int)(rand() - rand()) : (float)((float)(rand()) / (float)(rand())-(float)(rand()) / (float)(rand()));
    const float VAL_MIN = -abs(A);
    const float VAL_MAX = abs(B);
    const float VAL_RANGE = VAL_MAX - VAL_MIN;
    int threadsPerBlock = 256;
    int blocksPerGrid = (TOT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    scale_and_shift<<<blocksPerGrid, threadsPerBlock>>>(mat, TOT_SIZE, VAL_MIN, VAL_RANGE);
    checkCuda( cudaDeviceSynchronize() );

    curandDestroyGenerator(prng);
}

__global__ void scale_and_shift(matrix mat, const int SIZE, const matrix_element VAL_MIN, const matrix_element VAL_RANGE) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        mat[idx] = VAL_MIN + mat[idx] * VAL_RANGE;
    }
}

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}