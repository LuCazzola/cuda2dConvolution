extern "C" { 
    #include "headers/matrix.h"
	#include "headers/common.h"
    #include "headers/opt_parser.h"
    #include "headers/pngUtils.h"
}
#include "headers/convolution.h"

// set to true/false to enable/disable debugging outputs
#define PRINT_MATRICES false
#define PRINT_MAT_ERROR false

void print_metrics (double exec_time, const int SIZE){
    // metrics evaluation
    printf("\n\n========================== METRICS ==========================\n");

    // each element in the matrix (except the diagonal) is
    // subject to one read and one write operation
    // total reads + writes = 2 * size^2 (expressed in bytes)
    double Br_Bw = sizeof(matrix_element) * (SIZE * SIZE) * 2;

    // effective bandwidth (expressed in GB/s)
    double effective_bandwidth = ( Br_Bw / pow(10,9) ) / exec_time;

    // print out values
    printf("\nExecution time :       %f s\n", exec_time);
    printf("\nEffective Bandwidth :  %f GB/s\n\n", effective_bandwidth);
}


void print_run_infos(char *method, const int N, const int block_size, const int th_size_x, const int th_size_y){
    printf("\n-   Matrix elemets datatype : %s\n", VALUE(MATRIX_ELEM_DTYPE));
    printf("-   Matrix size       :       2^%d x 2^%d\n", N, N);
    printf("-   Matrix block size :       2^%d x 2^%d\n\n", block_size, block_size);
    
    printf("Method: %s on GPU\n", method);
    printf("-   Grid  dim :                2^%d x 2^%d\n", N-block_size, N-block_size);
    printf("-   Block dim :               2^%d x 2^%d\n", th_size_x, th_size_y);
    printf("\nREMINDER : in Cuda the dimensions are expressed in CARTESIAN coordinates !\n");
}

int* generate_image(int dim_x, int dim_y){
    // Generate image for testing, use loaded image later
    int* image = (int*) malloc(dim_x * dim_y * sizeof(int));
    for(int i = 0; i < dim_x * dim_y; i++){
        image[i] = rand() % 100;
    }
    return image;
}

int main(int argc, char * argv []){

    // ===================================== Parameters Setup =====================================

    int K_dim = 3;
    float K[] = {1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0};
   
    int padding = K_dim/2;
    PngImage* image = read_png("images/lenna.png", padding);

    printf("Size: %d x %d x %d\n", image->C, image->H, image->W);
    printf("Padding: %d\n", image->PAD);
    
    // ===================================== Memory Allocations =====================================

    PngImage* cpu_output_image = (PngImage*) malloc(sizeof(PngImage));
    cpu_output_image->PAD = 0;
    cpu_output_image->W = image->W;
    cpu_output_image->H = image->H;
    cpu_output_image->C = image->C;
    cpu_output_image->color_type = image->color_type;
    cpu_output_image->val = (matrix) malloc(sizeof(matrix_element)*image->W*image->H*image->C);

    // ===================================== RUN =====================================
    
    TIMER_DEF;
    TIMER_START;
    cpu_convolution(image, K_dim, K, cpu_output_image);
    TIMER_STOP;

    printf("Time: %.2f\n", TIMER_ELAPSED);
    write_png("images/output.png", cpu_output_image);    
    return 0;
    
    int* host_image;
    int* dev_image;
    int* gpu_output;
    int* dev_output;
    float *dev_K;

    // Run serial convolution
    //gpu_output = (matrix) malloc(sizeof(matrix_element) * IMAGE_DIM_X*IMAGE_DIM_Y);
    //cpu_convolution(image, K_dim, K, cpu_output);

    // Run parallel convolution
    /*
    TIMER_DEF;
    dim3 dimBlock(IMAGE_DIM_X, IMAGE_DIM_Y);
    checkCuda( cudaMalloc((void **)&dev_image, IMAGE_DIM_X*IMAGE_DIM_Y*sizeof(int)) );
    checkCuda( cudaMalloc((void **)&dev_output, IMAGE_DIM_X*IMAGE_DIM_Y*sizeof(int)) );
    checkCuda( cudaMalloc((void **)&dev_K, K_dim*K_dim*sizeof(float)) );
    checkCuda( cudaMemset((void*)&dev_output, 0, IMAGE_DIM_X*IMAGE_DIM_Y*sizeof(int)) );
    checkCuda( cudaMemcpy(dev_image, host_image, IMAGE_DIM_X*IMAGE_DIM_Y*sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_K, K, K_dim*K_dim*sizeof(float), cudaMemcpyHostToDevice) );
    TIMER_START;
    gpu_convolution<<<1, dimBlock>>>(IMAGE_DIM_X, IMAGE_DIM_Y, dev_image, K_dim, dev_K, dev_output);
    checkCuda( cudaDeviceSynchronize() );
    TIMER_STOP;
    checkCuda( cudaMemcpy(gpu_output, (void*)dev_output, IMAGE_DIM_X*IMAGE_DIM_Y*sizeof(int), cudaMemcpyDeviceToHost) );

    //Check for errors
    float error = 0.0;
    for(int y = 0; y < IMAGE_DIM_Y; y++){
        for(int x = 0; x < IMAGE_DIM_X; x++){
            int idx = x*IMAGE_DIM_X+y;
            error += (cpu_output[idx] - gpu_output[idx]) < 0 ? -(cpu_output[idx] - gpu_output[idx]) : (cpu_output[idx] - gpu_output[idx]) ;
        }
    }
    float time = TIMER_ELAPSED;
    */

    // ===================================== FREE MEMORY =====================================

    free(host_image);  
    free(cpu_output_image);  
    free(gpu_output);
    checkCuda( cudaFree(dev_image) ); 
    checkCuda( cudaFree(dev_output) );
    
    return 0;
}
