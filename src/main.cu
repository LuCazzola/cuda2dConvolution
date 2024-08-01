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

    /*
    // options to set in "launch.sh" file
    char method [30];
    int N, block_size, th_size_x, th_size_y;
    // parse command line options
    process_transpose_gpu_options(argc, argv, method, &N, &block_size, &th_size_x, &th_size_y);

    const int ROWS = (int) pow(2, N), COLS = ROWS, SIZE = ROWS;  // number of elements in a matrix ROW or COLUMN (SIZE)
    const int BLK_SIZE = (int) pow(2, block_size);               // matrix block size (same concept as Tiles)
    const int THREAD_DIM_X = (int) pow(2, th_size_x);            // defines thread blockDim.x
    const int THREAD_DIM_Y = (int) pow(2, th_size_y);            // defines thread blockDim.y
     
    // Grid dimension is assumed to be large such that it covers the entire input matrix
    dim3 gridSize((int)(SIZE / BLK_SIZE), (int)(SIZE / BLK_SIZE), 1);
    // Thread block dimensions according to input
    dim3 blockSize(THREAD_DIM_X, THREAD_DIM_Y, 1);    
    */
    int K_dim = 3;
    int padding = K_dim/2;
    PngImage* image = read_png("images/lenna_gray.png", padding);

    printf("Size: %d x %d", image->H, image->W);

    // ===================================== Memory Allocations =====================================


    // ===================================== RUN =====================================
    
    int IMAGE_DIM_X = 3;
    int IMAGE_DIM_Y = 5;

    int* host_image;
    int* dev_image;
    int* gpu_output;
    int* dev_output;
    int* cpu_output;
    float *dev_K;

    host_image = generate_image(IMAGE_DIM_X, IMAGE_DIM_Y);
    float K[] = {1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0};

    // Run serial convolution
    //gpu_output = (matrix) malloc(sizeof(matrix_element) * IMAGE_DIM_X*IMAGE_DIM_Y);
    cpu_output = (matrix) malloc(sizeof(matrix_element) * IMAGE_DIM_X*IMAGE_DIM_Y);
    cpu_convolution(IMAGE_DIM_X, IMAGE_DIM_Y, host_image, K_dim, K, cpu_output);

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

    printf("\nImage\n");
    for(int y = 0; y < IMAGE_DIM_Y; y++){
        for(int x = 0; x < IMAGE_DIM_X; x++){
            printf("%2d ", host_image[x*IMAGE_DIM_X+y]);
        }
        printf("\n");
    }
    
    printf("\nCPU Output\n");
    for(int y = 0; y < IMAGE_DIM_Y; y++){
        for(int x = 0; x < IMAGE_DIM_X; x++){
            int idx = x*IMAGE_DIM_X+y;
            printf("%2d ", cpu_output[idx]);
        }
        printf("\n");
    }

    printf("\nGPU Output\n");
    for(int y = 0; y < IMAGE_DIM_Y; y++){
        for(int x = 0; x < IMAGE_DIM_X; x++){
            int idx = x*IMAGE_DIM_X+y;
            printf("%2d ", gpu_output[idx]);
        }
        printf("\n");
    }

    printf("Time: %.3f\n", time);
    //printf("Error: %.3f\n", error);

    // ===================================== FREE MEMORY =====================================

    free(host_image);  
    free(cpu_output);  
    free(gpu_output);
    checkCuda( cudaFree(dev_image) ); 
    checkCuda( cudaFree(dev_output) );

	return 0;
}
