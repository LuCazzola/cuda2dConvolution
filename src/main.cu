extern "C" { 
	#include "headers/common.h"
    #include "headers/opt_parser.h"
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


int main(int argc, char * argv []){

    // ===================================== Parameters Setup =====================================

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
    

    // ===================================== Memory Allocations =====================================

    // host matrix, device matrix, copy of host matrix pre-transposition
    matrix h_mat, d_mat, cpy, filler;

    h_mat = (matrix) malloc(ROWS*COLS*sizeof(matrix_element));  // Allocate matrix on Host
    checkCuda( cudaMalloc((void **)&d_mat, ROWS*COLS*sizeof(matrix_element)) );  // Allocate space on the DEVICE global memory
    checkCuda( cudaMemset(d_mat, 0, ROWS*COLS*sizeof(matrix_element)) );
    
    fill_matrix(h_mat, ROWS, COLS);  // fill the host matrix with random values
	checkCuda( cudaMemcpy(d_mat, h_mat, ROWS*COLS*sizeof(matrix_element), cudaMemcpyHostToDevice) );  // Copy from Host to device

    if (PRINT_MAT_ERROR){
        cpy = matrix_copy(h_mat, ROWS, COLS);
    }

    // ===================================== GPU TRANSPOSE KERNEL LAUNCH =====================================
    
    // run infos
    print_run_infos(method, N, block_size, th_size_x, th_size_y);
    // Setup a timer
    TIMER_DEF;
    // "Wake up" the GPU before executing the kernel
    warm_up_gpu<<<gridSize, blockSize>>> ();
    checkCuda( cudaDeviceSynchronize() );
    // allocate a filler block as large as the L2 cache and access it
    // that's used to basically flush the L2 cache from previous accesses
    checkCuda( cudaMalloc((void **)&filler, 2*L2_CACHE_SIZE) );  
    checkCuda( cudaMemset(filler, 0, 2*L2_CACHE_SIZE) );
    checkCuda( cudaFree(filler) );

    // run test on transpose_blocks_gpu() kernel
    if (strcmp(method, "blocks_naive") == 0) {
        TIMER_START;
        transpose_blocks_gpu <<<gridSize, blockSize>>> (d_mat, SIZE, BLK_SIZE, SIZE*SIZE);
        checkCuda( cudaDeviceSynchronize() );
        TIMER_STOP;
    }
    // run test on transpose_blocks_gpu_coalesced() kernel
    if (strcmp(method, "blocks_coalesced") == 0) {
        size_t sharedBlockSize = 2 * BLK_SIZE*BLK_SIZE;

        TIMER_START;
        transpose_blocks_gpu_coalesced <<<gridSize, blockSize, sharedBlockSize*sizeof(matrix_element)>>> (d_mat, SIZE, BLK_SIZE, SIZE*SIZE);
        checkCuda( cudaDeviceSynchronize() );
        TIMER_STOP;
    }
    
    // copy back to host
    checkCuda( cudaMemcpy(h_mat, d_mat, ROWS * COLS * sizeof(matrix_element), cudaMemcpyDeviceToHost) );
    // Print out execution time & effective bandwidth
    print_metrics (TIMER_ELAPSED, SIZE);

    if (PRINT_MAT_ERROR){
        print_transpose_error(cpy, h_mat, SIZE);
    }
    if (PRINT_MATRICES){
        printf("=== Original matrix === :\n=\n");
        print_matrix(cpy, ROWS, COLS);
        printf("\n=== Transposed matrix === :\n=\n");
        print_matrix(h_mat, ROWS, COLS);
    }

    // Free memory
    free(h_mat);  // on Host
    checkCuda( cudaFree(d_mat) );  // on Device
    if (PRINT_MAT_ERROR){
        free(cpy);
    }

	return 0;
}
