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
    int K_DIM = 0;
    char pngPath [50];
    char method [50];
    process_main_args (argc, argv, method, pngPath, &K_DIM);
    
    // get input image
    PngImage* input_image = read_png(pngPath, (int)(K_DIM/2));
    
    const int TOT_SIZE = (input_image->W + 2*input_image->PAD) * (input_image->H + 2*input_image->PAD) * input_image->C;
    const int TOT_SIZE_NOPAD = input_image->W * input_image->H * input_image->C;
    const int TOT_K_DIM = K_DIM*K_DIM; 

    // struct on which to store the output image once it's computed
    PngImage* output_image = (PngImage*) malloc(sizeof(PngImage));
    output_image->PAD = 0;
    output_image->W = input_image->W;
    output_image->H = input_image->H;
    output_image->C = input_image->C;
    output_image->color_type = input_image->color_type;
    output_image->val = (matrix) malloc(sizeof(matrix_element)* TOT_SIZE_NOPAD);

    // ===================================== Memory Allocations =====================================

    // HOST side
    //
    matrix h_in_image, h_out_image, h_K;
    // kernel
    h_K = (matrix) malloc(sizeof(matrix_element) * TOT_K_DIM);
    fill_mean_kernel(h_K, K_DIM);
    // input image
    h_in_image = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE);
    memcpy(h_in_image, input_image->val, sizeof(matrix_element) * TOT_SIZE);
    // output image
    h_out_image = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE_NOPAD);

    // DEVICE side
    //
    matrix d_in_image, d_out_image, d_K;
    // kernel
    checkCuda( cudaMalloc(&d_K, TOT_K_DIM * sizeof(matrix_element)) );
    checkCuda( cudaMemcpy(d_K, h_K, TOT_K_DIM * sizeof(matrix_element), cudaMemcpyHostToDevice) );
    // input image
    checkCuda( cudaMalloc(&d_in_image, TOT_SIZE * sizeof(matrix_element)) );
    checkCuda( cudaMemcpy(d_in_image, h_in_image, TOT_SIZE * sizeof(matrix_element), cudaMemcpyHostToDevice) );
    // output image
    checkCuda( cudaMalloc(&d_out_image, TOT_SIZE_NOPAD * sizeof(matrix_element)) );
    checkCuda( cudaMemset(d_out_image, 0, TOT_SIZE_NOPAD * sizeof(matrix_element)) );

    // ===================================== RUN =====================================
    TIMER_DEF;

    if (strcmp(method, "cpu_naive") == 0){
        TIMER_START;
        cpu_convolution_naive(h_in_image, h_K, h_out_image, input_image->W, input_image->H, input_image->C, input_image->PAD, K_DIM);
        TIMER_STOP;
    }
    else if (strcmp(method, "gpu_naive") == 0){
        int th_size_x = 16;
        int th_size_y = 16;
        dim3 numBlocks((input_image->W - K_DIM + 1) / th_size_x, (input_image->H - K_DIM + 1) / th_size_y, 1);
        dim3 dimBlocks(th_size_x, th_size_y, 1);

        TIMER_START;
        gpu_convolution_naive<<<numBlocks, dimBlocks>>>(d_in_image, d_K, d_out_image, input_image->W, input_image->H, input_image->C, input_image->PAD, K_DIM);
        checkCuda( cudaDeviceSynchronize() );
        TIMER_STOP;
        checkCuda( cudaMemcpy(h_out_image, d_out_image, TOT_SIZE_NOPAD*sizeof(matrix_element), cudaMemcpyDeviceToHost) );
    }
    // Print first 10 elements to stderr
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "%f ", h_out_image[i]);
    }
    fprintf(stderr, "\n");

    //Check for errors
    /*
    float error = 0.0;
    for(int y = 0; y < input_image->H; y++){
        for(int x = 0; x < input_image->W; x++){
            int idx = x*IMAGE_DIM_X + y;
            error += (h_out_image[idx] - gpu_output[idx]) < 0 ? -(h_out_image[idx] - gpu_output[idx]) : (h_out_image[idx] - gpu_output[idx]) ;
        }
    }
    float time = TIMER_ELAPSED;
    */

    // ===================================== SHOW RESULTS =====================================
    memcpy(output_image->val, h_out_image, sizeof(matrix_element) * TOT_SIZE_NOPAD);
    write_png("images/output.png", output_image);    

    // ===================================== FREE MEMORY =====================================

    // structs
    del_img(input_image);  
    del_img(output_image);
    // host side
    free(h_in_image);
    free(h_out_image);
    free(h_K);
    // device side
    checkCuda( cudaFree(d_in_image) ); 
    checkCuda( cudaFree(d_out_image) );
    checkCuda( cudaFree(d_K) );
    
    return 0;
}