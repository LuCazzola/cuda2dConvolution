extern "C" { 
    #include "headers/matrix.h"
	#include "headers/common.h"
    #include "headers/opt_parser.h"
    #include "headers/pngUtils.h"
}
#include "headers/convolution.h"


void print_metrics (double exec_time, const int W, const int H, const int C, const int K_DIM){
    // metrics evaluation
    printf("\n\n========================== METRICS ==========================\n");

    long int Br_Bw = get_conv_bytes_read_write(W, H, C, K_DIM);
    long int Flo = get_conv_flops(W, H, C, K_DIM);

    // effective bandwidth (expressed in GB/s)
    double effective_bandwidth = ((double)Br_Bw / (double)pow(10,9)) / (exec_time + 1e-8);
    // flops (expressed in TFLOP/s)
    double flops = ((double)Flo / (double)pow(10,12)) / (exec_time + 1e-8); 

    // print out values
    printf("\nExecution time      :  %f s", exec_time);
    printf("\nEffective Bandwidth :  %f GB/s", effective_bandwidth);
    printf("\nFLOPS               :  %f TFLOP/s\n", flops);
    printf("\n");
}


void print_run_infos(char *method, const int W, const int H, const int C, const int K_DIM, const int th_size_x, const int th_size_y){
    printf("\n========================== RUN INFO ==========================\n");

    printf("\n-   datatype        : %s\n", VALUE(MATRIX_ELEM_DTYPE));
    printf("-   png size        : (%d x %d x %d)\n\n", W, H, C);
    
    printf("-   Method:         : %s()\n", method);
    printf("-   kernel filter   : (%d x %d)\n", K_DIM, K_DIM);
    printf("-   Block dim       : (%d x %d)\n", th_size_x, th_size_y);
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
    int K_DIM, TH_SIZE_X, TH_SIZE_Y;
    K_DIM = TH_SIZE_X = TH_SIZE_Y = 0;
    char input_png_path [50];
    char output_png_path [50];
    char method [50];

    process_main_args (argc, argv, method, input_png_path, output_png_path, &K_DIM, &TH_SIZE_X, &TH_SIZE_Y);

    // get input image
    PngImage* input_image = read_png(input_png_path);

    const int TOT_SIZE = input_image->W * input_image->H * input_image->C;
    const int TOT_K_DIM = K_DIM*K_DIM; 
    TH_SIZE_X = (int) pow(2, TH_SIZE_X);
    TH_SIZE_Y = (int) pow(2, TH_SIZE_Y);

    // struct on which to store the output image once it's computed
    PngImage* output_image = (PngImage*) malloc(sizeof(PngImage));
    output_image->W = input_image->W;
    output_image->H = input_image->H;
    output_image->C = input_image->C;
    output_image->color_type = input_image->color_type;
    output_image->val = (matrix) malloc(sizeof(matrix_element)* TOT_SIZE);

    // Show run infos
    print_run_infos(method, input_image->W, input_image->H, input_image->C, K_DIM, TH_SIZE_X, TH_SIZE_Y);

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
    h_out_image = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE);

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
    checkCuda( cudaMalloc(&d_out_image, TOT_SIZE * sizeof(matrix_element)) );
    checkCuda( cudaMemset(d_out_image, 0, TOT_SIZE * sizeof(matrix_element)) );

    // "Wake up" the GPU before executing the kernels
    dim3 blockSizeWarmup(512, 1, 1);
    dim3 gridSizeWarmup(256, 1, 1);
    warm_up_gpu<<<gridSizeWarmup, blockSizeWarmup>>> ();
    checkCuda( cudaDeviceSynchronize() );

    // ===================================== RUN =====================================
    TIMER_DEF;

    if (strcmp(method, "cpu_naive") == 0){
        TIMER_START;
        cpu_convolution_naive(h_in_image, h_K, h_out_image, input_image->W, input_image->H, input_image->C, K_DIM);
        TIMER_STOP;
    }
    else if (strcmp(method, "gpu_naive") == 0){
        const int BLOCK_X = (int) (input_image->W + 1) / TH_SIZE_X; 
        const int BLOCK_Y = (int) (input_image->H + 1) / TH_SIZE_Y;
        dim3 numBlocks(BLOCK_X, BLOCK_Y, 1);
        dim3 dimBlocks(TH_SIZE_X, TH_SIZE_Y, 1);

        TIMER_START;
        gpu_convolution_naive<<<numBlocks, dimBlocks>>>(d_in_image, d_K, d_out_image, input_image->W, input_image->H, input_image->C, K_DIM);
        checkCuda( cudaDeviceSynchronize() );
        TIMER_STOP;
        checkCuda( cudaMemcpy(h_out_image, d_out_image, TOT_SIZE*sizeof(matrix_element), cudaMemcpyDeviceToHost) );
    }
    else if (strcmp(method, "gpu_shared") == 0){
        const int BLOCK_X = (int) ((input_image->W) + 1) / TH_SIZE_X; 
        const int BLOCK_Y = (int) ((input_image->H) + 1) / TH_SIZE_Y;
        dim3 numBlocks(BLOCK_X, BLOCK_Y, 1);
        dim3 dimBlocks(TH_SIZE_X, TH_SIZE_Y, 1);
        size_t shared_mem_size = TH_SIZE_X * TH_SIZE_Y * input_image->C * sizeof(matrix_element);

        TIMER_START;
        gpu_convolution_shared<<<numBlocks, dimBlocks, shared_mem_size>>>(d_in_image, d_K, d_out_image, input_image->W, input_image->H, input_image->C, 0, K_DIM);
        checkCuda( cudaDeviceSynchronize() );
        TIMER_STOP;
        checkCuda( cudaMemcpy(h_out_image, d_out_image, TOT_SIZE*sizeof(matrix_element), cudaMemcpyDeviceToHost) );
    }

    // ===================================== SHOW RESULTS =====================================
    
    // print metrics
    print_metrics (TIMER_ELAPSED, output_image->W, output_image->H, output_image->C, K_DIM);
    // save output image
    memcpy(output_image->val, h_out_image, sizeof(matrix_element) * TOT_SIZE);
    write_png(output_png_path, output_image);
    
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