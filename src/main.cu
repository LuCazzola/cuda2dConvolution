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
#define DEBUG

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
    matrix_element K[] = {1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0};
   
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
    cpu_convolution_naive(image, K_dim, K, cpu_output_image);
    TIMER_STOP;

    printf("CPU Time: %.2f\n", TIMER_ELAPSED);
    write_png("images/cpu_output.png", cpu_output_image);    
    
    PngImage* dev_image;
    float* dev_image_val;

    PngImage* gpu_output;
    float* gpu_output_val;

    PngImage* dev_output;
    matrix dev_K;
    
    // Run parallel convolution
    dim3 dimBlock(image->W+2*image->PAD, image->H+2*image->PAD);
    
    // Move kernel to GPU
    checkCuda( cudaMalloc((void **)&dev_K, K_dim*K_dim*sizeof(matrix_element)) );
    checkCuda( cudaMemcpy(dev_K, K, K_dim*K_dim*sizeof(matrix_element), cudaMemcpyHostToDevice) );
    
    // Move input image to GPU
    checkCuda( cudaMalloc((void **)&dev_image, sizeof(PngImage) ));
    checkCuda( cudaMemset((void*)&dev_image->W, image->W, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&dev_image->H, image->H, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&dev_image->C, image->C, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&dev_image->PAD, image->PAD, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&dev_image->color_type, image->color_type, sizeof(png_byte)) );

    checkCuda( cudaMalloc((void**)&dev_image_val, (image->W+2*image->PAD)*(image->H+2*image->PAD)*image->C*sizeof(matrix_element)));
    checkCuda( cudaMemcpy(dev_image_val, image->val, (image->W+2*image->PAD)*(image->H+2*image->PAD)*image->C*sizeof(matrix_element), cudaMemcpyHostToDevice ));
    checkCuda( cudaMemcpy(&(dev_image->val), &dev_image_val, sizeof(float*), cudaMemcpyHostToDevice) );
    
    // Move output buffer to GPU
    checkCuda( cudaMalloc((void **)&gpu_output, sizeof(PngImage) ));
    checkCuda( cudaMemset((void*)&gpu_output->W, image->W, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&gpu_output->H, image->H, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&gpu_output->C, image->C, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&gpu_output->PAD, image->PAD, sizeof(unsigned int)) );
    checkCuda( cudaMemset((void*)&gpu_output->color_type, image->color_type, sizeof(png_byte)) );
    
    checkCuda( cudaMalloc((void**)&gpu_output_val, image->W*image->H*image->C*sizeof(matrix_element)));
    checkCuda( cudaMemset((void*)&gpu_output_val, 0, image->W*image->H*image->C*sizeof(matrix_element), cudaMemcpyHostToDevice ));
    checkCuda( cudaMemcpy(&(gpu_output->val), &gpu_output_val, sizeof(float*), cudaMemcpyHostToDevice) );

    // Allocate host output buffer
    dev_output = (PngImage*) malloc(sizeof(PngImage)); 
    dev_output->PAD = image->PAD;
    dev_output->C = image->C;
    dev_output->W = image->W;
    dev_output->H = image->H;
    dev_output->color_type = image->color_type;
    dev_output->val = (matrix) malloc(sizeof(matrix_element)*image->W*image->H*image->C);

    TIMER_START;
    gpu_convolution_naive<<<1, dimBlock>>>(dev_image, K_dim, dev_K, gpu_output);
    checkCuda( cudaDeviceSynchronize() );
    TIMER_STOP;
    checkCuda( cudaMemcpy(dev_output, (void*)gpu_output, sizeof(PngImage), cudaMemcpyDeviceToHost) );

    //Check for errors
    float error = 0.0;
    for(int y = 0; y < image->H; y++){
        for(int x = 0; x < image->W; x++){
	     int idx = x*image->W+y;
             error += (cpu_output_image->val[idx] - dev_output->val[idx]) < 0 ? -(cpu_output_image->val[idx] - dev_output->val[idx]) : (cpu_output_image->val[idx] - dev_output->val[idx]) ;
        }
    }
    float time = TIMER_ELAPSED;

    // ===================================== FREE MEMORY =====================================

    free(image);  
    free(cpu_output_image);  
    free(dev_output);
    checkCuda( cudaFree(gpu_output) );
    checkCuda( cudaFree(dev_image) ); 

    return 0;
}
