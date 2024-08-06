extern "C" { 
	#include "headers/common.h"
	#include "headers/opt_parser.h"
	#include "headers/matrix.h"
    #include "headers/pngUtils.h"
}
#include "headers/cudaUtils.h"
#include "headers/convolution.h"

// File specific utility functions
double mean(double X [], const int SIZE){
    double sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += X[i];
    }
    return (sum / SIZE);
}
double stdev(double X [], double mean, const int SIZE){
    double sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += pow(X[i] - mean, 2);
    }
    return sqrt(sum / SIZE);
}

double*** init_3d_vec(const int A, const int B, const int C){
    double ***vec = (double***)malloc(sizeof(double**) * A);
    for (int i = 0; i < A; i++) {
        vec[i] = (double**) malloc(sizeof(double*) * B);
        for (int j = 0; j < B; j++) {
            vec[i][j] = (double*) malloc(sizeof(double) * C);
        }
    }
    return vec;
}
void del_3d_vec(double*** vec, const int A, const int B, const int C){
    for (int i = 0; i < A; i++){
        for (int j = 0; j < B; j++){
            free(vec[i][j]);
        }
        free(vec[i]);
    }
    free(vec);
}

void print_csv(char* output_filename, double*** exec_time, double*** effective_bandwidth, double*** flops,
            int num_kernel_configurations, int num_size_configurations, int iterations_per_config, int min_kernel_size, int max_kernel_size, int min_powerof2){

    FILE* fp = fopen(output_filename, "w");
    // Additional information about the data
    fprintf(fp,"Additional run info,,Matrix element datatype,%s,kernel_size,%dx%d to %dx%d\n\n", VALUE(MATRIX_ELEM_DTYPE), min_kernel_size, min_kernel_size, max_kernel_size, max_kernel_size);

    // print column ID's
    fprintf(fp, "kernel_size,matrix_size");
    for (int j = 0; j < iterations_per_config; j++){
        fprintf(fp, ",exec_time-%d,effective_bandwidth-%d,flops-%d", j, j, j);
    }
    fprintf(fp, ",,mean_exec_time,stdev_exec_time,mean_effective_bandwidth,stdev_effective_bandwidth,mean_flops,stdev_flops\n");

    double mean_et, mean_eb, mean_flops, stdev_et, stdev_eb, stdev_flops;

    // print data
    for (int k = 0; k < num_kernel_configurations; k++){
        for (int i = 0; i < num_size_configurations; i++){
            fprintf(fp, "%dx%d", min_kernel_size + 2*k, min_kernel_size + 2*k);
            fprintf(fp, ",2^{%d}", (i+min_powerof2)*2);

            for (int j = 0; j < iterations_per_config; j++){
                fprintf(fp, ",%f,%f,%f", exec_time[k][i][j], effective_bandwidth[k][i][j], flops[k][i][j]);
            }
            
            mean_et = mean(exec_time[k][i], iterations_per_config);
            stdev_et = stdev(exec_time[k][i], mean_et, iterations_per_config);

            mean_eb = mean(effective_bandwidth[k][i], iterations_per_config);
            stdev_eb = stdev(effective_bandwidth[k][i], mean_eb, iterations_per_config);

            mean_flops = mean(flops[k][i], iterations_per_config);
            stdev_flops = stdev(flops[k][i], mean_flops, iterations_per_config);
            
            fprintf(fp, ",,%f,%f,%f,%f,%f,%f\n",mean_et, stdev_et, mean_eb, stdev_eb, mean_flops, stdev_flops);
        }
    }

    fclose(fp);
}


// NOTE : as this file's role is to only benchmark the kernels there's no need to initialize host matrix
//        I do this to make the benchmarking faster (filling the host matrix randomly is really slow)
int main(int argc, char * argv []){

    // ===================================== Parameters Setup =====================================

    // options to set in "run_benchmark.sh" file
    char method [30];
    int min_powerof2, max_powerof2, min_kernel_size, max_kernel_size, iterations_per_config, TH_SIZE_X, TH_SIZE_Y;
    // parse command line options
    process_benchmark_args(argc, argv, method, &min_powerof2, &max_powerof2, &min_kernel_size, &max_kernel_size, &iterations_per_config, &TH_SIZE_X, &TH_SIZE_Y);
        
    // === BENCHMARK vars ===

    // benchmark iterator limits
    const int warmup_runs = (int)(0.05 * iterations_per_config) + 1;            // first "warmup_runs" per kernel configuration are used as warmup and not recorded
    int num_size_configurations = max_powerof2 - min_powerof2 + 1;              // number of different input matrix sizes to evaluate
    int num_kernel_configurations = (max_kernel_size - min_kernel_size)/2 + 1;  // number of different kernel sizes to evaluate

    // storage of benchmark values
    double*** exec_time = init_3d_vec(num_kernel_configurations, num_size_configurations, iterations_per_config);           // keep track of measured execution times 
    double*** effective_bandwidth = init_3d_vec(num_kernel_configurations, num_size_configurations, iterations_per_config); // keep track of measured effective bandwidths
    double*** flops = init_3d_vec(num_kernel_configurations, num_size_configurations, iterations_per_config);               // keep track of measured flops

    // fake images
    matrix h_in_img, h_out_img, h_k;
    matrix d_in_img, d_out_img, d_k, filler;

    int K_SIZE;           // kernel size (assumed to be squared), K_SIZE = 3 means the kernel is (3x3)
    int TOT_SIZE;         // total size of an image (including padding)
    int TOT_SIZE_NOPAD;   // total size of an image (excluded padding)
    int TOT_K_SIZE;
    int W, H, C, PAD;

    long int Br_Bw;            // Bytes-read Bytes-wrote done
    long int Flo;              // Float in point operations done

    int BLOCK_X, BLOCK_Y;
    TH_SIZE_X = (int) pow(2, TH_SIZE_X);
    TH_SIZE_Y = (int) pow(2, TH_SIZE_Y);

    /*
        ================================================================================================================
        ===================================== PERFORM TEST CPU_convolution_naive() =====================================
        ================================================================================================================
    */

    TIMER_DEF;      // start timer
    int k,i,j;      // loops iterators
    if (strcmp(method, "cpu_naive") == 0 || strcmp(method, "all") == 0){
        printf("\nComputing statystics for : 'cpu_convolution_naive()' :");

        for (k = 0; k < num_kernel_configurations; k++){
            // setup kernel
            K_SIZE = min_kernel_size + 2*k;
            TOT_K_SIZE = K_SIZE*K_SIZE;
            printf("\n   kernel size : %d x %d", K_SIZE, K_SIZE);
            
            // setup fake .png for benchmark
            C = 3;                 // in benchmark, number of channels is kept fixed = 3
            PAD = (int)(K_SIZE/2); // padding size is always dependent on the choice of kernel

            for (i = 0; i < num_size_configurations; i++){

                // set size parameters
                W = (int)pow(2,i+min_powerof2);
                H = W;
                TOT_SIZE = (W + 2*PAD)*(H + 2*PAD)*C;
                TOT_SIZE_NOPAD = W * H * C;
                printf("\n   matrix size : [%d x %d x %d] + %d pad ", W, H, C, PAD);

                // define metrics
                Br_Bw = get_conv_bytes_read_write(W, H, C, PAD, K_SIZE);
                Flo = get_conv_flops(W, H, C, PAD, K_SIZE);

                for (j = 0; j < (iterations_per_config+warmup_runs); j++){

                    // allocate mem.
                    h_in_img = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE);         // input image (including padding)
                    h_out_img = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE_NOPAD);   // output image (which has no padding)
                    h_k = (matrix) malloc(sizeof(float) * TOT_K_SIZE);                    // kernel filter
                    // fill with random values
                    fill_matrix_random(h_in_img, TOT_SIZE);
                    fill_matrix_random(h_k, TOT_K_SIZE);

                    // ---- START ----
                    TIMER_START;
                    cpu_convolution_naive(h_in_img, h_k, h_out_img, W, H, C, PAD, K_SIZE);
                    TIMER_STOP;
                    // ----- END -----

                    // begin storing values only after "warmup_runs" are done
                    if (j >= warmup_runs){
                        exec_time[k][i][j-warmup_runs] = TIMER_ELAPSED;                                                             // seconds
                        effective_bandwidth[k][i][j-warmup_runs] = (Br_Bw / pow(10,9)) / (exec_time[k][i][j-warmup_runs] + 1e-8);   // GB/s
                        flops[k][i][j-warmup_runs] = (Flo / pow(10,12)) / (exec_time[k][i][j-warmup_runs] + 1e-8);                  // TFLOPS
                    }
                    // free host matrices
                    free(h_in_img);
                    free(h_out_img);
                    free(h_k);
                }
            }
            printf("\n");
        }

        // Write results on a .csv file
        char output_filename[100] = "";  // filename buffer
        sprintf(output_filename,"data/CPU-conv-naive_%d-to-%d-kernel_%d-to-%d-size_%d-iter.csv", min_kernel_size, max_kernel_size, min_powerof2, max_powerof2, iterations_per_config);
        print_csv(output_filename, exec_time, effective_bandwidth, flops, num_kernel_configurations, num_size_configurations, iterations_per_config, min_kernel_size, max_kernel_size, min_powerof2);
    }

    /*
        ================================================================================================================
        ===================================== PERFORM TEST GPU_convolution_naive() =====================================
        ================================================================================================================
    */

    if (strcmp(method, "gpu_naive") == 0 || strcmp(method, "all") == 0){
        printf("\nComputing statystics for : 'gpu_convolution_naive()' :");
        
        for (k = 0; k < num_kernel_configurations; k++){
            // setup kernel
            K_SIZE = min_kernel_size + 2*k;
            TOT_K_SIZE = K_SIZE*K_SIZE;
            printf("\n   kernel size : %d x %d", K_SIZE, K_SIZE);
            
            // setup fake .png for benchmark
            C = 3;     // in benchmark, number of channels is kept fixed = 3
            PAD = (int)(K_SIZE/2); // padding size is always dependent on the choice of kernel
            
            for (i = 0; i < num_size_configurations; i++){
                // set size parameters
                W = (int)pow(2, i+min_powerof2);
                H = W;
                TOT_SIZE = (W + 2*PAD)*(H + 2*PAD)*C;
                TOT_SIZE_NOPAD = W * H * C;
                printf("\n   matrix size : [%d x %d x %d] + %d pad ", W, H, C, PAD);

                // define metrics
                Br_Bw = get_conv_bytes_read_write(W, H, C, PAD, K_SIZE);
                Flo = get_conv_flops(W, H, C, PAD, K_SIZE);

                BLOCK_X = (int) ((W + 2*PAD) + 1) / TH_SIZE_X; 
                BLOCK_Y = (int) ((H + 2*PAD) + 1) / TH_SIZE_Y;
                dim3 numBlocks(BLOCK_X, BLOCK_Y, 1);
                dim3 dimBlocks(TH_SIZE_X, TH_SIZE_Y, 1);

                for (j = 0; j < iterations_per_config + warmup_runs; j++){
                    // Allocate space on the DEVICE global memory (both for image and kernel)
                    checkCuda( cudaMalloc((void **)&d_in_img, TOT_SIZE * sizeof(matrix_element)) );
                    checkCuda( cudaMalloc((void **)&d_out_img, TOT_SIZE_NOPAD * sizeof(matrix_element)) );
                    checkCuda( cudaMalloc((void **)&d_k, TOT_K_SIZE * sizeof(matrix_element)) );
                    checkCuda( cudaMemset(d_in_img, 0, TOT_SIZE * sizeof(matrix_element)) );
                    checkCuda( cudaMemset(d_out_img, 0, TOT_SIZE_NOPAD * sizeof(matrix_element)) );
                    checkCuda( cudaMemset(d_k, 0, TOT_K_SIZE * sizeof(matrix_element)) );

                    // Fill the matrix with random values
                    gpu_fill_rand(d_in_img, TOT_SIZE);
                    gpu_fill_rand(d_k, TOT_K_SIZE);

                    // allocate a filler block as large as the L2 cache and access it
                    // that's used to basically flush the L2 cache from previous accesses
                    checkCuda( cudaMalloc((void **)&filler, 2*L2_CACHE_SIZE) );  
                    checkCuda( cudaMemset(filler, 0, 2*L2_CACHE_SIZE) );
                    checkCuda( cudaFree(filler) );

                    // ---- START ----
                    TIMER_START;
                    gpu_convolution_naive<<<numBlocks, dimBlocks>>>(d_in_img, d_k, d_out_img, W, H, C, PAD, K_SIZE);
                    checkCuda( cudaDeviceSynchronize() );
                    TIMER_STOP;
                    // ----- END -----

                    // begin storing values only after "warmup_runs" are done
                    if (j >= warmup_runs){
                        exec_time[k][i][j-warmup_runs] = TIMER_ELAPSED;                                                             // seconds
                        effective_bandwidth[k][i][j-warmup_runs] = (Br_Bw / pow(10,9)) / (exec_time[k][i][j-warmup_runs] + 1e-8);   // GB/s
                        flops[k][i][j-warmup_runs] = (Flo / pow(10,12)) / (exec_time[k][i][j-warmup_runs] + 1e-8);                  // TFLOPS
                    }

                    /// Free memory
                    checkCuda( cudaFree(d_in_img) );
                    checkCuda( cudaFree(d_out_img) );
                    checkCuda( cudaFree(d_k) );
                }
            }
            printf("\n");
        }

        // Write results on a .csv file
        char output_filename[100] = "";  // filename buffer
        sprintf(output_filename,"data/GPU-conv-naive_%d-to-%d-kernel_%d-to-%d-size_%d-by-%d-th-per-block_%d-iter.csv", min_kernel_size, max_kernel_size, min_powerof2, max_powerof2, TH_SIZE_X, TH_SIZE_Y, iterations_per_config);
        print_csv(output_filename, exec_time, effective_bandwidth, flops, num_kernel_configurations, num_size_configurations, iterations_per_config, min_kernel_size, max_kernel_size, min_powerof2);
    }

    del_3d_vec(exec_time, num_kernel_configurations, num_size_configurations, iterations_per_config);
    del_3d_vec(effective_bandwidth, num_kernel_configurations, num_size_configurations, iterations_per_config);
    del_3d_vec(flops, num_kernel_configurations, num_size_configurations, iterations_per_config);

    return 0;
}