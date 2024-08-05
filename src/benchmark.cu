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
    int min_powerof2, max_powerof2, min_kernel_size, max_kernel_size, iterations_per_config;
    // parse command line options
    process_benchmark_args(argc, argv, method, &min_powerof2, &max_powerof2, &min_kernel_size, &max_kernel_size, &iterations_per_config);
        
    // === BENCHMARK vars ===

    // benchmark iterator limits
    const int warmup_runs = (int)(0.05 * iterations_per_config) + 1;            // first "warmup_runs" per kernel configuration are used as warmup and not recorded
    int num_size_configurations = max_powerof2 - min_powerof2 + 1;              // number of different input matrix sizes to evaluate
    int num_kernel_configurations = (max_kernel_size - min_kernel_size)/2 + 1;  // number of different kernel sizes to evaluate

    // storage of benchmark values
    double*** exec_time = init_3d_vec(num_kernel_configurations, num_size_configurations, iterations_per_config);           // keep track of measured execution times 
    double*** effective_bandwidth = init_3d_vec(num_kernel_configurations, num_size_configurations, iterations_per_config); // keep track of measured effective bandwidths
    double*** flops = init_3d_vec(num_kernel_configurations, num_size_configurations, iterations_per_config);               // keep track of measured flops

    // other varaibles
    PngImage* img = (PngImage*) malloc(sizeof(PngImage));   // input png image (randomly initialized during bench.)
    PngImage* out = (PngImage*) malloc(sizeof(PngImage));   // output png image

    matrix K;             // kernel filter
    int K_SIZE;           // kernel size (assumed to be squared), K_SIZE = 3 means the kernel is (3x3)
    int TOT_SIZE;         // total size of an image (including padding)
    int TOT_SIZE_NOPAD;   // total size of an image (excluded padding)
    
    long int Br_Bw;            // Bytes-read Bytes-wrote done
    long int Flo;              // Float in point operations done

    int RANDFILL_TILE = (int)pow(2, 4);        // tile for rand_fill() kernel
    
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
            printf("\n   kernel size : %d x %d", K_SIZE, K_SIZE);
            
            // setup fake .png for benchmark
            img->C = out->C = 3;     // in benchmark, number of channels is kept fixed = 3
            img->PAD = (int)(K_SIZE/2); // padding size is always dependent on the choice of kernel
            out->PAD = 0;

            for (i = 0; i < num_size_configurations; i++){

                // set size parameters
                img->W = out->W = (int)pow(2,i+min_powerof2);
                img->H = out->H = img->W;
                TOT_SIZE = (img->W + 2*img->PAD)*(img->H + 2*img->PAD)*img->C;
                TOT_SIZE_NOPAD = img->W * img->H * img->C;
                printf("\n   matrix size : [2^%d x 2^%d] + %d padding ", i+min_powerof2, i+min_powerof2, img->PAD);

                // define metrics
                Br_Bw = sizeof(matrix_element) * ((TOT_SIZE_NOPAD * (K_SIZE*K_SIZE)*(K_SIZE*K_SIZE)*img->C) + TOT_SIZE_NOPAD);
                Flo = TOT_SIZE_NOPAD * (K_SIZE*K_SIZE) * 2;

                for (j = 0; j < (iterations_per_config+warmup_runs); j++){

                    // allocate mem.
                    img->val = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE);         // input image (including padding)
                    out->val = (matrix) malloc(sizeof(matrix_element) * TOT_SIZE_NOPAD);   // output image (which has no padding)
                    K = (matrix) malloc(sizeof(float) * K_SIZE*K_SIZE);                    // kernel filter
                    // fill with random values
                    fill_matrix_random(img->val, TOT_SIZE);
                    fill_matrix_random(K, K_SIZE*K_SIZE);

                    // ---- START ----
                    TIMER_START;
                    cpu_convolution_naive(img, K_SIZE, K, out);
                    TIMER_STOP;
                    // ----- END -----

                    // begin storing values only after "warmup_runs" are done
                    if (j >= warmup_runs){
                        exec_time[k][i][j-warmup_runs] = TIMER_ELAPSED;                                                             // seconds
                        effective_bandwidth[k][i][j-warmup_runs] = (Br_Bw / pow(10,9)) / (exec_time[k][i][j-warmup_runs] + 1e-8);   // GB/s
                        flops[k][i][j-warmup_runs] = (Flo / pow(10,12)) / (exec_time[k][i][j-warmup_runs] + 1e-8);                  // TFLOPS
                    }
                    // free host matrices
                    free(img->val);
                    free(out->val);
                    free(K);
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

    // kernels Variables
    matrix d_mat, d_kernel, filler;

    if (strcmp(method, "gpu_naive") == 0 || strcmp(method, "all") == 0){
        printf("\nComputing statystics for : 'gpu_convolution_naive()' :");
        
        for (k = 0; k < num_kernel_configurations; k++){
            // setup kernel
            // setup kernel
            K_SIZE = min_kernel_size + 2*k;
            printf("\n   kernel size : %d x %d", K_SIZE, K_SIZE);
            
            // setup fake .png for benchmark
            img->C = out->C = 3;     // in benchmark, number of channels is kept fixed = 3
            img->PAD = (int)(K_SIZE/2); // padding size is always dependent on the choice of kernel
            out->PAD = 0;
            
            for (i = 0; i < num_size_configurations; i++){
                // set size parameters
                img->W = out->W = (int)pow(2,i+min_powerof2);
                img->H = out->H = img->W;
                TOT_SIZE = (img->W + 2*img->PAD)*(img->H + 2*img->PAD)*img->C;
                TOT_SIZE_NOPAD = img->W * img->H * img->C;
                printf("\n   matrix size : [2^%d x 2^%d] + %d padding ", i+min_powerof2, i+min_powerof2, img->PAD);

                // define metrics
                Br_Bw = sizeof(matrix_element) * ((TOT_SIZE_NOPAD * (K_SIZE*K_SIZE)*(K_SIZE*K_SIZE)*img->C) + TOT_SIZE_NOPAD);
                Flo = TOT_SIZE_NOPAD * (K_SIZE*K_SIZE) * 2;

                for (j = 0; j < iterations_per_config + warmup_runs; j++){
                    // Allocate space on the DEVICE global memory (both for image and kernel)
                    checkCuda( cudaMalloc((void **)&d_mat, TOT_SIZE * sizeof(matrix_element)) );
                    checkCuda( cudaMalloc((void **)&d_kernel, K_SIZE*K_SIZE * sizeof(matrix_element)) );
                    checkCuda( cudaMemset(d_mat, 0, TOT_SIZE * sizeof(matrix_element)) );
                    checkCuda( cudaMemset(d_kernel, 0, K_SIZE*K_SIZE * sizeof(matrix_element)) );

                    // Fill the matrix with random values
                    gpu_fill_rand(d_mat, TOT_SIZE);
                    gpu_fill_rand(d_kernel, K_SIZE*K_SIZE);

                    // allocate a filler block as large as the L2 cache and access it
                    // that's used to basically flush the L2 cache from previous accesses
                    checkCuda( cudaMalloc((void **)&filler, 2*L2_CACHE_SIZE) );  
                    checkCuda( cudaMemset(filler, 0, 2*L2_CACHE_SIZE) );
                    checkCuda( cudaFree(filler) );

                    // ---- START ----
                    TIMER_START;
                    // -> PUT KERNEL THERE
                    // checkCuda( cudaDeviceSynchronize() );
                    TIMER_STOP;
                    // ----- END -----

                    // begin storing values only after "warmup_runs" are done
                    if (j >= warmup_runs){
                        exec_time[k][i][j-warmup_runs] = TIMER_ELAPSED;                                                             // seconds
                        effective_bandwidth[k][i][j-warmup_runs] = (Br_Bw / pow(10,9)) / (exec_time[k][i][j-warmup_runs] + 1e-8);   // GB/s
                        flops[k][i][j-warmup_runs] = (Flo / pow(10,12)) / (exec_time[k][i][j-warmup_runs] + 1e-8);                  // TFLOPS
                    }

                    /// Free memory
                    checkCuda( cudaFree(d_mat) );
                    checkCuda( cudaFree(d_kernel) );
                }
            }
            printf("\n");
        }

        // Write results on a .csv file
        char output_filename[100] = "";  // filename buffer
        sprintf(output_filename,"data/GPU-conv-naive_%d-to-%d-kernel_%d-to-%d-size_%d-iter.csv", min_kernel_size, max_kernel_size, min_powerof2, max_powerof2, iterations_per_config);
        print_csv(output_filename, exec_time, effective_bandwidth, flops, num_kernel_configurations, num_size_configurations, iterations_per_config, min_kernel_size, max_kernel_size, min_powerof2);
    }

    free(img);
    free(out);
    del_3d_vec(exec_time, num_kernel_configurations, num_size_configurations, iterations_per_config);
    del_3d_vec(effective_bandwidth, num_kernel_configurations, num_size_configurations, iterations_per_config);
    del_3d_vec(flops, num_kernel_configurations, num_size_configurations, iterations_per_config);

    return 0;
}