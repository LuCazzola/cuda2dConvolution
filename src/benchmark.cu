extern "C" { 
	#include "headers/common.h"
	#include "headers/opt_parser.h"
	#include "headers/matrix.h"
}
#include "headers/transpose_gpu.h"
#include "headers/common_cuda.h"

// functions to compute metrics
double mean(double X [], const int SIZE){
    double sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += X[i];
    }
    return (sum/SIZE);
}
double stdev(double X [], double mean, const int SIZE){
    double sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += pow(X[i] - mean, 2);
    }
    return sqrt(sum/SIZE);
}

// NOTE : as this file's role is to only benchmark the kernels there's no need to initialize host matrix
//        I do this to make the benchmarking faster (filling the host matrix randomly is really slow)
int main(int argc, char * argv []){

    // ===================================== Parameters Setup =====================================

    // options to set in "benchmark_gpu.sh" file
    char method [30];
    int min_powerof2, max_powerof2, iterations_per_config, block_size, th_size_x, th_size_y;
    // parse command line options
    process_benchmark_gpu_options(argc, argv, method, &min_powerof2, &max_powerof2, &iterations_per_config, &block_size, &th_size_x, &th_size_y);

    // kernels Variables
    const int BLK_SIZE = (int) pow(2, block_size);               // matrix block size (same concept as Tiles)
    const int THREAD_DIM_X = (int) pow(2, th_size_x);            // defines thread blockDim.x
    const int THREAD_DIM_Y = (int) pow(2, th_size_y);            // defines thread blockDim.y
    int SIZE;
    size_t sharedBlockSize;
    matrix d_mat, filler;
    
    // === BENCHMARK vars ===
    const int warmup_runs = (int)(0.05 * iterations_per_config);  // first "warmup_runs" per kernel configuration are used as warmup and not recorded
    int num_configurations = max_powerof2 - min_powerof2 + 1;  // number of different data settings
    double Br_Bw;  // Bytes-read Bytes-wrote by the application    
    double exec_time [num_configurations][iterations_per_config];  // keep track of measured execution times 
    double effective_bandwidth [num_configurations][iterations_per_config]; // keep track of measured effective bandwidths
    double mean_et, stdev_et, mean_eb, stdev_eb;
    FILE *fp;
    char output_filename[100] = "";
    // lower and upper bound for generate_in_a_b() kernel
    const float A = FLT_MIN/2,  B = FLT_MAX/2; 

    TIMER_DEF;
    // ===================================== PERFORM TEST transpose_blocks_gpu_coalesced() =====================================

    if (strcmp(method, "blocks_coalesced") == 0 || strcmp(method, "all") == 0) {
        printf("\nComputing statystics of : 'transpose_blocks_gpu_coalesced()' :");

        for (int i = 0; i < num_configurations; i++){
            SIZE = (int)pow(2,i+min_powerof2);

            // Wake up the GPU before executing the main kernel
            dim3 gridSize((int)(SIZE / BLK_SIZE), (int)(SIZE / BLK_SIZE), 1);
            dim3 blockSize(THREAD_DIM_X, THREAD_DIM_Y, 1);
            warm_up_gpu <<<gridSize, blockSize >>>();  
            checkCuda( cudaDeviceSynchronize() );

            printf("\n   matrix size : 2^%d x 2^%d", i+min_powerof2, i+min_powerof2);
            for (int j = 0; j < iterations_per_config + warmup_runs; j++){
                // Allocate space on the DEVICE global memory
                checkCuda( cudaMalloc((void **)&d_mat, SIZE*SIZE * sizeof(matrix_element)) );
                checkCuda( cudaMemset(d_mat, 0, SIZE*SIZE*sizeof(matrix_element)) );
                // Fill the matrix with random values
                gpu_fill_rand(d_mat, SIZE);
                generate_in_a_b <<<gridSize, blockSize>>> (d_mat, A, B, SIZE, BLK_SIZE, SIZE*SIZE);  
                checkCuda( cudaDeviceSynchronize() );

                // allocate a filler block as large as the L2 cache and access it
                // that's used to basically flush the L2 cache from previous accesses
                checkCuda( cudaMalloc((void **)&filler, 2*L2_CACHE_SIZE) );  
                checkCuda( cudaMemset(filler, 0, 2*L2_CACHE_SIZE) );
                checkCuda( cudaFree(filler) );

                // ---- START ----
                sharedBlockSize = 2 * BLK_SIZE*BLK_SIZE;
                TIMER_START;
                transpose_blocks_gpu_coalesced <<<gridSize, blockSize, sharedBlockSize*sizeof(matrix_element)>>> (d_mat, SIZE, BLK_SIZE, SIZE*SIZE);
                checkCuda( cudaDeviceSynchronize() );
                TIMER_STOP;
                // ----- END -----

                // Free memory
                checkCuda( cudaFree(d_mat) );

                // Compute parameters after "warmup_runs" are done
                if (j >= warmup_runs){
                    exec_time[i][j-warmup_runs] = TIMER_ELAPSED;
                    Br_Bw = sizeof(matrix_element) * SIZE*SIZE * 2;
                    effective_bandwidth[i][j-warmup_runs] = ( Br_Bw / pow(10,9) ) / exec_time[i][j-warmup_runs];
                }
            }
        }
        printf("\n");

        // ========== WRITE CSV FILE ==========

        // Write results on a .csv file
        sprintf(output_filename,"data/blocks-gpu-coalesced_%d-to-%d-steps_%d-blocksize_%d-thX_%d-thY.csv",min_powerof2,max_powerof2, block_size, th_size_x, th_size_y);
        fp = fopen(output_filename, "w");
        
        // Additional information about the data
        fprintf(fp,"Additional run info,,Matrix element datatype,%s,block_size,%d,th_size_x,%d,th_size_y,%d\n", VALUE(MATRIX_ELEM_DTYPE), block_size, th_size_x, th_size_y);

        // print column ID's
        fprintf(fp, "\nmatrix_size");
        for (int j = 0; j < iterations_per_config; j++){
            fprintf(fp, ",exec_time-%d,effective_bandwidth-%d", j, j);
        }
        fprintf(fp, ",mean_exec_time,stdev_exec_time,mean_effective_bandwidth,stdev_effective_bandwidth\n");

        // print data
        for (int i = 0; i < num_configurations; i++){
            fprintf(fp, "2^%d", i+min_powerof2);

            for (int j = 0; j < iterations_per_config; j++){
                fprintf(fp, ",%f,%f", exec_time[i][j], effective_bandwidth[i][j]);
            }
            
            mean_et = mean(exec_time[i], iterations_per_config);
            mean_eb = mean(effective_bandwidth[i], iterations_per_config);
            stdev_et = stdev(exec_time[i], mean_et, iterations_per_config);
            stdev_eb = stdev(effective_bandwidth[i], mean_eb, iterations_per_config);
            
            fprintf(fp, ",%f,%f,%f,%f\n",mean_et, stdev_et, mean_eb, stdev_eb);
        }
        fclose(fp);
    }



    // ===================================== PERFORM TEST transpose_blocks_gpu() (NAIVE non-coalesced) =====================================

    if (strcmp(method, "blocks_naive") == 0 || strcmp(method, "all") == 0) {
        printf("\nComputing statystics of : 'transpose_blocks_gpu()' :");

        for (int i = 0; i < num_configurations; i++){
            SIZE = (int)pow(2,i+min_powerof2);
            
            // Wake up the GPU before executing the main kernel
            dim3 gridSize((int)(SIZE / BLK_SIZE), (int)(SIZE / BLK_SIZE), 1);
            dim3 blockSize(THREAD_DIM_X, THREAD_DIM_Y, 1);
            warm_up_gpu <<<gridSize, blockSize >>>();  
            checkCuda( cudaDeviceSynchronize() );

            printf("\n   matrix size : 2^%d x 2^%d", i+min_powerof2, i+min_powerof2);
            for (int j = 0; j < iterations_per_config + warmup_runs; j++){
                // Allocate space on the DEVICE global memory
                checkCuda( cudaMalloc((void **)&d_mat, SIZE*SIZE * sizeof(matrix_element)) );
                checkCuda( cudaMemset(d_mat, 0, SIZE*SIZE*sizeof(matrix_element)) );
                // Fill the matrix with random values
                gpu_fill_rand(d_mat, SIZE);
                generate_in_a_b <<<gridSize, blockSize>>> (d_mat, A, B, SIZE, BLK_SIZE, SIZE*SIZE);  
                checkCuda( cudaDeviceSynchronize() );

                // allocate a filler block as large as the L2 cache and access it
                // that's used to basically flush the L2 cache from previous accesses
                checkCuda( cudaMalloc((void **)&filler, 2*L2_CACHE_SIZE) );  
                checkCuda( cudaMemset(filler, 0, 2*L2_CACHE_SIZE) );
                checkCuda( cudaFree(filler) );
                
                // ---- START ----
                TIMER_START;
                transpose_blocks_gpu <<<gridSize, blockSize>>> (d_mat, SIZE, BLK_SIZE, SIZE*SIZE);
                checkCuda( cudaDeviceSynchronize() );
                TIMER_STOP;
                // ----- END -----

                // Free DEVICE memory
                checkCuda( cudaFree(d_mat) );

                // Compute parameters after "warmup_runs" are done
                if (j >= warmup_runs){
                    exec_time[i][j-warmup_runs] = TIMER_ELAPSED;
                    Br_Bw = sizeof(matrix_element) * SIZE*SIZE * 2;
                    effective_bandwidth[i][j-warmup_runs] = ( Br_Bw / pow(10,9) ) / exec_time[i][j-warmup_runs];
                }
            }
        }
        printf("\n");

        // ========== WRITE CSV FILE ==========

        // Write results on a .csv file
        sprintf(output_filename,"data/blocks-gpu_%d-to-%d-steps_%d-blocksize_%d-thX_%d-thY.csv",min_powerof2,max_powerof2, block_size, th_size_x, th_size_y);
        fp = fopen(output_filename, "w");
        
        // Additional information about the data
        fprintf(fp,"Additional run info,,Matrix element datatype,%s,block_size,%d,th_size_x,%d,th_size_y,%d\n", VALUE(MATRIX_ELEM_DTYPE), block_size, th_size_x, th_size_y);

        // print column ID's
        fprintf(fp, "\nmatrix_size");
        for (int j = 0; j < iterations_per_config; j++){
            fprintf(fp, ",exec_time-%d,effective_bandwidth-%d", j, j);
        }
        fprintf(fp, ",mean_exec_time,stdev_exec_time,mean_effective_bandwidth,stdev_effective_bandwidth\n");

        // print data
        for (int i = 0; i < num_configurations; i++){
            fprintf(fp, "2^%d", i+min_powerof2);

            for (int j = 0; j < iterations_per_config; j++){
                fprintf(fp, ",%f,%f", exec_time[i][j], effective_bandwidth[i][j]);
            }
            
            mean_et = mean(exec_time[i], iterations_per_config);
            mean_eb = mean(effective_bandwidth[i], iterations_per_config);
            stdev_et = stdev(exec_time[i], mean_et, iterations_per_config);
            stdev_eb = stdev(effective_bandwidth[i], mean_eb, iterations_per_config);
            
            fprintf(fp, ",%f,%f,%f,%f\n",mean_et, stdev_et, mean_eb, stdev_eb);
        }
        fclose(fp);
    }

    return 0;
}