#include "headers/opt_parser.h"

// Function to process the provided options
void process_transpose_options(int argc, char *argv[], char *method, int *N, int *block_size) {
    
    // Long options structure
    static struct option long_options[] = {
        {"N", required_argument, 0, 0},
        {"method", required_argument, 0, 0},
        {"block_size", optional_argument, 0, 0},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (c) {
            case 0: // Long option found
                if (strcmp(long_options[option_index].name, "N") == 0) {
                    *N = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "method") == 0){

                    if (strcmp(optarg, "naive") == 0 || strcmp(optarg, "blocks") == 0){
                        strcpy(method, optarg);
                    }
                    else{
                        fprintf(stderr, "Unsupported 'method' value\n");
                        exit(EXIT_FAILURE);
                    }
                } else if (strcmp(long_options[option_index].name, "block_size") == 0) {
                    if (optarg != NULL) {
                        *block_size = (int)atoi(optarg);
                    }
                }

                break;
            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }
}


// Function to process the provided options
void process_transpose_gpu_options(int argc, char *argv[], char *method, int *N, int *block_size, int *th_size_x, int *th_size_y) {
    
    // Long options structure
    static struct option long_options[] = {
        {"method", required_argument, 0, 0},
        {"N", required_argument, 0, 0},
        {"block_size", required_argument, 0, 0},
        {"th_size_x", required_argument, 0, 0},
        {"th_size_y", required_argument, 0, 0},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (c) {
            case 0: // Long option found
                if (strcmp(long_options[option_index].name, "N") == 0) {
                    *N = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "method") == 0){
                    if (strcmp(optarg, "blocks_naive") == 0 || strcmp(optarg, "blocks_coalesced") == 0){
                        strcpy(method, optarg);
                    }
                    else{
                        fprintf(stderr, "Unsupported 'method' value\n");
                        exit(EXIT_FAILURE);
                    }
                } else if (strcmp(long_options[option_index].name, "block_size") == 0) {
                    *block_size = (int)atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_x") == 0) {
                    *th_size_x = (int)atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_y") == 0) {
                    *th_size_y = (int)atoi(optarg);
                }
                break;
            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }
    
    if (*th_size_x > *block_size || *th_size_y > *block_size){
        fprintf(stderr, "===\nERROR: th_size_x & th_size_y CAN'T exceed block_size\n===\n");
        exit(EXIT_FAILURE);
    }
    if (*th_size_x + *th_size_y > 11){
        fprintf(stderr, "===\nERROR: unpredictable behaviour\ntotal number of threads per block can't exceed 1024\nplease set 'th_size_x' + 'th_size_y' <= 11\n===\n");
        exit(EXIT_FAILURE);
    }
    if (*block_size > 6){
        fprintf(stderr, "===\nERROR: shared memory overflow\nplease set 'block_size' <= 6\n===\n");
        exit(EXIT_FAILURE);
    }
}


// Function to process the provided options
void process_benchmark_options(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* iterations_per_config, int* block_size) {
    
    // Long options structure
    static struct option long_options[] = {
        {"min_powerof2", required_argument, 0, 0},
        {"max_powerof2", required_argument, 0, 0},
        {"iterations_per_config", required_argument, 0, 0},
        {"method", required_argument, 0, 0},
        {"block_size", optional_argument, 0, 0},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (c) {
            case 0: // Long option found
                if (strcmp(long_options[option_index].name, "min_powerof2") == 0) {
                    *min_powerof2 = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "max_powerof2") == 0) {
                    *max_powerof2 = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "iterations_per_config") == 0) {
                    *iterations_per_config = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "method") == 0){

                    if (strcmp(optarg, "naive") == 0 || strcmp(optarg, "blocks") == 0 || strcmp(optarg, "all") == 0){
                        strcpy(method, optarg);
                    }
                    else{
                        fprintf(stderr, "Unsupported 'method' value\n");
                        exit(EXIT_FAILURE);
                    }
                } else if (strcmp(long_options[option_index].name, "block_size") == 0) {
                    if (optarg != NULL) {
                        *block_size = (int)atoi(optarg);
                    }
                }

                break;
            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }
}


// Function to process the provided options
void process_benchmark_gpu_options(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* iterations_per_config, int* block_size, int* th_size_x, int* th_size_y) {
    
    // Long options structure
    static struct option long_options[] = {
        {"method", required_argument, 0, 0},
        {"min_powerof2", required_argument, 0, 0},
        {"max_powerof2", required_argument, 0, 0},
        {"iterations_per_config", required_argument, 0, 0},
        {"block_size", required_argument, 0, 0},
        {"th_size_x", required_argument, 0, 0},
        {"th_size_y", required_argument, 0, 0},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (c) {
            case 0: // Long option found
                if (strcmp(long_options[option_index].name, "min_powerof2") == 0) {
                    *min_powerof2 = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "method") == 0){
                    if (strcmp(optarg, "blocks_naive") == 0 || strcmp(optarg, "blocks_coalesced") == 0 || strcmp(optarg, "all") == 0){
                        strcpy(method, optarg);
                    }
                    else{
                        fprintf(stderr, "Unsupported 'method' value\n");
                        exit(EXIT_FAILURE);
                    }        
                } else if (strcmp(long_options[option_index].name, "max_powerof2") == 0) {
                    *max_powerof2 = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "iterations_per_config") == 0) {
                    *iterations_per_config = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "block_size") == 0) {
                    *block_size = (int)atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_x") == 0) {
                    *th_size_x = (int)atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_y") == 0) {
                    *th_size_y = (int)atoi(optarg);
                }
                break;

            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }

    if (*th_size_x > *block_size || *th_size_y > *block_size){
        fprintf(stderr, "===\nERROR: th_size_x & th_size_y CAN'T exceed block_size\n===\n");
        exit(EXIT_FAILURE);
    }
    if (*th_size_x + *th_size_y > 11){
        fprintf(stderr, "===\nERROR: unpredictable behaviour\ntotal number of threads per block can't exceed 1024\nplease set 'th_size_x' + 'th_size_y' <= 11\n===\n");
        exit(EXIT_FAILURE);
    }
    if (*block_size > 6){
        fprintf(stderr, "===\nERROR: shared memory overflow\nplease set 'block_size' <= 6\n===\n");
        exit(EXIT_FAILURE);
    }
}