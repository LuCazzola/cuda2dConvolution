#include "headers/opt_parser.h"

void process_main_args (int argc, char *argv[], char* method, char* input_png_path, char* output_png_path, int* kernel_size, int* th_size_x, int* th_size_y){
    // Long options structure
    static struct option long_options[] = {
        {"method", optional_argument, 0, 0},
        {"input_png_path", optional_argument, 0, 0},
        {"output_png_path", optional_argument, 0, 0},
        {"kernel_size", required_argument, 0, 0},
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
                if (strcmp(long_options[option_index].name, "method") == 0) {
                    if(strcmp(optarg, "cpu_naive") == 0 || strcmp(optarg, "gpu_naive") == 0 || strcmp(optarg, "gpu_shared") == 0){
                        strcpy(method, optarg);
                    }
                    else{
                        fprintf(stderr, "Unsupported 'method' value\n");
                        exit(EXIT_FAILURE);
                    }
                }
                else if (strcmp(long_options[option_index].name, "kernel_size") == 0) {
                    *kernel_size = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "input_png_path") == 0){             
                    strcpy(input_png_path, optarg);
                } else if (strcmp(long_options[option_index].name, "output_png_path") == 0){             
                    strcpy(output_png_path, optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_x") == 0){             
                    *th_size_x = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_y") == 0){             
                    *th_size_y = (int) atoi(optarg);
                }
                break;
            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }

    if (*th_size_x + *th_size_y >= 11) {
        fprintf(stderr, "===\nERROR: 'th_size_x' + 'th_size_y' must be < 11 (Cuda allows max 1024 threads per block)\n===\n");
        exit(EXIT_FAILURE);
    }
}


void process_benchmark_args(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* min_kernel_size, int* max_kernel_size, int* iterations_per_config, int* th_size_x, int* th_size_y) {
    // Long options structure
    static struct option long_options[] = {
        {"method", required_argument, 0, 0},
        {"min_powerof2", required_argument, 0, 0},
        {"max_powerof2", required_argument, 0, 0},
        {"min_kernel_size", required_argument, 0, 0},
        {"max_kernel_size", required_argument, 0, 0},
        {"iterations_per_config", required_argument, 0, 0},
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
                } else if (strcmp(long_options[option_index].name, "max_powerof2") == 0) {
                    *max_powerof2 = (int) atoi(optarg);
                }  else if (strcmp(long_options[option_index].name, "min_kernel_size") == 0){
                    *min_kernel_size = (int) atoi(optarg);
                }else if (strcmp(long_options[option_index].name, "max_kernel_size") == 0){
                    *max_kernel_size = (int) atoi(optarg);
                }else if (strcmp(long_options[option_index].name, "method") == 0){
                    if (strcmp(optarg, "cpu_naive") == 0 || strcmp(optarg, "gpu_naive") == 0 || strcmp(optarg, "gpu_shared") == 0 || strcmp(optarg, "all") == 0){
                        strcpy(method, optarg);
                    }
                    else{
                        fprintf(stderr, "Unsupported 'method' value\n");
                        exit(EXIT_FAILURE);
                    }        
                } else if (strcmp(long_options[option_index].name, "iterations_per_config") == 0) {
                    *iterations_per_config = (int) atoi(optarg);
                }  else if (strcmp(long_options[option_index].name, "th_size_x") == 0){             
                    *th_size_x = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "th_size_y") == 0){             
                    *th_size_y = (int) atoi(optarg);
                }
                break;

            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }

    if (*min_kernel_size < 1 || *max_kernel_size < 1 || *min_kernel_size % 2 == 0 || *max_kernel_size % 2 == 0 ){
        fprintf(stderr, "===\nERROR: max/min Kernel sizes must be positive and odd\n===\n");
        exit(EXIT_FAILURE);
    }
    if(*min_kernel_size > *max_kernel_size){
        fprintf(stderr, "===\nERROR: 'min_kernel_size' must < than 'max_kernel_size'\n===\n");
        exit(EXIT_FAILURE);
    }
    if (*th_size_x + *th_size_y >= 11) {
        fprintf(stderr, "===\nERROR: 'th_size_x' + 'th_size_y' must be < 11 (Cuda allows max 1024 threads per block)\n===\n");
        exit(EXIT_FAILURE);
    }
}