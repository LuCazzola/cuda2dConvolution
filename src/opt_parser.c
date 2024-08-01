#include "headers/opt_parser.h"

void process_main_args (int argc, char *argv[], char* pngPath, int* kernel_size, bool* custom_input, int* custom_input_width, int* custom_input_height, int* custom_input_channels) {
    // Long options structure
    static struct option long_options[] = {
        {"kernel_size", required_argument, 0, 0},
        {"pngPath", optional_argument, 0, 0},
        {"custom_input", optional_argument, 0, 0},
        {"custom_input_width", optional_argument, 0, 0},
        {"custom_input_height", optional_argument, 0, 0},
        {"custom_input_channels", optional_argument, 0, 0},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (c) {
            case 0: // Long option found
                if (strcmp(long_options[option_index].name, "kernel_size") == 0) {
                    *kernel_size = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "pngPath") == 0){             
                    strcpy(pngPath, optarg);
                } else if (strcmp(long_options[option_index].name, "custom_input") == 0) {
                    *custom_input = (bool) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "custom_input_width") == 0) {
                    *custom_input_width = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "custom_input_height") == 0) {
                    *custom_input_height = (int) atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "custom_input_channels") == 0) {
                    *custom_input_channels = (int) atoi(optarg);
                }

                break;
            default:
                fprintf(stderr, "Unknown option\n");
                exit(EXIT_FAILURE);
        }
    }

    if (*custom_input){
        fprintf(stderr, "custom input enabled, input image path will be ignored\n");
    }
}


void process_benchmark_args(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* iterations_per_config, int* th_size_x, int* th_size_y) {
    // Long options structure
    static struct option long_options[] = {
        {"method", required_argument, 0, 0},
        {"min_powerof2", required_argument, 0, 0},
        {"max_powerof2", required_argument, 0, 0},
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

    if (*th_size_x + *th_size_y > 11){
        fprintf(stderr, "===\nERROR: unpredictable behaviour\ntotal number of threads per block can't exceed 1024\nplease set 'th_size_x' + 'th_size_y' <= 11\n===\n");
        exit(EXIT_FAILURE);
    }
}