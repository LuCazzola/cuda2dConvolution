#ifndef OPT_PARSER_H
#define OPT_PARSER_H

#include "common.h"
#include <unistd.h>
#include <getopt.h>
#include <string.h>

// parse CL options for main.cu
void process_main_args (int argc, char *argv[], char* method, char* input_png_path, char* output_png_path, int* kernel_size, int* th_size_x, int* th_size_y);
// parse CL options for benchmark.cu
void process_benchmark_args(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* min_kernel_size, int* max_kernel_size, int* iterations_per_config, int* th_size_x, int* th_size_y);

#endif