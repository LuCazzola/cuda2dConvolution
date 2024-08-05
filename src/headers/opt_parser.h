#ifndef OPT_PARSER_H
#define OPT_PARSER_H

#include "common.h"
#include <unistd.h>
#include <getopt.h>
#include <string.h>

// parse CL options for main.cu
void process_main_args (int argc, char *argv[], char* method, char* pngPath, int* kernel_size);
// parse CL options for benchmark.cu
void process_benchmark_args(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* min_kernel_size, int* max_kernel_size, int* iterations_per_config);

#endif