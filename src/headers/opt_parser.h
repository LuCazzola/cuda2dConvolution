#ifndef OPT_PARSER_H
#define OPT_PARSER_H

#include "common.h"
#include <unistd.h>
#include <getopt.h>
#include <string.h>

// parse CL options for main.c 
void process_transpose_options(int argc, char *argv[], char *method, int *N, int *block_size);
// parse CL options for main_gpu.c 
void process_transpose_gpu_options(int argc, char *argv[], char *method, int *N, int *block_size, int *th_rows, int *th_cols);
// parse CL options for benchmark.c
void process_benchmark_options(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* iterations_per_config, int* block_size);
// parse CL options for benchmark_gpu.c
void process_benchmark_gpu_options(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* iterations_per_config, int* block_size, int* th_size_x, int* th_size_y);

#endif