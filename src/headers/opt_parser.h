#ifndef OPT_PARSER_H
#define OPT_PARSER_H

#include "common.h"
#include <unistd.h>
#include <getopt.h>
#include <string.h>

// parse CL options for main.c 
void process_main_args (int argc, char *argv[], char* pngPath, int* kernel_size, bool* custom_input, int* custom_input_width, int* custom_input_height, int* custom_input_channels);
// parse CL options for benchmark.c
void process_benchmark_args(int argc, char *argv[], char *method, int* min_powerof2, int* max_powerof2, int* iterations_per_config, int* th_size_x, int* th_size_y);

#endif