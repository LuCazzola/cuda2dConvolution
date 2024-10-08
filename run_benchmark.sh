#!/bin/bash

### Options to run the test on the Marzola cluster ###

#SBATCH --job-name=bench_imgProcessing
#SBATCH --output=output.out
#SBATCH --error=error.err

#SBATCH --partition=edu5
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


# DESCRIPTION :
# runs Convolution operation algorithms : 
# - The program runs cpu_convolution_naive() / gpu_convolution_naive() / gpu_convolution_shared() / gpu_convolution_shared_constk() / gpu_convolution_shared_constk_cached()
# - The kernels are performed with different parameters configuration
# - each configuration runs "iterations_per_config", then the matrix size is increased by one : 
#      - starting with (2^min_powerof2 x 2^min_powerof2) up to (2^max_powerof2 x 2^max_powerof2) 
# - during each configuration some data are stored into .csv files which can be found into the "data" folder

# OUTPUT :
# one or more .csv files containing :
#  - Details about the run such as : block_size, th_size_x, th_size_y, & data type of matrix elements
#  - a table which stores : 
#      - { execution time, effective bandwidth, flops } "iterations_per_config" times per configuration
#      - { average execution time, average effective bandwidth, average flops } (per each configuration)
#      - { standard deviation of execution time, standard deviation of effective bandwidth, standard deviation of flops } (per each configuration)  

### User Variables ###

# chose which version of convolution operation to run :
method="gpu_only"
#   method = "all"                      : run ALL METHODS
#   method = "gpu_only"                 : run ALL METHODS excluded cpu_convolution_naive()
#   method = "cpu_naive"                : run cpu_convolution_naive()
#   method = "gpu_naive"                : run gpu_convolution_naive()
#   method = "gpu_shared"               : run gpu_convolution_shared()
#   method = "gpu_shared_constk"        : run gpu_convolution_shared_constk()
#   method = "gpu_shared_constk_cached" : run gpu_convolution_shared_constk_cached()

# size of the FIRST kernel matrix : ( min_kernel_size x min_kernel_size )
min_kernel_size=3
# size of the LAST kernel matrix  : ( max_kernel_size x max_kernel_size )
max_kernel_size=9

# size of the FIRST tested matrix : ( 2^min_powerof2 x 2^min_powerof2 )
min_powerof2=8
# size of the LAST tested matrix : ( 2^max_powerof2 x 2^max_powerof2 )
max_powerof2=14

# number of times each configuration of parameters is repeated executed
iterations_per_config=50

# thread block size in the x direction (as a power of 2 => 2^th_size_x)
th_size_x=4
# thread block size in the y direction (as a power of 2 => 2^th_size_y)
th_size_y=4
# th_size_x & th_size_y MUST BE EQUAL

./bin/benchmark  --method=$method --min_kernel_size=$min_kernel_size --max_kernel_size=$max_kernel_size --min_powerof2=$min_powerof2 --max_powerof2=$max_powerof2 --iterations_per_config=$iterations_per_config --th_size_x=$th_size_x --th_size_y=$th_size_y
