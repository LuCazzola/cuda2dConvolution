#!/bin/bash

### Options to run the test on the Marzola cluster ###

#SBATCH --job-name=parallelImageProcessing
#SBATCH --output=output.out
#SBATCH --error=error.err

#SBATCH --partition=edu5
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


# DESCRIPTION : 
# ...

# OUTPUT :
# ...

### User Variables ###

method="blocks_naive"   # chose which version of transpose to run between "transpose_blocks_gpu()" and "transpose_blocks_gpu_coalesced()" by setting :
                        #   method="blocks_naive" : run transpose_blocks_gpu() 
                        #   method="blocks_coalesced" : run transpose_blocks_gpu_coalesced()

N=12  # defines the matrix size : ( 2^N x 2^N )

block_size=5  # in method transpose_blocks() defines the block size : ( 2^block_size x 2^block_size )

th_size_x=3  # defines threadBlock shape ( 2^th_size_x x 2^th_size_y )
th_size_y=5

./bin/project-imageProcessing --method=$method --N=$N --block_size=$block_size --th_size_x=$th_size_x --th_size_y=$th_size_y