#!/bin/bash

# DESCRIPTION : 
# "homework-1" runs one out of 2 matrix transposition algorithms showing the related results

# OUTPUT :
# prompts details about the run on the terminal and computes 2 metrics :
#  - execution time : time to compute the transpose() function
#  - effective bandwidth : estimated bandwidth of transpose() function


### User Variables ###
N=8               # defines the matrix size : ( 2^N x 2^N )

method="blocks"    # chose which version of transpose() to run between "transpose_naive()" and "transpose_blocks()" by setting :
                   #   method="naive" : run transpose_naive()
                   #   method="blocks" : run transpose_blocks()

block_size=4       # in method transpose_blocks() defines the block size : ( 2^block_size x 2^block_size )


./bin/homework-1 --method=$method --N=$N --block_size=$block_size
