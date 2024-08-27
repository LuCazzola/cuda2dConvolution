#!/bin/bash

### Options to run the test on the Marzola cluster ###

#SBATCH --job-name=main_imgProcessing
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
# method to use for image processing, available methods :
method="gpu_shared_constk"  
# - "cpu_naive"
# - "gpu_naive"
# - "gpu_shared"
# - "gpu_shared_constk"
# - "gpu_shared_constk_cached"

# path to the image to process
input_png_path="images/lenna.png"
# path to the output image
output_png_path="images/out.png"

# kernel filter size : ( kernel_size x kernel_size )
kernel_size=5
# - MUST BE ODD NUMBER

# thread block size in the x direction (as a power of 2 => 2^th_size_x)
th_size_x=4
# thread block size in the y direction (as a power of 2 => 2^th_size_y)
th_size_y=4
# th_size_x & th_size_y MUST BE EQUAL

./bin/project-imageProcessing --method=$method --input_png_path=$input_png_path --output_png_path=$output_png_path --kernel_size=$kernel_size --th_size_x=$th_size_x --th_size_y=$th_size_y
