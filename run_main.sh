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
method="gpu_naive"          # method to use for image processing, available methods :
                            # - cpu_naive
                            # - gpu_naive

pngPath="images/lenna.png"   # path to the image to process

kernel_size=3           # kernel filter size : ( kernel_size x kernel_size )
                        # - MUST BE ODD NUMBER

./bin/project-imageProcessing --method=$method --pngPath=$pngPath --kernel_size=$kernel_size