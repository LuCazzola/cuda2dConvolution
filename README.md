# cuda2dConvolution
The following repository contains all the material related to both the final project on **Parallel image processing** assigned during the GPU computing course : University of Trento (Italy) a.y. 2023/2024.

<br>

To see the report and better understand what this work is about, click [**Here**](report.pdf) <br>
Authors : [@LuCazzola](https://github.com/LuCazzola) [@chrisdalvit](https://github.com/chrisdalvit) 

## Requisites
```
git clone https://github.com/LuCazzola/cuda2dConvolution.git
cd cuda2dConvolution
```

download the following modules or load them if you're in a SLURM cluster
```
moudle load cuda/12.1 libpng
```

<br>

## Commands
Makefile defines 4 rules :
* **make** : builds main.cu + dependancies executables
* **make benchmark** : builds benchmark.cu + dependancies executables
* **make debug** :  builds both main.cu and benchmark.cu + dependancies with debugging flags
* **make clean** : cleans all object files
<br>

Bash scripts [**run_main.sh**](/run_main.sh) and [**run_benchmark.sh**](/run_benchmark.sh) each containing launching instructions and customizable variables are made available.

### Main
```
sbatch ./run_main.sh
```
Takes as input an .png image and returns as output a Gaussian blurred version of the image obtained with the selected kernel and configuration.
<br>
Results are stored in [**images folder**](images)

### Benchmark
```
sbatch ./run_benchmark.sh
```
Generates a .csv benchmark file measuring mean + standard deviation of each secified algorithm : execution time, effective bandwidth, FLOPS.
<br>
Results are stored in [**data folder**](data)

