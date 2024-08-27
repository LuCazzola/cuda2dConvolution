# cuda2dConvolution

## Requisites

Before testing commands run :
```
git clone https://github.com/LuCazzola/cuda2dConvolution.git
cd cuda2dConvolution
moudle load cuda/12.1 libpng
```

## Commands
Makefile defines 4 rules :
* **make** : builds main.cu + dependancies executables
* **make benchmark** : builds benchmark.cu + dependancies executables
* **make debug** :  builds both main.cu and benchmark.cu + dependancies with debugging flags
* **make clean** : cleans all object files
<br>

Bash scripts [run_main.sh](run_main.sh) and [run_benchmark.sh](run_benchmark.sh) each containing launching instructions and customizable variables are made available.
To execute them run :
```
sbatch ./run_main.sh
```
or
```
sbatch ./run_benchmark.sh
```
