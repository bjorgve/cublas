#!/bin/sh
#SBATCH --account=nn9997k --job-name=MyJob
#SBATCH --partition=accel --gpus=1
#SBATCH --time=0-0:10:0
#SBATCH --mem-per-cpu=1G

ml CUDA/12.0.0
srun matrix_multiplication_complex_eigen

