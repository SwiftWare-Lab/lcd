#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=hosseinalbakri3@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 1:00:00
#SBATCH --mem=50G  # Request 32 GB of memory

module load StdEnv/2020
module load intel/2022.2.1
module load cmake
module load gcc
module load python
module load cuda

source venv/bin/activate

cd ./benchmark_kernels

python3 run_experiment2.py
