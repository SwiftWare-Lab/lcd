#!/bin/bash


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH -t 3:59:00
#SBATCH --constraint=cascade

MAT_FILE=$1
MAT_DIR=$2


module load python/3.11.5

source ~/.virtualenvs/myenv/bin/activate

export PYTHONPATH="$SCRATCH/projects/gpu-backend"

python compiler/benchmarks.py $MAT_FILE $MAT_DIR > sddmm_spmm_benchmark

deactivate

