#!/bin/bash

#SBATCH --gres=gpu:v100:1  # Request 1 V100 GPU
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=prachyosarker565@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 0:25:00
#SBATCH --mem=50G  # Request 32 GB of memory


module load StdEnv/2023
module load gcc
module load python
module load cuda/12.2

source venv/bin/activate


python3 -u run_experiment4.py