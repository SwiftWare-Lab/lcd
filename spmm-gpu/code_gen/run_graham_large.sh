#!/bin/bash

#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=prachyosarker565@gmail.com
#SBATCH --nodes=1
#SBATCH -t 3:00:00
#SBATCH --mem=50G  # Request 50 GB of memory

# Set date and time format for output file
DATE=$(date +'%d%m%y_%H%M')
OUTPUT_FILE="output/data_large_a100_${DATE}.out"

# Make sure the output directory exists
mkdir -p output

#SBATCH --output="${OUTPUT_FILE}"

module load StdEnv/2023
module load gcc
module load python
module load cuda/12.2

source venv/bin/activate

python3 -u code_gen_experiment.py