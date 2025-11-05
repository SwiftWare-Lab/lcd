#! /bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="lpc"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100000M
#SBATCH --output="lpc.%j.%N.out"
#SBATCH -t 3:59:00

UFDB=$1
EXP=$2
BUILD=$3
PROFILING=$4

module load StdEnv/2023
module load python

MAT_FILE=$1
MAT_DIR=$2


module load python/3.11.5

source ~/.virtualenvs/myenv/bin/activate

export PYTHONPATH="$SCRATCH/projects/gpu-backend"

python compiler/fusion_benchmark/spmm_softmax.py $MAT_FILE $MAT_DIR > spmm_softmax_benchmark

deactivate