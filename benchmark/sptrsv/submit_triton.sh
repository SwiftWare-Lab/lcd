#! /bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="lcd-triton"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="lcd-triton.%j.%N.out"
#SBATCH -t 11:59:00

MAT_FOLDER=$1
MAT_FILE=$2
VENV_PATH=$3
MATRIX_LIST_FILE=${MAT_FOLDER/$MAT_FILE}
readarray -t MATRICES < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

module load StdEnv/2023
module load gcc
module load cuda
module load python


source $VENV_PATH/bin/activate

python triton_benchmarks.py

deactivate


