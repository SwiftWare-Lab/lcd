#! /bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="lpc"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="lpc.%j.%N.out"
#SBATCH -t 11:59:00


MAT_FOLDER=$1
MAT_FILE=$2
VENV_PATH=$3

module load StdEnv/2023
module load gcc
module load cuda
module load python


echo "matrix,time,throughput" > yysptrsv_bench

source $VENV_PATH/bin/activate

while read line; do
  mat="$MAT_FOLDER/$line"
  echo $mat
  python yysptrsv_bench.py >> yysptrsv_bench
done < $MAT_FILE

deactivate