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

module load StdEnv/2023
module load gcc
module load cuda

#mkdir -p build/ && cd build
#cmake -DCMAKE_BUILD_TYPE=Release ..
#
#cmake --build . --config Release -- -j4

echo "matrix,nnz,time,throughput"

while read line; do
  mat="$MAT_FOLDER/$line"
#  echo $mat
#  ./sptrsv_bench $mat 100 10
  python test_wrapper.py $mat
done < $MAT_FILE

deactivate