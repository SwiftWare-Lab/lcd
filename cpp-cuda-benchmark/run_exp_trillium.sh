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


echo "----- Building -----"

export CUDACXX=$(which nvcc)

# Create build folder
mkdir -p build && cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --config Release -- -j4

echo "impl,time,matrix" > sptrsv_supernodal_bench

while read line; do
  mat="$MAT_FOLDER/$line"
  echo $mat
  ./bench  $mat >> sptrsv_supernodal_bench
done < $MAT_FILE
