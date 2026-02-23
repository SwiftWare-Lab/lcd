#! /bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="mg-sn-sptrsv"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4         #Number of GPUs
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --output="mg-sn-sptrsv.%j.%N.out"
#SBATCH -t 02:59:00


MAT_PATH=$1

module load StdEnv/2023
module load gcc
module load cuda
module load openmpi

echo "----- Building -----"

export CUDACXX=$(which nvcc)

# Create build folder
mkdir -p build && cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --config Release -- -j4

echo "Running on $SLURM_NTASKS MPI ranks"
echo "GPUs per rank: 1"

srun ./multi_gpu_partitioned_test $MAT_PATH