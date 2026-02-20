#! /bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="lpc"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks=2                     # number of MPI ranks
#SBATCH --cpus-per-task=4              # CPU threads per rank
#SBATCH --gpus-per-task=1              # one GPU per rank
#SBATCH --mem=20G
#SBATCH --output="lpc.%j.%N.out"
#SBATCH -t 01:59:00

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