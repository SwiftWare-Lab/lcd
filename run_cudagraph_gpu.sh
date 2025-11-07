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



module load StdEnv/2023
module load python
module load gcc
module load cuda

MAT_FILE=$1
MAT_DIR=$2
COUNT=$3

source ./venv/bin/activate

export TRITON_CACHE_DIR=$SCRATCH/.triton/cache
echo 'impl,matrix,time,gflops' > sptrsv_cudagraph_benchmark_$COUNT
#export PYTHONPATH="$SCRATCH/projects/gpu-backend"
while read line; do
  mat=$line
  python benchmark/sptrsv_cudagraph_benchmark.py $mat $MAT_DIR >> sptrsv_cudagraph_benchmark_$COUNT
done < $MAT_FILE
#python benchmark/sptrsv_benchmark.py $MAT_DIR $MAT_FILE > sptrsv_benchmark_$COUNT

deactivate
