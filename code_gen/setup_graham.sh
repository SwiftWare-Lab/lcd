#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=prachyosarker565@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 1:00:00
#SBATCH --mem=50G  # Request 32 GB of memory


module load StdEnv/2020
module load intel/2022.2.1

module load cmake
module load gcc
module load python
module load cuda/12.2

python3 -m venv venv
source venv/bin/activate

pip install numpy
pip install triton
pip install torch
pip install scipy
pip install matplotlib
pip install cupy

nvcc --version
