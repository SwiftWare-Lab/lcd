#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="gauss-seidel"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="gauss-seidel.%A-%a.out"
#SBATCH -t 05:59:00
#SBATCH --array=0-1

module load StdEnv/2023
module load python
module load gcc
module load cuda
set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON_BIN=/scratch/dama/venv/bin/python
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-$ROOT/matrix_list.txt}

readarray -t MATRICES < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

if [[ ${#MATRICES[@]} -eq 0 ]]; then
  echo "No matrices listed in $MATRIX_LIST_FILE"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Run with: sbatch --array=0-$((${#MATRICES[@]}-1)) $0"
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= ${#MATRICES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} out of range (size ${#MATRICES[@]})"
  exit 1
fi

export SPTRSV_MATRIX=${MATRICES[$SLURM_ARRAY_TASK_ID]}
echo "Benchmarking ${SPTRSV_MATRIX} (task ${SLURM_ARRAY_TASK_ID}/${#MATRICES[@]})"

export CUPY_CACHE_DIR=/scratch/dama/cupy_cache
mkdir -p /scratch/dama/cupy_cache

cd "$ROOT"
source /scratch/dama/venv/bin/activate
$PYTHON_BIN slurm_gauss_seidel_benchmark.py

