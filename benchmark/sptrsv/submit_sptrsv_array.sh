#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="sptrsv"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="sptrsv.%A-%a.out"
#SBATCH -t 03:59:00
#SBATCH --array=0-1

module load StdEnv/2023
module load python
module load gcc
module load cuda

set -euo pipefail

ROOT=${ROOT:-$PWD}
PYTHON_BIN=/scratch/dama/venv/bin/python
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-$(dirname "$0")/matrix_list.txt}

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
export SPTRSV_SKIP_CORRECTNESS=${SPTRSV_SKIP_CORRECTNESS:-1}
export SPTRSV_QUICK=${SPTRSV_QUICK:-1}
echo "Benchmarking ${SPTRSV_MATRIX} (task ${SLURM_ARRAY_TASK_ID}/${#MATRICES[@]})"

cd "$ROOT"
$PYTHON_BIN slurm_sptrsv_benchmark.py

