#!/bin/bash
#SBATCH -J sptrsv-bench
#SBATCH -o sptrsv-%A-%a.out
#SBATCH -e sptrsv-%A-%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=0-1

set -euo pipefail

ROOT=${ROOT:-$PWD}
PYTHON_BIN=${PYTHON_BIN:-python}
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

