#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="sptrsv"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="sptrsv.%A-%a.out"
#SBATCH -t 03:59:00
#SBATCH --array=0-2

module load StdEnv/2023
module load python
module load gcc
module load cuda

set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON_BIN=${PYTHON_BIN:-/scratch/dama/venv/bin/python}
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-$ROOT/matrix_list.txt}

# Parent directory to prepend to each matrix path in MATRIX_LIST_FILE
MATRIX_PARENT_DIR=${MATRIX_PARENT_DIR:-$ROOT}

# Number of array jobs to divide the matrix list into
NUM_ARRAY_JOBS=${NUM_ARRAY_JOBS:-8}

# Read non-empty, non-comment lines
readarray -t MATRIX_REL_PATHS < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

if [[ ${#MATRIX_REL_PATHS[@]} -eq 0 ]]; then
  echo "No matrices listed in $MATRIX_LIST_FILE"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Run with: sbatch --array=0-$(($NUM_ARRAY_JOBS-1)) $0"
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= NUM_ARRAY_JOBS )); then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} out of range for NUM_ARRAY_JOBS=${NUM_ARRAY_JOBS}"
  exit 1
fi

TOTAL_MATRICES=${#MATRIX_REL_PATHS[@]}

# Ceiling division: chunk size = ceil(TOTAL_MATRICES / NUM_ARRAY_JOBS)
CHUNK_SIZE=$(( (TOTAL_MATRICES + NUM_ARRAY_JOBS - 1) / NUM_ARRAY_JOBS ))

START_INDEX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_INDEX=$(( START_INDEX + CHUNK_SIZE ))

if (( END_INDEX > TOTAL_MATRICES )); then
  END_INDEX=$TOTAL_MATRICES
fi

if (( START_INDEX >= TOTAL_MATRICES )); then
  echo "Task ${SLURM_ARRAY_TASK_ID}: no matrices assigned"
  exit 0
fi

export SPTRSV_SKIP_CORRECTNESS=${SPTRSV_SKIP_CORRECTNESS:-1}
export SPTRSV_QUICK=${SPTRSV_QUICK:-1}

export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-$SCRATCH/cupy_cache}
mkdir -p "$CUPY_CACHE_DIR"

cd "$ROOT"
source $SCRATCH/projects/gpu-backend/venv/bin/activate

echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_ARRAY_JOBS}: processing indices [$START_INDEX, $((END_INDEX-1))]"

for (( i=START_INDEX; i<END_INDEX; i++ )); do
  MATRIX_REL_PATH="${MATRIX_REL_PATHS[$i]}"
  export SPTRSV_MATRIX="${MATRIX_PARENT_DIR}/${MATRIX_REL_PATH}"

  echo "Benchmarking matrix $((i+1))/$TOTAL_MATRICES: ${SPTRSV_MATRIX}"

  $PYTHON_BIN slurm_sptrsv_benchmark.py
done