#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="hpcg"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output="hpcg.%A-%a.out"
#SBATCH -t 01:59:00
#SBATCH --array=0-1

module load StdEnv/2023
module load python
module load gcc
module load cuda

set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON_BIN=/scratch/dama/venv/bin/python
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-$ROOT/matrix_list.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/hpcg_results_cluster}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Read matrix list
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

MATRIX=${MATRICES[$SLURM_ARRAY_TASK_ID]}
MATRIX_NAME=$(basename "$MATRIX" .mtx)

echo "========================================"
echo "HPCG Benchmark - Task ${SLURM_ARRAY_TASK_ID}/${#MATRICES[@]}"
echo "========================================"
echo "Matrix: ${MATRIX_NAME}"
echo "Path: ${MATRIX}"
echo "Node: ${SLURM_NODELIST}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Set up CuPy cache
export CUPY_CACHE_DIR=/scratch/dama/cupy_cache/${SLURM_JOB_ID}
mkdir -p "$CUPY_CACHE_DIR"

# Run benchmark for this matrix
cd "$ROOT"
source /scratch/dama/venv/bin/activate

echo "Starting benchmark at $(date)"
$PYTHON_BIN hpcg.py --matrix "$MATRIX" --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "Benchmark completed at $(date)"

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Status: SUCCESS"
else
  echo "Status: FAILED (exit code: $EXIT_CODE)"
fi

# Clean up CuPy cache for this job
rm -rf "$CUPY_CACHE_DIR"

exit $EXIT_CODE
