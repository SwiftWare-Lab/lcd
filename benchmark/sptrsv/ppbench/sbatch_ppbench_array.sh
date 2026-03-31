#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="ppbench"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output="ppbench.%A-%a.out"
#SBATCH -t 11:59:00
#SBATCH --array=0-49

module load StdEnv/2023
module load python
module load gcc

set -euo pipefail

PPBENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}

MATRIX_LIST_FILE=${1:-matrix_list_orig.txt}
MATRIX_ROOT=${2:-}
NUM_ARRAY_JOBS=${NUM_ARRAY_JOBS:-50}
TIMEOUT=${MATRIX_TIMEOUT:-14340}
PPBENCH_REPEATS=${PPBENCH_REPEATS:-5}
PPBENCH_BAND_WIDTH=${PPBENCH_BAND_WIDTH:-64}

PPBENCH_VENV="${PPBENCH_VENV:-/scratch/dama/venv}"
if [[ ! -f "${PPBENCH_VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found: PPBENCH_VENV=${PPBENCH_VENV}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PPBENCH_VENV}/bin/activate"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "${PYTHON_BIN}" -c "import numpy, scipy, numba" 2>/dev/null; then
  echo "ERROR: numpy/scipy/numba not available in $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" >&2
  exit 1
fi

readarray -t MATRIX_ENTRIES < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

if [[ ${#MATRIX_ENTRIES[@]} -eq 0 ]]; then
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

TOTAL_MATRICES=${#MATRIX_ENTRIES[@]}
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

LIST_STEM="$(basename "$MATRIX_LIST_FILE")"
LIST_STEM="${LIST_STEM%.txt}"
OUT_DIR="${OUT_DIR:-${ROOT}/ppbench-results/${LIST_STEM}}"
mkdir -p "$OUT_DIR"
TASK_OUTPUT="${OUT_DIR}/ppbench-${SLURM_ARRAY_TASK_ID}.csv"

echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_ARRAY_JOBS}: processing indices [$START_INDEX, $((END_INDEX-1))]"
echo "Task ${SLURM_ARRAY_TASK_ID}: writing ${TASK_OUTPUT}"

CMD=(
  "${PYTHON_BIN}" "${PPBENCH_DIR}/benchmark_numba_preprocess.py"
  --matrix-list "${MATRIX_LIST_FILE}"
  --start-index "${START_INDEX}"
  --end-index "${END_INDEX}"
  --repeats "${PPBENCH_REPEATS}"
  --band-width "${PPBENCH_BAND_WIDTH}"
  --output "${TASK_OUTPUT}"
)

if [[ -n "${MATRIX_ROOT}" ]]; then
  CMD+=( --matrix-root "${MATRIX_ROOT}" )
fi

if timeout --kill-after=10 --signal=TERM ${TIMEOUT} "${CMD[@]}"; then
  echo "Task ${SLURM_ARRAY_TASK_ID}: done"
else
  EXIT_CODE=$?
  if [[ $EXIT_CODE -eq 124 ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: timeout"
  else
    echo "Task ${SLURM_ARRAY_TASK_ID}: error (${EXIT_CODE})"
  fi
  exit $EXIT_CODE
fi
