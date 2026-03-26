#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="sptrsv"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output="sptrsv.%A-%a.out"
#SBATCH -t 11:59:00
#SBATCH --array=0-49

module load StdEnv/2023
module load python
module load gcc
module load cuda
module load cmake

set -euo pipefail

# ============================================================================
# Paths
# ============================================================================
REPO_ROOT="/scratch/dama/additional_experiments/gpu-backend"
SPTRSV_DIR="${REPO_ROOT}/benchmark/sptrsv"

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
MATRIX_LIST_FILE=$1
NUM_ARRAY_JOBS=${NUM_ARRAY_JOBS:-50}

# Python for CMake/pybind11 and for slurm_sptrsv_benchmark.py must be the same
# interpreter (venv with pybind11, cupy, etc.). Activate *before* building .so files —
# bare `module load python` does not provide pybind11.
SPTRSV_VENV="${SPTRSV_VENV:-/scratch/dama/venv}"
if [[ ! -f "${SPTRSV_VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found: SPTRSV_VENV=${SPTRSV_VENV}" >&2
  echo "  Create it or set SPTRSV_VENV, then: python -m pip install pybind11" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SPTRSV_VENV}/bin/activate"
export PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "${PYTHON_BIN}" -c "import pybind11" 2>/dev/null; then
  echo "ERROR: pybind11 not installed for: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" >&2
  echo "  Run:  ${PYTHON_BIN} -m pip install pybind11" >&2
  exit 1
fi

readarray -t MATRIX_PATHS < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

if [[ ${#MATRIX_PATHS[@]} -eq 0 ]]; then
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

TOTAL_MATRICES=${#MATRIX_PATHS[@]}
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

# ============================================================================
# Build external pybind11 modules (once across all array tasks)
# ============================================================================
# Uses flock so that exactly one task builds; the rest wait for it to finish.
LOCK_FILE="${SPTRSV_DIR}/.external_modules.lock"
MARKER_FILE="${SPTRSV_DIR}/.external_modules.built"

build_external_modules() {
  echo "Task ${SLURM_ARRAY_TASK_ID}: building external pybind modules (yysptrsv + cusparse_sptrsv) ..."
  bash "${SPTRSV_DIR}/build_external_modules.sh"
  touch "${MARKER_FILE}"
  echo "Task ${SLURM_ARRAY_TASK_ID}: external modules built and installed"
}

(
  flock -x 200
  shopt -s nullglob
  _yy_so=( "${SPTRSV_DIR}"/yysptrsv_wrapper*.so )
  _cs_so=( "${SPTRSV_DIR}"/cusparse_sptrsv*.so )
  shopt -u nullglob
  if [[ ! -f "${MARKER_FILE}" ]] || [[ ${#_yy_so[@]} -eq 0 ]] || [[ ${#_cs_so[@]} -eq 0 ]]; then
    build_external_modules
  else
    echo "Task ${SLURM_ARRAY_TASK_ID}: external modules already built, skipping"
  fi
) 200>"${LOCK_FILE}"

# ============================================================================
# Benchmark configuration
# ============================================================================
export SPTRSV_SKIP_CORRECTNESS=${SPTRSV_SKIP_CORRECTNESS:-1}
export SPTRSV_QUICK=${SPTRSV_QUICK:-1}
export SPTRSV_VERBOSE=${SPTRSV_VERBOSE:-0}
export SPTRSV_WARMUP=${SPTRSV_WARMUP:-}
export SPTRSV_REP=${SPTRSV_REP:-}
export SPTRSV_PROVIDERS=${SPTRSV_PROVIDERS:-}

export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-$SCRATCH/cupy_cache}
mkdir -p "$CUPY_CACHE_DIR"

cd "$ROOT"
# venv already active (see SPTRSV_VENV above)

echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_ARRAY_JOBS}: processing indices [$START_INDEX, $((END_INDEX-1))]"

TIMEOUT=${MATRIX_TIMEOUT:-14340}
for (( i=START_INDEX; i<END_INDEX; i++ )); do
  export SPTRSV_MATRIX="${MATRIX_PATHS[$i]}"

  echo "Benchmarking matrix $((i+1))/$TOTAL_MATRICES: ${SPTRSV_MATRIX}"

  if timeout --kill-after=10 --signal=TERM ${TIMEOUT} \
       python "${SPTRSV_DIR}/slurm_sptrsv_benchmark.py"; then
    echo "Done: ${SPTRSV_MATRIX}"
  else
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 124 ]]; then
      echo "Timeout: ${SPTRSV_MATRIX}"
    else
      echo "Error ($EXIT_CODE): ${SPTRSV_MATRIX}"
    fi
  fi
done
