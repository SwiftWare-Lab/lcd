#!/bin/bash
# Local benchmark script for running sptrsv benchmarks on individual matrices
# This avoids single point of failure by running each matrix separately

set -euo pipefail

# Configuration
ROOT=${ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-$ROOT/matrix_list_3.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/results_local_3}
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/logs_3}

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Read matrix list (skip comments and empty lines)
readarray -t MATRICES < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

if [[ ${#MATRICES[@]} -eq 0 ]]; then
  echo "No matrices listed in $MATRIX_LIST_FILE"
  exit 1
fi

echo "Found ${#MATRICES[@]} matrices to benchmark"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""

# Benchmark configuration (override via environment variables)
export SPTRSV_SKIP_CORRECTNESS=${SPTRSV_SKIP_CORRECTNESS:-1}
export SPTRSV_QUICK=${SPTRSV_QUICK:-1}
export SPTRSV_VERBOSE=${SPTRSV_VERBOSE:-1}
export SPTRSV_WARMUP=${SPTRSV_WARMUP:-}    # empty = use benchmark default (2 if QUICK, 25 otherwise)
export SPTRSV_REP=${SPTRSV_REP:-}          # empty = use benchmark default (5 if QUICK, 100 otherwise)
export SPTRSV_PROVIDERS=${SPTRSV_PROVIDERS:-}  # empty = all; e.g. "cusparse,csc-sf-base"
# Default: require cusparse-wrapper + yysptrsv (pybind .so next to slurm_sptrsv_benchmark.py).
# Set to 1 only if those modules are unavailable (CSV will omit those baselines; stderr warns).
export SPTRSV_SKIP_EXTERNAL=${SPTRSV_SKIP_EXTERNAL:-0}

echo "Benchmark config:"
echo "  SPTRSV_SKIP_CORRECTNESS=$SPTRSV_SKIP_CORRECTNESS"
echo "  SPTRSV_QUICK=$SPTRSV_QUICK"
echo "  SPTRSV_VERBOSE=$SPTRSV_VERBOSE"
echo "  SPTRSV_SKIP_EXTERNAL=$SPTRSV_SKIP_EXTERNAL"
[[ -n "$SPTRSV_WARMUP" ]] && echo "  SPTRSV_WARMUP=$SPTRSV_WARMUP"
[[ -n "$SPTRSV_REP" ]] && echo "  SPTRSV_REP=$SPTRSV_REP"
[[ -n "$SPTRSV_PROVIDERS" ]] && echo "  SPTRSV_PROVIDERS=$SPTRSV_PROVIDERS"
echo ""

# Optional: Set CUPY cache directory to avoid conflicts
if [[ -n "${CUPY_CACHE_DIR:-}" ]]; then
  mkdir -p "$CUPY_CACHE_DIR"
fi

# Ensure YY / cuSPARSE-wrapper pybind modules exist (they are gitignored .so files).
if [[ "${SPTRSV_SKIP_EXTERNAL:-0}" != "1" ]]; then
  shopt -s nullglob
  _yy_so=( "${ROOT}"/yysptrsv_wrapper*.so )
  _cs_so=( "${ROOT}"/cusparse_sptrsv*.so )
  shopt -u nullglob
  if [[ ${#_yy_so[@]} -eq 0 || ${#_cs_so[@]} -eq 0 ]]; then
    echo "External baseline .so modules missing under $ROOT; running build_external_modules.sh ..."
    export PYTHON_BIN
    if ! bash "${ROOT}/build_external_modules.sh"; then
      echo "ERROR: build_external_modules.sh failed. Install cmake, CUDA, pybind11, or set SPTRSV_SKIP_EXTERNAL=1." >&2
      exit 1
    fi
  fi
fi

# Track results
SUCCESS_COUNT=0
FAILED_COUNT=0
TIMEOUT_COUNT=0
FAILED_MATRICES=()
TIMEOUT_MATRICES=()

# Timeout duration (3 hours = 10800 seconds)
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}

# Run benchmark for each matrix
for i in "${!MATRICES[@]}"; do
  MATRIX="${MATRICES[$i]}"
  MATRIX_NAME=$(basename "$MATRIX" .mtx)
  
  echo "[$((i+1))/${#MATRICES[@]}] Processing: $MATRIX_NAME"
  echo "  Matrix path: $MATRIX"
  
  # Set environment variable for this specific matrix
  export SPTRSV_MATRIX="$MATRIX"
  
  # Create log file for this matrix
  LOG_FILE="$LOG_DIR/${MATRIX_NAME}.log"
  RESULT_DIR="$OUTPUT_DIR/$MATRIX_NAME"
  mkdir -p "$RESULT_DIR"
  
  # Run benchmark and capture output
  START_TIME=$(date +%s)
  
  # Use timeout command to limit execution time (3 hours = 10800 seconds)
  # timeout returns 124 if the command times out
  if timeout $TIMEOUT_SECONDS bash -c "cd '$ROOT' && $PYTHON_BIN slurm_sptrsv_benchmark.py" > "$LOG_FILE" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # Move generated files to result directory
    if [[ -f "sptrsv-performance.csv" ]]; then
      mv sptrsv-performance.csv "$RESULT_DIR/"
    fi
    if [[ -f "runtime_overhead_${MATRIX_NAME}.csv" ]]; then
      mv "runtime_overhead_${MATRIX_NAME}.csv" "$RESULT_DIR/"
    fi
    # Move any other generated CSVs
    for csv in sptrsv-performance-*.csv runtime_overhead*.csv; do
      [[ -f "$csv" ]] && mv "$csv" "$RESULT_DIR/" 2>/dev/null || true
    done
    
    echo "  ✓ SUCCESS (${ELAPSED}s)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    if [[ $EXIT_CODE -eq 124 ]]; then
      # Timeout occurred
      echo "  ⏱ TIMEOUT after ${TIMEOUT_SECONDS}s - see $LOG_FILE"
      echo "TIMEOUT: Execution exceeded ${TIMEOUT_SECONDS} seconds" >> "$LOG_FILE"
      TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
      TIMEOUT_MATRICES+=("$MATRIX_NAME")
    else
      # Regular failure
      echo "  ✗ FAILED (${ELAPSED}s) - see $LOG_FILE"
      FAILED_COUNT=$((FAILED_COUNT + 1))
      FAILED_MATRICES+=("$MATRIX_NAME")
    fi
  fi
  
  echo ""
done

# Print summary
echo "========================================"
echo "BENCHMARK SUMMARY"
echo "========================================"
echo "Total matrices: ${#MATRICES[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Timed out: $TIMEOUT_COUNT"
echo ""

if [[ $TIMEOUT_COUNT -gt 0 ]]; then
  echo "Timed out matrices (exceeded ${TIMEOUT_SECONDS}s = $((TIMEOUT_SECONDS/3600))h):"
  for matrix in "${TIMEOUT_MATRICES[@]}"; do
    echo "  - $matrix (log: $LOG_DIR/${matrix}.log)"
  done
  echo ""
fi

if [[ $FAILED_COUNT -gt 0 ]]; then
  echo "Failed matrices:"
  for matrix in "${FAILED_MATRICES[@]}"; do
    echo "  - $matrix (log: $LOG_DIR/${matrix}.log)"
  done
  echo ""
fi

if [[ $FAILED_COUNT -gt 0 ]] || [[ $TIMEOUT_COUNT -gt 0 ]]; then
  echo "Some benchmarks did not complete successfully."
  exit 1
else
  echo "All benchmarks completed successfully!"
  exit 0
fi
