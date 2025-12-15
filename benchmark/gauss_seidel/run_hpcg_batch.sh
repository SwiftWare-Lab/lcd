#!/bin/bash
# Local benchmark script for running HPCG benchmarks on individual matrices
# This avoids single point of failure by running each matrix separately

set -euo pipefail

# Configuration
ROOT=${ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-$ROOT/matrix_list.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/hpcg_results}
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/logs}

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Read matrix list (skip comments and empty lines)
readarray -t MATRICES < <(grep -v '^\s*#' "$MATRIX_LIST_FILE" | sed '/^\s*$/d')

if [[ ${#MATRICES[@]} -eq 0 ]]; then
  echo "No matrices listed in $MATRIX_LIST_FILE"
  exit 1
fi

echo "========================================"
echo "HPCG BATCH BENCHMARK"
echo "========================================"
echo "Found ${#MATRICES[@]} matrices to benchmark"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""

# Track results
SUCCESS_COUNT=0
FAILED_COUNT=0
TIMEOUT_COUNT=0
FAILED_MATRICES=()
TIMEOUT_MATRICES=()

# Timeout duration (default: 30 minutes = 1800 seconds)
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}

# Run benchmark for each matrix
for i in "${!MATRICES[@]}"; do
  MATRIX="${MATRICES[$i]}"
  MATRIX_NAME=$(basename "$MATRIX" .mtx)
  
  echo "[$((i+1))/${#MATRICES[@]}] Processing: $MATRIX_NAME"
  
  # Create log file for this matrix
  LOG_FILE="$LOG_DIR/${MATRIX_NAME}.log"
  
  # Run benchmark and capture output
  START_TIME=$(date +%s)
  
  # Use timeout command to limit execution time
  # timeout returns 124 if the command times out
  if timeout $TIMEOUT_SECONDS bash -c "cd '$ROOT' && $PYTHON_BIN hpcg.py --matrix '$MATRIX' --output-dir '$OUTPUT_DIR'" > "$LOG_FILE" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
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

# Merge all individual CSV files into one
echo "Merging results..."
MERGED_CSV="$OUTPUT_DIR/hpcg_results_merged.csv"

# Get header from first successful result
FIRST_CSV=$(find "$OUTPUT_DIR" -name "hpcg_results_*.csv" -type f | head -n 1)
if [[ -n "$FIRST_CSV" ]]; then
  head -n 1 "$FIRST_CSV" > "$MERGED_CSV"
  
  # Append data from all CSVs (skip header)
  for csv_file in "$OUTPUT_DIR"/hpcg_results_*.csv; do
    if [[ -f "$csv_file" ]] && [[ "$csv_file" != "$MERGED_CSV" ]]; then
      tail -n +2 "$csv_file" >> "$MERGED_CSV"
    fi
  done
  
  echo "Merged results saved to: $MERGED_CSV"
fi

# Print summary
echo ""
echo "========================================"
echo "BENCHMARK SUMMARY"
echo "========================================"
echo "Total matrices: ${#MATRICES[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Timed out: $TIMEOUT_COUNT"
echo ""

if [[ $TIMEOUT_COUNT -gt 0 ]]; then
  echo "Timed out matrices (exceeded ${TIMEOUT_SECONDS}s):"
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
