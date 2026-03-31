# ppbench

Standalone host-side preprocessing benchmark bundle for Trillium.

## Files

- `host_inspector_numba.py`: local copy of the Numba host inspector kernels.
- `benchmark_numba_preprocess.py`: benchmarks Numba preprocessing and writes CSV.
- `sbatch_ppbench_array.sh`: Slurm array launcher.
- `merge_ppbench_csv.py`: merges per-task CSV shards.

## Requirements

- Python environment with `numpy`, `scipy`, and `numba`.
- Matrix files available on the cluster.
- Matrix list file available on the cluster.

## Supported matrix lists

The array script works with either of these list files:

- `matrix_list_orig.txt`
- `matrix_list.txt`

List entries can be:

- absolute paths to `.mtx` files
- relative paths to `.mtx` files
- matrix names without `.mtx`, together with `--matrix-root`

## Direct run

```bash
python benchmark_numba_preprocess.py \
  --matrix-list matrix_list_orig.txt \
  --matrix-root /path/to/matrices \
  --output results.csv
```

## Array run

```bash
sbatch sbatch_ppbench_array.sh matrix_list_orig.txt /path/to/matrices
```

Useful environment variables:

- `NUM_ARRAY_JOBS`: number of array tasks, default `50`
- `PPBENCH_REPEATS`: timed repetitions per matrix, default `5`
- `PPBENCH_BAND_WIDTH`: h-level band width, default `64`
- `PPBENCH_VENV`: python venv path, default `/scratch/dama/venv`
- `OUT_DIR`: output directory for task CSVs

Example:

```bash
NUM_ARRAY_JOBS=32 PPBENCH_REPEATS=3 sbatch --array=0-31 \
  sbatch_ppbench_array.sh matrix_list.txt /path/to/matrices
```

## Merge task outputs

```bash
python merge_ppbench_csv.py \
  --input-glob "ppbench-results/matrix_list_orig/ppbench-*.csv" \
  --output ppbench-results/matrix_list_orig_merged.csv
```
