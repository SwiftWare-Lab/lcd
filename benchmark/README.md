
# benchmark

## Dense Cholesky: baseline and optimized milestones

`cholesky_dense.py` implements a serial, unblocked Numba baseline for
`A = L @ L.T`. It accepts real symmetric positive-definite matrices, returns a
new float64 lower-triangular factor, and leaves the input unchanged.
`cholesky_dense_optimized.py` adds a separately named blocked Numba implementation.
The first two of four milestones are implemented and verified locally. Standalone
sparse CSC (symbolic + numeric) and supernodal sparse Cholesky remain to be completed.
See [the work log](../WORK_LOG.md) for the detailed history and experiment evidence.

From the repository root, use Python 3.10–3.13 to create a CPU-only environment:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmark.cholesky_dense_benchmark --repeats 15 --csv results/dense_optimized.csv
```

This environment is sufficient for the dense milestone; existing GPU and
CHOLMOD experiments have additional dependencies. No matrix downloads are needed.

```python
import numpy as np
from benchmark.cholesky_dense import cholesky_dense_numba
from benchmark.cholesky_dense_optimized import cholesky_dense_numba_optimized

A = np.array([[4.0, 2.0], [2.0, 3.0]])
L = cholesky_dense_numba(A)
L_optimized = cholesky_dense_numba_optimized(A)  # default block_size=64
assert np.allclose(A, L @ L.T)
assert np.allclose(L, L_optimized)
```

The public function checks shape, real numeric dtype, finite values, and symmetry
(`rtol=1e-12`, `atol=1e-14`); nonpositive pivots raise `numpy.linalg.LinAlgError`.
Empty matrices are rejected. Integer and float32 inputs are promoted to float64.
For accepted roundoff-level asymmetry, the lower triangle defines the factor.
Both functions share these checks. The optimized function also accepts a positive
integer `block_size`; booleans/nonintegers are rejected, blocks larger than the
matrix are allowed, and partial final blocks are supported.

### What is optimized

The baseline computes each entry using a scalar dot-product loop. The optimized
implementation copies the transpose into row-contiguous working storage, factors
a panel, and updates tiles of the remaining matrix. Contiguous row views simplify
the inner loops. Each trailing update applies four panel contributions to a loaded
destination value before storing it, while preserving the order of subtraction.
It then converts the upper-triangular working factor into lower-triangular output.
Both algorithms take O(n^3) work and O(n^2) storage and preserve the input.

The optimization uses serial Numba loops with no BLAS factorization call,
parallelism, or fastmath. Block size 64 was chosen from a local 16/32/64/128 sweep:
32 did slightly better at intermediate sizes, while 64 improved the 1024 case
and was close to 128 there. This default is a measured compromise, not a universal
optimum or a per-machine autotuner. Small inputs may see little benefit because
allocation, dispatch, and validation are a large part of their runtime.

### Benchmark methodology

The benchmark uses seeded `M @ M.T + n * I` matrices. It warms every method before
timing, interleaves methods in a seeded random order, reports medians/minima/maxima,
and checks agreement with NumPy plus the relative reconstruction residual outside
timed regions. Detected BLAS thread pools are limited to one thread; undetected
backends (including Apple Accelerate) have unverified thread counts, so these
results should not be described as a guaranteed equal-thread comparison.
Output allocation is
included; compilation, matrix generation, and correctness checks are excluded.
`numba_kernel` times the baseline on prepared inputs; `numba_validated` includes
the public function's validation/conversion overhead. `optimized_b64_kernel` and
`optimized_b64_validated` provide the corresponding optimized measurements.
NumPy and SciPy are optimized
library references; SciPy's finite checking is disabled for these prepared inputs.
The CSV sidecar records versions, platform, library build information, detected
thread pools, seed, block sizes, raw timing samples, and run settings.
Matrix generation and reconstruction checks
use direct `einsum` contractions independently of BLAS. This also avoids the
[macOS matrix-multiplication warnings reported upstream](https://github.com/numpy/numpy/issues/28687)
with the pinned NumPy version; no numerical warnings are suppressed.
For example, use `--sizes 32 128 256 --repeats 10 --seed 42` to customize a run.
To reproduce the block-size selection experiment:

```sh
.venv/bin/python -W error -m benchmark.cholesky_dense_benchmark \
  --sizes 32 128 256 512 1024 --block-sizes 16 32 64 128 --repeats 11 \
  --csv results/dense_block_sweep.csv
```

Tests cover known factors, random and ill-conditioned SPD matrices, reconstruction,
agreement with NumPy/SciPy, input preservation, array layouts, and invalid inputs.
The optimized tests additionally cover 144 matrix-size/block-size combinations,
late-panel pivot failure, multi-panel ill-conditioning, and a three-right-hand-side
linear solve. Run with `-W error` to treat warnings as failures.

Saved results: `results/dense_baseline.csv` is the original baseline run;
`results/dense_block_sweep.csv` records tuning; `results/dense_optimized.csv` is the
final same-run comparison; `results/dense_optimized_odd_sizes.csv` checks another
seed and dimensions that do not divide evenly into blocks. Each has a metadata
sidecar. Performance on these dense synthetic matrices does not predict sparse
performance. The next milestone is independent CSC symbolic analysis.

## Table of operations

-------------------
| Operation  | Format         | Description |
|------------|----------------|-------------|
| sptrsv-fwd | CSR            | Forward substitution |
| sptrsv-bwd | CSR            | Backward substitution |
| ilu0       | CSR            | Incomplete LU factorization |
| ilu0-solve | CSR            | Solve using ILU0 factorization |
| ilu0       | CSC            | Incomplete LU factorization |
| Gauss-seidel | CSR            | Gauss-Seidel iteration |
| Cholesky   | CSC            | Cholesky factorization |
| Cholesky   | Supernodal CSC | Cholesky factorization |
| sptrsv-fwd | Supernodal CSC | Solve using Cholesky factorization |






Each operation is added as a separate benchmark. Each operation has a 
separate benchmark and implemetation file. The benchmark file is named 
`<operation>_benchmark.py` and the implementation file is named 
`<operation>.py`. 


The benchmark file loads a list of sparse matrices from the file using:
```python
mtx_list = get_matrix_list("/home/kazem/UFDB/spd_list.txt", "/home/kazem/UFDB/")
```
The list of matrices is stored in the file `spd_list.txt` and the matrices are stored in the directory `UFDB`. 

Or a few matrix paths can be hard coded as below
```python
mtx_list = [
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx"
            ]
```
