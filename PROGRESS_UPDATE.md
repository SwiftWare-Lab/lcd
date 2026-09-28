# Draft progress update — dense Cholesky

Dr. Chechmi,

I have implemented and locally validated the first two items from our meeting:
a basic dense Cholesky factorization in Numba and a separate optimized version.
Both compute a lower-triangular factor L for real symmetric positive-definite
matrices, preserve the input, and report invalid inputs or nonpositive pivots.

The baseline uses a straightforward scalar algorithm. The optimized version uses
blocked updates with contiguous memory access and reuses each loaded destination
value across four updates. Both run serially in Numba, without fastmath or a call
to a library Cholesky implementation. I retained the original baseline for direct
comparison and selected a default block size of 64 after measuring four choices.

In a same-run comparison on the local Mac, the optimized factorization kernel
achieved the following median times after compilation warmup:

| Matrix size | Baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| 128 × 128 | 0.131 ms | 0.058 ms | 2.24x |
| 256 × 256 | 1.132 ms | 0.409 ms | 2.77x |
| 512 × 512 | 10.153 ms | 3.696 ms | 2.75x |
| 1024 × 1024 | 94.236 ms | 26.836 ms | 3.51x |

These measurements use identical seeded matrices, 15 timed calls per method,
and interleaved execution order. Allocation is included; compilation, matrix
generation, and correctness checks are excluded. Including public API validation,
the 1024 × 1024 improvement is 3.09x. Tiny matrices show little end-to-end benefit.
A second run with another seed and dimensions that do not divide evenly into
blocks also showed improvements, reaching 3.62x kernel speedup at 1025 × 1025.

All 21 test methods pass with warnings treated as errors. Coverage includes
comparison with NumPy and SciPy, 144 matrix-size/block-size combinations,
ill-conditioned matrices spanning multiple panels, input preservation, invalid
inputs, and a linear-system solve with three right-hand sides. Relative
reconstruction errors in the final benchmarks remained below 10^-15.

SciPy is still faster than the optimized custom implementation. Its underlying
library thread count could not be verified on this Mac, so I am treating it as a
reference rather than claiming a controlled equal-thread comparison. The reported
custom-kernel speedups compare two serial Numba implementations. Results are local
measurements on synthetic dense matrices, not conclusions about sparse workloads
or other hardware.

The next step is standalone sparse CSC symbolic analysis: compute the factor
pattern and dependencies directly from the input matrix, then connect them to
the existing numeric kernel. The current sparse benchmark obtains its structure
from a CHOLMOD factor, so that stage is not yet independent. Supernodal symbolic
analysis and factorization will follow once the CSC path is validated.

The detailed change list, exploratory results, reproducible commands, and remaining
work are in [WORK_LOG.md](WORK_LOG.md). Final evidence is saved in
[the comparison CSV](results/dense_optimized.csv),
[the block-size sweep](results/dense_block_sweep.csv), and
[the second-seed comparison](results/dense_optimized_odd_sizes.csv), each with an
environment and raw-timing metadata sidecar.
