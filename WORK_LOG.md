# Cholesky project work log

This is the running laundry list for the progress update to Dr. Chechmi.
Record completed work, evidence, decisions, unsuccessful experiments, and remaining
work. A function existing in the repository does not by itself mean a milestone
is complete.

## Requested milestones

1. Numba dense Cholesky baseline — implemented and verified locally.
2. Optimized Numba dense Cholesky — implemented, tested, and benchmarked locally.
3. Standalone sparse CSC Cholesky — existing numeric code and partial symbolic
   helpers; the existing benchmark borrows structure from CHOLMOD.
4. Supernodal sparse Cholesky — existing prototype and unfinished symbolic wiring.

## 2026-09-28 — repository review and dense baseline (previous session)

- [x] Reviewed the benchmark, utility, compiler, and GPU multiplication code.
- [x] Located sparse numeric Cholesky variants in `benchmark/cholesky.py`.
- [x] Identified the CHOLMOD-derived factor pattern and dependencies in the
  current sparse benchmark, hardcoded matrix paths, and missing inspector import.
- [x] Identified unfinished postordering/supernodal integration in the symbolic
  utilities. These findings have not yet been repaired or runtime-validated.
- [x] Added `requirements-cpu.txt` and installed a local `.venv` with Python
  3.11.15, NumPy 2.1.3, Numba 0.61.0, SciPy 1.15.1, and threadpoolctl 3.6.0.
- [x] Implemented a serial, unblocked dense Numba factorization with no fastmath.
- [x] Added validation for shape, real dtype, finiteness, symmetry, and positive
  pivots; float64 output; input preservation; explicit exception behavior.
- [x] Added seven test methods covering a known factor, random SPD matrices,
  diagonal/scalar matrices, ill-conditioning, layouts, and invalid inputs.
- [x] Compared against both NumPy and SciPy and checked reconstruction error.
- [x] Added a seeded benchmark with warmups, median timings, separate kernel and
  validated API timings, CSV output, and environment metadata.
- [x] Preserved the initial measurements in `results/dense_baseline.csv` and its
  metadata sidecar. Sizes: 32, 128, 256, 512; seed: 42; seven timed calls/method.
- [x] Replaced BLAS matrix multiplication in generation/residual checks with
  direct einsum after encountering spurious macOS warnings. No warnings suppressed.
- [x] Documented that Apple Accelerate thread counts are not verified by
  threadpoolctl here; library comparisons are not guaranteed equal-thread tests.
- [x] Passed seven tests with warnings treated as errors, benchmark execution,
  dependency consistency (`pip check`), and whitespace checks.
- [x] Added setup, example usage, timing methodology, and limitations to the README.

## 2026-09-28 — optimized dense milestone (current session)

### Scope and decisions

- [x] Preserve the original baseline kernel for direct, same-run comparison.
- [x] Create this detailed work log before starting optimization.
- [x] Implement and measure a pure Numba optimization; retain serial execution
  and strict floating-point semantics so the comparison isolates algorithm/layout.
- [x] Reuse the same input-validation contract for both public functions.
- [x] Test block boundaries, remainder blocks, layout/dtype handling, rejection
  behavior, numerical accuracy, and input preservation.
- [x] Benchmark candidate block sizes before choosing a default; save evidence.
- [x] Run final baseline/optimized/library comparisons on identical matrices,
  report both kernel and public API timings, and preserve run metadata.
- [x] Update documentation and write a ready-to-use progress summary.

### Experiment results and validation

Exploratory timings below are five-call medians in milliseconds from console
experiments, seed 42, sizes 32/128/256/512/1024 in that order. They are useful
for design selection, not the final headline comparison (different runs).

| Approach | 256 | 512 | 1024 | Decision |
| --- | ---: | ---: | ---: | --- |
| Initial blocked, 2D element indexing, block 64 | 1.2013 | 9.4683 | 74.4669 | Insufficient improvement; slower than baseline at smaller sizes |
| Contiguous row views, block 64 | 0.5195 | 4.6779 | 38.6807 | Retain row-view helper |
| Four ordered contributions per destination load/store, block 64 | 0.4103 | 3.4835 | 25.6241 | Retain four-row update and scalar remainder |

- The final approach factors panels in transposed storage, updates contiguous
  row tiles, and reuses a loaded destination value for four contributions.
- It does not call BLAS, delegate the factorization to NumPy/SciPy, enable
  fastmath, or run parallel loops. It returns conventional lower-triangular L.
- Both public APIs now share validation; the baseline numeric loop is unchanged.
- Twenty tests passed with warnings treated as errors after the first optimized
  implementation. Coverage includes 144 dimension/block-size combinations,
  partial panels and four-update remainders, late pivot failure, lower-triangle
  semantics, input preservation, and solving a system with three right-hand sides.
- Extended the benchmark to support block-size sweeps, interleaved method order,
  NumPy factor agreement, min/max timings, and raw timing samples in metadata.
- Saved the full sweep in `results/dense_block_sweep.csv` and its metadata JSON:
  sizes 32/128/256/512/1024, blocks 16/32/64/128, seed 42, 11 samples per method.
- Chose default block size 64. Block 32 was slightly faster at 128/256/512;
  block 64 improved the 1024 result from 32.5870 ms (block 32) to 26.8866 ms,
  close to block 128's 26.6491 ms. No claim that one size is optimal everywhere.
- Added a multi-panel ill-conditioning test with prescribed condition numbers
  10^4, 10^8, and 10^10 and block sizes 3/16/32/64. Final test count: 21 methods,
  including 144 boundary/remainder subcases and 12 multi-panel conditioning subcases.

### Final measurements

`results/dense_optimized.csv`: default block 64, seed 42, 15 timed calls per
method after warmup. Methods interleaved in seeded random order. Values below
are medians in milliseconds; speedup is baseline divided by optimized.

| Dimension | Baseline kernel | Optimized kernel | Kernel speedup | Baseline API | Optimized API | API speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.0029 | 0.0025 | 1.15x | 0.0151 | 0.0150 | 1.01x |
| 128 | 0.1306 | 0.0584 | 2.24x | 0.1942 | 0.1223 | 1.59x |
| 256 | 1.1322 | 0.4086 | 2.77x | 1.4048 | 0.6848 | 2.05x |
| 512 | 10.1530 | 3.6960 | 2.75x | 11.4549 | 4.9075 | 2.33x |
| 1024 | 94.2360 | 26.8362 | 3.51x | 99.2771 | 32.1701 | 3.09x |

- Maximum reconstruction residual across all methods in this run: 8.344e-16.
  Optimized maximum: 8.264e-16. Residual is `||A - L L.T||_F / ||A||_F`.
- SciPy remains faster: 0.8318 ms at 512 and 4.2557 ms at 1024. These are library
  reference timings, not evidence of matched threading; Accelerate thread counts
  are unverified. The baseline-versus-optimized comparison uses serial Numba loops.
- `results/dense_optimized_odd_sizes.csv`: second seed 7, dimensions
  65/129/257/513/1025, seven samples per method. Kernel speedups were
  1.49x/2.25x/3.13x/3.24x/3.62x respectively; maximum residual across methods
  was 8.173e-16. This checks another matrix sample and nondivisible dimensions.
- Preserved the original baseline results; final speedups use the newly measured
  same-run baseline, not measurements from the previous session.
- Both new comparison runs and the block sweep save raw timing samples,
  min/max/median, versions, platform, detected thread pools, library build details,
  seed, sizes, and block sizes. No new package dependencies were added this session.

### Commands and outcomes

All commands run from the repository root on the local macOS arm64 environment.

```sh
.venv/bin/python -W error -m unittest discover -s tests -v
# PASS: 21 test methods; includes boundary and conditioning subtests.

.venv/bin/python -W error -m benchmark.cholesky_dense_benchmark \
  --sizes 32 128 256 512 1024 --block-sizes 16 32 64 128 --repeats 11 \
  --csv results/dense_block_sweep.csv
# PASS: all factors matched NumPy and passed residual checks.

.venv/bin/python -W error -m benchmark.cholesky_dense_benchmark \
  --repeats 15 --csv results/dense_optimized.csv
# PASS: final default-block comparison, seed 42.

.venv/bin/python -W error -m benchmark.cholesky_dense_benchmark \
  --sizes 65 129 257 513 1025 --seed 7 --repeats 7 \
  --csv results/dense_optimized_odd_sizes.csv
# PASS: independent seed and nondivisible dimensions.

git diff --check
# PASS: no whitespace errors.
```

### Deliverable inventory

- `benchmark/cholesky_dense.py`: original numeric baseline retained; validation
  extracted into a shared helper without changing its contract.
- `benchmark/cholesky_dense_optimized.py`: new blocked implementation, inlined
  row-update helpers, default block size, validated public API.
- `benchmark/cholesky_dense_benchmark.py`: baseline/optimized/reference comparisons,
  block sweeps, interleaving, correctness checks, and richer saved evidence.
- `tests/test_cholesky_dense.py`: shared baseline/optimized contract tests plus
  optimized-specific numerical and boundary cases.
- `benchmark/README.md`: setup, usage, algorithm explanation, benchmark method,
  tuning command, result inventory, and limitations.
- `WORK_LOG.md`: detailed running laundry list, including exploratory designs.
- `PROGRESS_UPDATE.md`: concise, ready-to-use draft update for Dr. Chechmi.
- `results/dense_block_sweep.*`, `results/dense_optimized.*`, and
  `results/dense_optimized_odd_sizes.*`: CSV results and metadata sidecars.

### Interpretation and limits

- This completes a locally verified optimized dense milestone, not the entire
  four-part project. Sparse CSC and supernodal factorization remain incomplete.
- The main benchmark matrices are well-conditioned synthetic SPD matrices.
  Ill-conditioned inputs are covered by correctness tests, not performance claims.
- Tiny matrices have little end-to-end improvement. At size 32, validation and
  dispatch dominate, and the kernel timings are too short for strong claims.
- No threading, GPU, BLAS-backed custom algorithm, relaxed arithmetic, sparse
  symbolic repair, or broad cross-machine tuning was included in this milestone.
- The chosen block size, timings, and speedups are machine-dependent. Raw samples
  are retained for inspection; this is not a statistical confidence study.

### Remaining work beyond this milestone

- Complete independent CSC symbolic analysis, then connect the sparse numeric kernel.
- Validate sparse factorization against a trusted reference and real sparse matrices.
- Complete supernodal symbolic analysis, storage, and numeric factorization.
- Broaden performance evaluation to additional machines and matrix families.
