# ILU0 row-dependency benchmark

This guide explains what [`ilu0_parallel_benchmark.py`](ilu0_parallel_benchmark.py) measures, how to run it, and how to interpret its results. The short version: across the four currently selected matrices, the CPU path is faster than either **end-to-end** GPU path. Sync-free is faster than wavefront, and its prepared path is fastest overall when preparation is reused.

## What is being measured?

This is **not a numerical ILU(0) factorization benchmark**. It uses the sparsity pattern of a CSR matrix to run a synthetic row-dependency recurrence. Matrix values are not used, and no incomplete L or U factors are produced. It is useful for studying the overhead and scheduling behavior of row dependencies, but its numbers must not be presented as full ILU(0) factorization performance.

For row `i`, each column index `j < i` in that row creates a dependency `j → i`: row `i` must wait until row `j` is complete. The benchmark computes

```text
scale(i)   = 0.5 / (number of stored entries in row i + 1)
result[i]  = 1 + Σ scale(i) × result[j], for stored columns j < i
```

The original CPU call and end-to-end GPU calls start with a zero-filled result array. Prepared CPU and GPU calls reuse their output buffers; dependency-correct execution overwrites every row before a successor reads it. The scale keeps values bounded on large matrices so a finite CPU reference can be compared with GPU output. The GPU implementations are defined in [`compiler/transform_tests/ilu0.py`](../compiler/transform_tests/ilu0.py); the benchmark's CPU reference is in [`ilu0_parallel_benchmark.py`](ilu0_parallel_benchmark.py).

## The six reported methods

| Method | Processor | Scheduling and what is timed |
| --- | --- | --- |
| `cpu-row-deps` | CPU | Sequential Python recurrence on CSR indices. Includes allocating/copying its output array; excludes loading the `.mtx` file. |
| `cpu-row-deps-prepared` | CPU | Reuses loaded CSR arrays and a preallocated output buffer; times only the sequential Python recurrence and call overhead. |
| `wavefront-e2e` | GPU + CPU setup | Constructs the dependency DAG and levels, transfers data, launches one GPU kernel per level, synchronizes, and copies the result back. |
| `wavefront-prepared` | GPU + CPU launch loop | Reuses the prepared DAG and device arrays. Times the host loop that launches and synchronizes each wavefront; no matrix upload or result download. |
| `sync-free-e2e` | GPU + CPU setup | Constructs the dependency DAG and counters, transfers data, launches the GPU work, synchronizes, and copies the result back. |
| `sync-free-prepared` | GPU + CPU reset/launch | Reuses the prepared DAG and device arrays. Each repetition **does** copy the original dependency counters to the GPU, then launches and synchronizes; no matrix upload or result download. |

“End-to-end” here means **from an already loaded CSR matrix to a returned result**. It does **not** include downloading or reading Matrix Market files, process startup, or JIT compilation. The benchmark validates and warms the GPU paths before timing. “Prepared” is **not pure GPU-kernel time**: GPU paths still include host launch/synchronization overhead, sync-free includes the mandatory counter reset, and CPU prepared includes Python call and loop overhead. CPU prepared reuses its output array but does not require DAG setup or data transfer. Preparation can be reused when repeatedly running the same sparsity pattern, but a prepared number alone is not a fair comparison with a full CPU or GPU call.

The CPU baseline is Python, not an optimized compiled CPU ILU(0) implementation. Compare it only with this specific recurrence and timing scope.

## Wavefront versus sync-free

Both modes use the same row-dependency edges; they differ in when and where readiness is decided.

- **Wavefront:** preparation groups rows into levels. All rows in a level can run together because their predecessors are in earlier levels. The host launches one kernel per level and waits before launching the next. It has no per-row GPU spin-wait, but many levels mean many launches and synchronizations.
- **Sync-free:** preparation builds each row's predecessor count and successor list. A GPU block waits until its count reaches zero, computes its row, then atomically decrements its successors' counts. Rows can become ready asynchronously within one kernel launch. “Sync-free” means no **global wavefront barrier**; it does **not** mean no synchronization. It uses atomics, spinning, and device memory fences.

The producer and consumer fences in [`compiler/cuda_lowering.py`](../compiler/cuda_lowering.py) are important: a successor must not read a predecessor's output before that output is visible. This follows [NVIDIA's atomic ready-flag memory-ordering pattern](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html). The two approaches also have different preparation costs: computing full levels takes additional DAG traversal, while sync-free must maintain counters during execution. The present benchmark does **not** isolate DAG-construction time as a separate reported metric, so it cannot establish a general “faster DAG discovery” claim.

## Matrices and how to run

Run from the repository root with the project environment and a CUDA-capable GPU:

```bash
./.venv/bin/python utils/dl_matrices.py 1138_bus bcsstk08 bcsstk09 bcsstk36
./.venv/bin/python benchmark/ilu0_parallel_benchmark.py
```

The downloader writes under `benchmark/matrices/suitesparse/` and records selected relative paths in `manifest.txt`. It skips matrices already present, so the command above can also be used to select existing downloads without fetching them again. The benchmark uses that manifest when present; if a listed matrix is missing, it fails instead of silently substituting a generated matrix. With no manifest, it discovers `.mtx` files in that directory. The current manifest selects all four matrices shown below.

The second command validates both GPU modes against the finite CPU reference for each matrix before reporting any throughput. It checks an end-to-end call and two prepared calls per GPU mode. A failed comparison stops the benchmark. For a one-off method call, `benchmark` is Triton's report wrapper and `benchmark.fn(...)` is the underlying benchmark function:

```python
from benchmark.ilu0_parallel_benchmark import benchmark

rows_per_second, low, high = benchmark.fn(
    matrices="benchmark/matrices/suitesparse/bcsstk36/bcsstk36.mtx",
    provider="sync-free-e2e",
)
```

The normal command is preferable when collecting the complete report. It performs 5 warm-up calls and 30 timed calls **per method and matrix**, measures wall-clock time with `time.perf_counter()`, and reports the median. The plotted throughput is `row_count / elapsed_seconds`; its lower and upper bounds come from the 80th and 20th **time** percentiles, respectively, because faster time means higher throughput. All six methods use the same wall-clock timer, and GPU calls synchronize before timing ends.

Results are written to [`results/ilu0-row-dependencies-validated.csv`](results/ilu0-row-dependencies-validated.csv) and a PNG in the same directory. The CSV contains rows/second, while the run also prints median milliseconds.

## Current results and interpretation

The selected SuiteSparse matrices span different sizes and numbers of lower-triangular entries (the entries that induce dependencies):

| Matrix | Rows | Stored entries (CSR `nnz`) | Lower-triangular entries |
| --- | ---: | ---: | ---: |
| `1138_bus` | 1,138 | 4,054 | 1,458 |
| `bcsstk08` | 1,074 | 12,960 | 5,943 |
| `bcsstk09` | 1,083 | 18,437 | 8,677 |
| `bcsstk36` | 23,052 | 1,143,140 | 560,044 |

One complete five-method run on 2026-09-30 passed the CPU-reference check for **all four matrices**. Median wall-clock times, in milliseconds (lower is better), were:

| Matrix | CPU | Wavefront E2E | Sync-free E2E | Wavefront prepared | Sync-free prepared |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1138_bus` | 1.384 | 3.158 | 2.110 | 1.443 | 0.134 |
| `bcsstk08` | 2.148 | 8.401 | 3.954 | 5.125 | 0.429 |
| `bcsstk09` | 2.722 | 9.789 | 4.489 | 5.787 | 0.382 |
| `bcsstk36` | 125.766 | 547.433 | 297.310 | 290.369 | 50.536 |

Hardware: AMD Ryzen 9 9955HX3D CPU and NVIDIA GeForce RTX 5080 Laptop GPU (driver 596.21). Software: Python 3.12.3, NumPy 2.4.4, SciPy 1.15.1, Triton 3.7.0, and Numba 0.65.1. The [saved CSV](results/ilu0-row-dependencies-validated.csv) gives the corresponding rows/second rather than milliseconds.

A separate one-matrix check of the newly added CPU-prepared method on `bcsstk36` measured **125.644 ms**, versus **126.544 ms** for the existing CPU method. Both passed validation. This single comparison is not part of the saved five-method CSV; rerun the full benchmark to generate a six-method report.

On each of these four matrices, sync-free beats wavefront within the same timing scope, while the CPU loop beats both **end-to-end** GPU calls. Sync-free prepared is faster than CPU, but that comparison excludes most GPU setup and transfers, so it does not demonstrate an end-to-end GPU speedup. The small matrices emphasize launch/setup costs; `bcsstk36` is the only large case in this set. One run on one machine is not a general ranking of the methods. The previous single-matrix `bcsstk36` run had different timings, so keep run-to-run variation in mind.

Earlier `ilu0-parallel-row-dependencies.csv` results and slides based on them should be treated as obsolete. The earlier recurrence overflowed on the downloaded matrix, sync-free output could be intermittently wrong, and prepared timing used a different timing API. The current implementation bounds the recurrence, uses CUDA fences around the ready-counter handoff, resets sync-free counters for each prepared run, validates GPU answers, and uses one timer across providers. In addition to the benchmark's checks, 50 consecutive prepared sync-free runs matched the CPU reference during the review.

## What would make this a stronger performance study?

Expand beyond these four matrices, especially with more medium and large cases and measured dependency-depth/width profiles; report DAG preparation, transfer, execution, and result retrieval separately; add an optimized compiled CPU baseline; and benchmark actual numerical ILU(0) if the intended claim is about factorization. Repeat complete runs to quantify run-to-run variation before publishing a performance chart.

Focused correctness tests can be rerun with:

```bash
./.venv/bin/python -m unittest \
    compiler.transform_tests.test_cuda_lowering \
    compiler.transform_tests.test_ilu0_e2e -q
```
