# Loop-Carried Dependence Transformation for Parallel Sparse Solvers on GPUs

This repository contains the artifact for:

> **Loop-Carried Dependence Transformation for Parallel Sparse Solvers on GPUs**  
> Da Ma, Mohammad Mahdi Salehi, Amir Mohammad Tavakkoli, Samirasadat
> Jamalidinan, Hossein Albkri, Mary Hall, and Kazem Cheshmi  
> *International Conference for High Performance Computing, Networking,
> Storage and Analysis, SC'26* **(Accepted)**

The artifact provides a Python decorator that transforms sparse loops with
loop-carried dependences into dependency-aware Numba-CUDA executions. It
identifies ordering dependences, recognizes supported associative reductions,
constructs a runtime task DAG, and generates a CUDA kernel and host launcher.

The implementation is experimental. Wavefront execution is the recommended
mode for correctness-oriented evaluation. The sync-free executor is also
available, but its busy-wait implementation still requires stronger CUDA
forward-progress and memory-publication guarantees.

## LCD representation

The Loop-Carried Dependence (LCD) representation is the compiler's frontend
view of the input operation. It records the loop structure, iterator bounds,
array accesses, predicates, and statement ordering needed to identify
dependences between different loop iterations.

The iterator selected by `block_map` represents one logical GPU task. Loops
outside that iterator remain host-side control flow; if an enclosing iterator
is referenced by the mapped operation, LCD treats it as a parameter. LCD is
then converted into the backend dependence representation used to classify
ordering and reduction dependences.

Implementation details and a module-level guide are available in
[compiler/README.md](compiler/README.md).

## Requirements

The artifact uses NumPy, SciPy, SymPy, Z3, Numba, and an NVIDIA CUDA driver.
Before running a demo, verify that Numba can initialize CUDA:

```bash
python -c "from numba import cuda; print(cuda.is_available())"
```

The development environment for this artifact uses NVIDIA's `cuda-python`
driver binding:

```bash
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
```

Set this variable before Python imports `numba.cuda` if Numba's legacy ctypes
binding fails in your environment.

## Using the decorator

Mark the loop whose iterations should become dependency-aware GPU tasks:

```python
import numpy as np

from compiler.decorator import parallel


properties = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
    ("Ap", "Ai"): ["triangularity"],
}


@parallel(block_map="i", function_props=properties, execution_mode="wavefront", threads_per_block=128)
def sptrsv_csc(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[i]) / Ax[Ap[i]]
        for j in range(Ap[i] + 1, Ap[i + 1]):
            x[Ai[j]] += Ax[j] * x[i]
    return x
```

Call the decorated function with ordinary host values:

```python
x = sptrsv_csc(Ap, Ai, Ax, b, n)
```

The wrapper inspects the concrete sparsity, prepares the selected executor,
transfers arrays, launches the generated CUDA code, and copies explicitly
returned results back to the host.

The source extractor currently expects `@parallel(...)` to be written on one
physical line. An updated array should be explicitly returned when its final
value is required on the host.

### Decorator parameters

| Parameter | Meaning |
| --- | --- |
| `block_map`, `loop`, `block_loop` | Name of the loop iterator mapped to logical GPU tasks. |
| `function_props` | Symbolic properties used during dependence analysis and DAG simplification. |
| `execution_mode` | `"wavefront"` or `"sync_free"`. The demos use wavefront mode. |
| `threads_per_block` | CUDA thread-block size. Default: `128`. |
| `thread_map`, `thread_loop` | Optional inner-loop iterator distributed across threads in a task block. |
| `thread_unroll`, `thread_unroll_factor` | Compile-time unrolling factor for the thread-mapped loop. Default: `1`. |
| `sync_free_ready_count` | Dependency-count value at which a sync-free task becomes ready. Default: `0`. |
| `validate_dag_endpoints` | Enables runtime bounds checks for generated DAG endpoints. |

An unrolling factor greater than one requires a thread-mapped loop. With
`thread_unroll=1`, the compiler emits the ordinary thread-strided loop without
unrolling-specific temporaries.

### Function properties

Function properties communicate facts about sparse index structures that
cannot be inferred from ordinary Python syntax. For example:

```python
properties = {
    "A_indptr": ["monotonically_increasing"],
    ("A_indptr", "A_indices"): [
        ("segmented_monotonically_increasing", "n"),
    ],
}
```

These properties allow the dependence analysis to reason about compressed
sparse segments and the ownership of pointer iterations.

### Execution modes

Wavefront execution groups independent DAG nodes into topological levels. The
host launcher issues one kernel for each level and synchronizes between levels.

Sync-free execution launches the task blocks together. Each task waits on its
incoming-dependency count and atomically releases its successors when it
finishes. CUDA simulation should not be used for this busy-wait executor.

## Demos

The [`compiler/demo`](compiler/demo) directory contains complete sparse
operations that exercise the decorator and CUDA code-generation path. Run the
following commands from the repository root. If required by your Numba
installation, set `NUMBA_CUDA_USE_NVIDIA_BINDING=1` before starting Python.

### Sparse triangular solve

[`compiler/demo/sptrsv.py`](compiler/demo/sptrsv.py) contains CSR and CSC
SpTRSV operations. The CSR operation consumes previously solved rows, while
the CSC operation uses atomic updates to future rows.

```bash
python -m compiler.demo.sptrsv
```

### Gauss-Seidel

[`compiler/demo/gauss_seidel.py`](compiler/demo/gauss_seidel.py) contains CSR
Gauss-Seidel operations with either a separate diagonal or a diagonal stored
in the sparse matrix. The outer sweep remains host-side, while the row loop is
mapped to DAG tasks.

```bash
python -m compiler.demo.gauss_seidel
```

### Incomplete LU factorization

[`compiler/demo/spilu0.py`](compiler/demo/spilu0.py) contains CSR and CSC ILU0
operations and helper functions that apply the decorator with a selected
execution mode and thread-block size.

```bash
python -m compiler.demo.spilu0
```
