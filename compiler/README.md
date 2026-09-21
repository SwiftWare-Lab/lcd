# Loop-Carried Dependence Compiler

The `compiler` package turns a Python loop with loop-carried array
dependences into a dependency-aware Numba-CUDA execution. The user marks one
logical task loop with `@parallel`; the compiler discovers ordering and
reduction dependences, constructs a task DAG, generates a CUDA kernel and host
launcher, transfers runtime data, executes the schedule, and restores the
function's return value on the host.

The current pipeline is experimental. Wavefront execution is the recommended
correctness path. Sync-free execution is available, but its busy-wait design
still needs stronger CUDA forward-progress and memory-publication safeguards.

## Pipeline

For each decorated function, the compiler:

1. normalizes aliases and raises the Python AST into the LCD frontend IR;
2. removes loops enclosing the block-mapped loop from LCD analysis and treats
   referenced enclosing iterators as host parameters;
3. converts LCD into the SPF dependence-analysis IR;
4. extracts read-after-write dependences and separates associative reductions;
5. generates and simplifies DAG edge rules for ordering dependences;
6. lowers the mapped loop and reductions to a Numba-CUDA kernel;
7. generates a wavefront or sync-free host launcher;
8. builds executor arrays, transfers values, launches, synchronizes, and
   copies explicitly returned results to the host.

Reduction dependences are excluded from the DAG when their updates can be
implemented by CUDA atomics. Ordering dependences remain DAG edges.

## Requirements

The compiler uses NumPy, SciPy, SymPy, Z3, Numba, and a working CUDA driver.
Check the selected Python environment before importing decorated demos:

```bash
python -c "from numba import cuda; print(cuda.is_available())"
```

The development environment used by this repository selects NVIDIA's
`cuda-python` driver binding:

```bash
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
```

This setting is needed if Numba's legacy ctypes binding crashes during
`cuda.to_device`. It must be set before Python imports `numba.cuda`.

## Basic usage

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

The source extractor currently expects the `@parallel(...)` expression to be
written on one physical line. Explicitly return arrays that must be copied
back. An input array modified only in place is not currently copied back to
the original host object unless it also appears in the return expression.

## Decorator parameters

| Parameter | Meaning |
| --- | --- |
| `block_map`, `loop`, `block_loop` | Required aliases naming the logical task-loop iterator. Exactly one matching loop must exist. |
| `function_props` | Symbolic properties used by SMT dependence analysis and ownership-based DAG simplification. |
| `execution_mode` | `"wavefront"` or `"sync_free"`. The default is `"sync_free"`; demos use `"wavefront"`. |
| `threads_per_block` | Positive integer controlling wavefront launch width. Default: `128`. |
| `thread_map`, `thread_loop` | Optional inner-loop iterator distributed across CUDA threads within each task block. |
| `thread_unroll`, `thread_unroll_factor` | Positive compile-time unroll factor for `thread_map`. Default: `1`. |
| `sync_free_ready_count` | Dependency-count value at which a sync-free task may start. Default: `0`. |
| `validate_dag_endpoints` | If true, emit runtime bounds checks around generated DAG endpoints. Default: false. |

### Block and thread mapping

Without `thread_map`, a wavefront launch uses `cuda.grid(1)` and assigns one
ready logical task to each CUDA thread. `threads_per_block` controls how those
task threads are grouped into CUDA blocks.

With `thread_map="j"`, one ready logical task is assigned to each CUDA block;
thread zero performs scalar task work, and the block's threads cooperatively
execute the selected `j` loop. In this configuration, `threads_per_block`
directly controls the number of workers for the inner loop.

```python
@parallel(block_map="i", thread_map="j", thread_unroll=4, threads_per_block=128, execution_mode="wavefront", function_props=properties)
def operation(...):
    for i in range(n):
        ...
        for j in range(start[i], end[i]):
            ...
```

For an unroll factor `U`, each thread handles up to `U` inner iterations per
loop step, and the generated stride is `cuda.blockDim.x * U`. With `U=1`, the
compiler emits the normal thread-strided loop and no unrolled copies or
unrolling-specific temporaries. An unroll factor greater than one requires a
`thread_map` iterator. The programmer must ensure that parallel inner-loop
iterations are independent or are protected by supported atomics.

Sync-free mode currently launches one CUDA block per DAG task. Without an
inner `thread_map`, it deliberately uses one thread per block. With an inner
thread mapping, `threads_per_block` supplies the worker count.

### Function properties

Currently used properties include:

```python
{
    "A_indptr": ["monotonically_increasing"],
    ("A_indptr", "A_indices"): [
        ("segmented_monotonically_increasing", "n"),
    ],
}
```

`monotonically_increasing` lets the DAG planner prove that half-open segments
`A_indptr[i]:A_indptr[i + 1]` have unique owners. The segmented property says
that `A_indices` is nondecreasing inside each of the first `n` segments.
`triangularity` describes the compressed-pointer/index relationship used by
the triangular-solve CSC example.

## Execution modes

### Wavefront

The CPU inspector computes DAG successors and incoming-edge counts. A
topological traversal is flattened into:

- `wavefronts`: all ready task IDs in wavefront order;
- `wavefront_ptr`: offsets delimiting individual wavefronts.

The generated launcher issues one CUDA kernel per wavefront and synchronizes
between launches. This gives a global ordering point before dependent tasks
run.

### Sync-free

Preprocessing flattens DAG out-edges into:

- `dep_count`: unfinished predecessor count for each task;
- `dag_ptr`: offsets into successor storage;
- `dag_index`: flattened successor IDs.

Each task busy-waits until its count reaches `sync_free_ready_count`, executes,
then atomically decrements all successor counts. Do not use CUDA simulation for
this busy-wait executor; simulator scheduling can deadlock independently of
the generated dependence relation.

## Current demos

The user-facing examples live in `compiler/demo` rather than the regression
test package.

### SpTRSV

`compiler.demo.sptrsv` contains CSR and CSC lower-triangular solves. The CSR
variant reads previously solved rows; the CSC variant atomically accumulates
updates into future rows. Both return a newly allocated solution vector.

```python
from compiler.demo.sptrsv import simple_sptrsv_csc, simple_sptrsv_csr
```

### Gauss-Seidel

`compiler.demo.gauss_seidel` contains:

- `simple_gauss_seidel_csr`, with a separately supplied diagonal;
- `full_gauss_seidel_csr`, with diagonal entries stored in CSR.

The outer `sweep` loop remains in the generated host launcher. The inner `i`
loop is the DAG task loop, so every sweep launches all wavefronts in order.

### ILU0

`compiler.demo.spilu0` contains undecorated CSR and CSC algorithms plus
factories that compile them:

```python
from compiler.demo.spilu0 import compile_ilu0_csc, compile_ilu0_csr

ilu0_csr = compile_ilu0_csr(
    execution_mode="wavefront",
    threads_per_block=128,
)
factor_data = ilu0_csr(data, indices, indptr)
```

Both examples explicitly return `A_data`, allowing the runtime to copy the
factorized values back. The CSC DAG inspector is currently conservative for
conditions represented by local scalars such as `u_kj`; this may add ordering
edges but must not omit required ones.

To print the current SpTRSV CUDA lowering without executing it:

```bash
python -m compiler.demo.show_cuda_lowering
```

## Inspecting generated artifacts

Decorated wrappers retain compiler and runtime artifacts:

```python
print(function.dag_source)
print(function.cuda_kernel_source)
print(function.cuda_launcher_source)

print(function.dependence_analysis.ordering_dependences)
print(function.dependence_analysis.reduction_dependences)

dag = function.build_dag(*arguments)
executor_data = function.build_execution_data(*arguments)
prepared = function.prepare_cuda_arguments(*arguments)
```

These helpers are intended for debugging. Normal callers should invoke the
decorated function directly.

## GPU end-to-end tests

The numerical suite covers:

- CSR and CSC SpTRSV against SciPy's triangular solver;
- simple and full CSR Gauss-Seidel against sequential references;
- CSR and CSC ILU0 against sequential implementations.

By default, these tests load
`data/ss/bcsstk17/bcsstk17.mtx` and use its leading 128-by-128 principal
sample. The real off-diagonal sparsity pattern and normalized values are
retained; the diagonal is strengthened to avoid zero-pivot or conditioning
failures unrelated to code generation. Select another Matrix Market input or
sample size with:

```bash
export GPU_BACKEND_E2E_MATRIX=data/ss/cant/cant.mtx
export GPU_BACKEND_E2E_MAX_N=256
```

Set `GPU_BACKEND_E2E_MAX_N=0` to use the complete matrix. Be aware that the
current conservative CSC-ILU0 inspector can generate a quadratic number of
candidate edges, so full large matrices are not yet appropriate for that test.

Run all physical-GPU tests from the repository root:

```bash
NUMBA_CUDA_USE_NVIDIA_BINDING=1 \
/home/salehm32/.virtualenvs/gpu-backend/bin/python -m unittest \
  compiler.demo.test_sptrsv_e2e \
  compiler.demo.test_gauss_seidel_e2e \
  compiler.demo.test_spilu0_e2e -v
```

These tests skip if a physical CUDA context cannot be created, and they also
skip when `NUMBA_ENABLE_CUDASIM=1`. Each test exercises decorator compilation,
DAG inspection, executor preprocessing, host-to-device transfer, kernel
launch, synchronization, result copy-back, and numerical comparison.

Run the complete structural and numerical regression suite with:

```bash
NUMBA_CUDA_USE_NVIDIA_BINDING=1 \
/home/salehm32/.virtualenvs/gpu-backend/bin/python -m unittest discover \
  -s compiler -t . -p "test_*.py" -v
```

## Current restrictions

- Source must be available through `inspect.getsource`.
- The mapped iterator must identify exactly one supported `range` loop.
- Mapped outer ranges currently need to be zero-based.
- Enclosing host loops must have a structurally supported body.
- Dependence extraction is primarily RAW-focused; WAR/WAW analysis remains
  incomplete.
- CUDA atomics are limited by Numba's supported operators and dtypes.
- No persistent JIT/DAG cache is implemented yet.
- Sync-free forward-progress guarantees require more work.

## Package layout

| Path | Responsibility |
| --- | --- |
| `demo/` | Decorated examples, physical-GPU end-to-end tests, SuiteSparse input helpers, and lowering display tools. |
| `transform_tests/` | Structural and code-generation regression tests. |
| `decorator.py` | Connects dependence analysis, DAG generation, lowering, and runtime execution. |
| `lcd/` | Frontend loop-carried-dependence IR and Python AST raising. |
| `spf_def.py`, `spf_raising.py` | Backend sparse polyhedral representation and LCD conversion. |
| `legality_check.py` | Dependence extraction, SMT checks, and reduction classification. |
| `SMT_Generator.py` | SMT constraints and symbolic function properties. |
| `dag_codegen.py` | DAG endpoint solving, ownership simplification, and inspector generation. |
| `dag_preprocessing.py` | Successor/count and wavefront executor arrays. |
| `cuda_lowering.py` | Numba-CUDA kernel and host-launcher generation. |
| `cuda_runtime.py` | Module binding, transfers, execution, and returned-value restoration. |

