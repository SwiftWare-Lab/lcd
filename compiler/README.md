# Loop-Carried Dependence Compiler

The `compiler` package turns a Python loop with loop-carried array
dependences into a dependency-aware Numba-CUDA execution.

Use the `@parallel` decorator to identify the outer loop that represents one
logical task. The compiler:

1. raises the Python AST into its sparse polyhedral representation;
2. extracts read-after-write dependences between loop iterations;
3. separates ordering dependences from associative reductions;
4. generates a task DAG from the ordering dependences;
5. lowers the selected loop to a Numba-CUDA kernel;
6. lowers reduction statements to CUDA atomics;
7. generates either a wavefront or sync-free launcher;
8. builds executor data, transfers arrays to the device, launches the kernel,
   and returns copied-back results.

Reduction dependences are excluded from the DAG because their ordering is
implemented by atomics. All remaining ordering dependences become DAG edges.

## Requirements

The compiler requires Python packages including:

- NumPy and SciPy;
- SymPy;
- Z3 (`z3-solver`);
- Numba with a working CUDA installation.

Installing `compiler/requirements.txt` alone may not install every compiler
runtime dependency. Verify CUDA before importing a decorated module:

```bash
python -c "from numba import cuda; print(cuda.is_available())"
```

The decorator compiles and binds its generated Numba-CUDA module when the
decorated function is defined. Therefore, importing a module containing
`@parallel` currently requires `numba.cuda` to import successfully.

## Basic usage

The following CSC triangular solve has a loop-carried dependence: iteration
`i` updates values consumed by later iterations.

```python
import numpy as np

from compiler.decorator import parallel


function_properties = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
    ("Ap", "Ai"): ["triangularity"],
}


@parallel(loop="i", function_props=function_properties, execution_mode="wavefront")
def sptrsv_csc(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        # CSC lower-triangular columns store their diagonal first.
        x[i] = (b[i] - x[i]) / Ax[Ap[i]]
        for j in range(Ap[i] + 1, Ap[i + 1]):
            x[Ai[j]] += Ax[j] * x[i]
    return x
```

Call the decorated function with ordinary host arrays:

```python
result = sptrsv_csc(
    matrix.indptr.astype(np.int32),
    matrix.indices.astype(np.int32),
    matrix.data,
    b,
    matrix.shape[0],
)
```

The wrapper builds the DAG from these values, prepares the CUDA arrays,
launches all required kernels, synchronizes, copies `x` back to the host, and
executes the original `return x` epilogue.

Keep the decorator on one line for now. The source extractor currently removes
single-line decorators before parsing the function.

## Decorator options

```python
@parallel(loop="i", function_props=properties, execution_mode="wavefront")
```

Additional keyword options such as `threads_per_block=128` and
`validate_dag_endpoints=False` can be added to the same decorator line.

| Option | Meaning |
| --- | --- |
| `loop` | Required name of the top-level task iterator. |
| `function_props` | Symbolic assumptions used during dependence analysis. |
| `execution_mode` | `"wavefront"` or `"sync_free"`; the default is `"sync_free"`. |
| `threads_per_block` | Wavefront CUDA block size; the default is 128. |
| `validate_dag_endpoints` | Emit runtime bounds checks for generated DAG endpoints. |
| `sync_free_ready_count` | Ready value used by sync-free busy waiting; the default is 0. |

Supported function properties currently include:

```python
{
    "Ap": ["monotonically_increasing"],
    ("Ap", "Ai"): ["triangularity"],
}
```

The tuple property describes the relationship between a compressed pointer
array and its index array.

## How dependences are handled

### Ordering dependences

An ordering dependence connects two different logical tasks. For example, if
iteration `i` writes `x[k]` and a later iteration reads `x[k]`, the compiler
generates an edge:

```text
producer iteration i  ->  consumer iteration k
```

The DAG stores out-edges as successor lists. The incoming-edge count of a node
is stored separately as `dep_count`.

### Reduction dependences

A dependence is classified as a reduction when:

- both accesses belong to the same statement schedule; and
- the statement uses a recognized associative update.

The corresponding statement is retained as reduction metadata for CUDA
lowering and omitted from DAG creation. For example:

```python
x[Ai[j]] += Ax[j] * x[i]
```

is lowered to:

```python
cuda.atomic.add(x, Ai[j], Ax[j] * x[i])
```

`+=` is the best-supported reduction path at present. Unsupported
operator/dtype combinations raise `UnsupportedAtomicOperation` during
lowering.

## Execution modes

### Wavefront

Wavefront mode performs a topological traversal of the DAG. Nodes with no
remaining incoming edges form one wavefront.

Preprocessing produces:

- `wavefronts`: every node stored in one flattened `int32` array;
- `wavefront_ptr`: offsets delimiting each wavefront.

The generated host launcher iterates over `wavefront_ptr`, launches one kernel
for each wavefront, and calls `cuda.synchronize()` between launches. Each CUDA
thread obtains its real task iterator from:

```python
i = wavefronts[wave_start + cuda.grid(1)]
```

Wavefront mode is currently the recommended mode for initial correctness
testing.

### Sync-free

Sync-free mode launches all task blocks once. Each task waits until its
incoming-dependency count reaches the configured ready value:

```python
while cuda.atomic.add(dep_count, i, 0) != 0:
    pass
```

After completing task `i`, it decrements the counts of all successors:

```python
for position in range(dag_ptr[i], dag_ptr[i + 1]):
    successor = dag_index[position]
    cuda.atomic.add(dep_count, successor, -1)
```

Preprocessing produces:

- `dep_count`: number of unfinished in-edges for every node;
- `dag_ptr`: offsets into each node's successor list;
- `dag_index`: flattened out-edges/successors.

The generated DAG excludes self-edges, so roots have `dep_count == 0`.
`sync_free_ready_count=1` is available only for externally supplied counts
that intentionally include a self-edge.

Sync-free busy waiting still needs stronger production safeguards for CUDA
block forward progress and global-memory publication ordering. It should be
tested on a real GPU rather than CUDA simulation.

## Function shape currently supported

The selected iterator must identify exactly one top-level `for` loop:

```python
def operation(arguments):
    # Host prelude: executed before device preparation.
    output = np.zeros(n)

    for i in range(n):
        # Lowered CUDA kernel body.
        ...

    # Host epilogue: executed after results are copied back.
    return output
```

Important restrictions:

- the decorated source must be available through `inspect.getsource`;
- the selected loop must currently use `range(stop)` or a supported
  `range(start, stop)` form;
- block-mapped lowering requires a zero-based outer range;
- the selected loop must be top-level in the function body;
- lifted locals used in the kernel must be defined in the host prelude;
- host/device handling currently supports scalars, NumPy arrays, sequences,
  and objects implementing `__cuda_array_interface__`;
- the CUDA atomic operation must support the target dtype.

Code before the selected loop is removed from the kernel and evaluated on the
host to create lifted values such as `x`. Code after the loop is retained as
the host epilogue and defines the decorated function's return behavior.

## Inspecting generated artifacts

The wrapper retains every major compiler artifact:

```python
print(sptrsv_csc.dag_source)
print(sptrsv_csc.cuda_kernel_source)
print(sptrsv_csc.cuda_launcher_source)
print(sptrsv_csc.cuda_source)

analysis = sptrsv_csc.dependence_analysis
print(analysis.ordering_dependences)
print(analysis.reduction_dependences)
```

Runtime helpers are also exposed:

```python
dag, scheduling_data = sptrsv_csc.build_dag(Ap, Ai, Ax, b, n)
execution_data = sptrsv_csc.build_execution_data(Ap, Ai, Ax, b, n)
prepared = sptrsv_csc.prepare_cuda_arguments(Ap, Ai, Ax, b, n)

sptrsv_csc.cuda_launcher(*prepared.launcher_arguments)
```

Normally, call the decorated function directly instead of invoking these
helpers. They are primarily useful for debugging each pipeline stage.

## Testing

Run the compiler and structural code-generation tests with:

```bash
python -m unittest discover -s compiler/transform_tests -p "test_*.py" -v
```

The numerical SpTRSV test compares generated CSR and CSC solvers with SciPy:

```bash
python -m unittest compiler.transform_tests.test_sptrsv_e2e -v
```

For wavefront execution, CUDA simulation can be enabled before Python imports
Numba:

```bash
NUMBA_ENABLE_CUDASIM=1 \
python -m unittest compiler.transform_tests.test_sptrsv_e2e -v
```

The end-to-end tests skip when Numba CUDA is unavailable. Do not use the CUDA
simulator for sync-free busy-wait execution.

## Package layout

| Module | Responsibility |
| --- | --- |
| `decorator.py` | Connects analysis, DAG generation, lowering, and runtime execution. |
| `spf_def.py`, `spf_raising.py` | Sparse polyhedral representation and Python AST raising. |
| `legality_check.py` | Dependence extraction, SMT checks, and reduction classification. |
| `SMT_Generator.py` | Generates SMT constraints and function-property assumptions. |
| `dag_codegen.py` | Converts ordering dependences into runtime DAG-building code. |
| `dag_preprocessing.py` | Compacts successors, counts, and wavefronts into executor arrays. |
| `cuda_lowering.py` | Generates the Numba-CUDA kernel and host launcher ASTs. |
| `cuda_runtime.py` | Binds generated modules, transfers arrays, launches, and restores returns. |
| `transform_tests/` | Code-generation and numerical regression tests. |

The older `fusion` decorator and scheduler modules are separate experimental
paths. The loop-carried CUDA pipeline described here is implemented by
`parallel`.
