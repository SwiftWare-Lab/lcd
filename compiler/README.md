# Compiler implementation

The project overview, decorator usage, execution modes, demos, and test
instructions are documented in the [repository README](../README.md).
This file describes the internal compiler pipeline and is intended for
contributors working on analysis, scheduling, code generation, or runtime
integration.

## Pipeline boundary

`decorator.py` is the pipeline coordinator. At decoration time it:

1. obtains and normalizes the Python AST;
2. raises the operation through the LCD frontend and SPF backend IRs;
3. extracts ordering and reduction dependences;
4. constructs a reusable DAG plan;
5. lowers the mapped loops to a Numba-CUDA kernel and host launcher; and
6. compiles and binds the generated module.

At invocation time, the wrapper binds concrete arguments, runs the generated
DAG inspector, constructs executor data, transfers CUDA arguments, launches
the kernels, and restores explicitly returned values on the host.

The current execution mode, unrolling factor, and thread-block size are
resolved in `parallel()` immediately before `lower_to_numba_cuda()` is called.
A sparsity-aware scheduler should integrate at this boundary. Because concrete
sparsity is available only when the wrapper is invoked, runtime scheduling
requires lowering and compilation to be deferred or cached as specialized
variants.

## Intermediate representations

- `lcd/` implements the frontend loop-carried-dependence representation.
  Loops enclosing the block-mapped loop are removed from device analysis, and
  referenced enclosing iterators become host parameters.
- `spf_def.py` and `spf_raising.py` implement the backend sparse polyhedral
  representation used by dependence analysis.
- `legality_check.py` and `SMT_Generator.py` build and solve access constraints,
  classify associative reductions, and retain ordering dependences.

Reduction dependences are represented as atomic-update metadata and excluded
from DAG construction. Ordering dependences are passed to `dag_codegen.py`.

## DAG and executor generation

`dag_codegen.py` turns ordering dependences into edge rules, applies ownership
and condition simplifications, and emits the runtime inspector. The inspector
builds successor lists and incoming-dependency counts. In wavefront mode it
also performs a topological traversal.

`dag_preprocessing.py` converts inspector output into one of two executor
formats:

- sync-free: `dep_count`, `dag_ptr`, and `dag_index`;
- wavefront: flattened `wavefronts` and `wavefront_ptr` arrays.

## CUDA lowering and runtime

`cuda_lowering.py` removes the block-mapped loop, assigns its iterator from
the selected DAG task, optionally maps an inner loop to CUDA threads, applies
compile-time unrolling, lowers reductions to atomics, and emits both kernel and
launcher ASTs.

`cuda_runtime.py` compiles and binds the generated module, evaluates lifted
host statements, moves array arguments to the device, invokes the generated
launcher, synchronizes, and copies returned values back to the host.

## Module map

| Module | Responsibility |
| --- | --- |
| `decorator.py` | End-to-end pipeline coordination and decorated wrapper. |
| `ast_normalization.py` | Local-alias normalization before dependence analysis. |
| `lcd/` | Frontend loop-carried-dependence IR. |
| `spf_def.py`, `spf_raising.py` | Backend sparse polyhedral IR and raising. |
| `legality_check.py` | Dependence extraction and reduction classification. |
| `SMT_Generator.py` | SMT constraints and symbolic function properties. |
| `dag_codegen.py` | DAG planning, simplification, and inspector generation. |
| `dag_preprocessing.py` | Executor-array construction. |
| `cuda_lowering.py` | CUDA kernel and launcher generation. |
| `cuda_runtime.py` | Compilation, transfers, execution, and result restoration. |
| `demo/` | User-facing examples and physical-GPU end-to-end tests. |
| `transform_tests/` | Structural and code-generation regression tests. |

