# Gauss-Seidel Iterative Solver - Comprehensive GPU Implementation

## Directory Structure

```
gauss_seidel/
├── gauss_seidel_benchmark.py       # Main benchmark script
└── kernels/                        # Kernel implementations
    ├── gauss_seidel.py             # Baselines (CPU + GPU + AMGX)
    ├── gauss_seidel_diag.py        # Diagonal-optimized baselines
    ├── gauss_seidel_hlevel.py      # H-Level scheduling with CUDA graphs
    ├── gauss_seidel_hlevel_diag.py # H-Level with diagonal optimization
    ├── gauss_seidel_unrolled.py    # Mapping & unrolled variants (u1, u4-u64)
    ├── gauss_seidel_unrolled_diag.py # Mapping & unrolled with diagonal optimization
    └── gauss_seidel_utils.py       # Helper utilities (diagonal extraction)
```

## Kernel Categories

### `gauss_seidel.py` - Baselines

#### CPU Baselines:
- **`gauss_seidel_csr`**: Standard sequential with diagonal search
- **`gauss_seidel_diag_csr`**: Optimized sequential with pre-extracted diagonal

#### GPU Baselines:
- **`gs_numba_wavefront`**: Wavefront scheduling (1 thread per row)
- **`gs_numba_syncfree`**: Sync-free with warp-level parallelism (1 warp per row)

#### Library Baseline (optional):
- **`AMGXGaussSeidelSolver`**: NVIDIA AMGX library implementation
  - Requires: `pip install pyamgx`
  - Industrial-strength GPU solver
  - Fair benchmarking (no data transfer overhead)

### `gauss_seidel_diag.py` - Diagonal-Optimized Baselines

GPU implementations with pre-extracted diagonal values and positions:
- **`gs_numba_wavefront_diag`**: Wavefront with diagonal optimization
- **`gs_numba_syncfree_diag`**: Sync-free with diagonal optimization

**Key Optimization**: Eliminates `if col == row` branch by pre-extracting diagonal.

### `gauss_seidel_unrolled.py` - Mapping & Unrolling Variants

#### Wavefront Variants:
- **Mapping (u1)**: Multiple threads per row (factors: 32, 64, 128, 256 threads)
- **Unrolling (u4-u64)**: Explicit loop unrolling (factors: 4, 8, 16, 32, 64)

#### Sync-Free Variants:
- **Baseline**: Fixed 128 threads per block (warp-level)
- **Unrolling (u4-u64)**: Warp-level with explicit unrolling (factors: 4, 8, 16, 32, 64) + configurable threads (32, 64, 128, 256)

**Strategy**: Reduce loop overhead and maximize instruction-level parallelism (ILP).

### `gauss_seidel_unrolled_diag.py` - Diagonal-Optimized Unrolling

Same as `gauss_seidel_unrolled.py` but with diagonal optimization applied to all variants.

### `gauss_seidel_hlevel.py` & `gauss_seidel_hlevel_diag.py` - H-Level

Hierarchical level-set scheduling with CUDA graphs:
- **`gs_h_level_analyze[_diag]`**: Pre-analysis phase (Union-Find chain decomposition)
- **`gs_numba_hlevel_solve[_diag]`**: Graph-based solver with sub-warp optimization

**Features**:
- Union-Find algorithm for chain detection (lower triangle only)
- CUDA graph capture for reduced launch overhead
- Sub-warp adaptive execution
- Band-width parameter: 2, 4, 8, 16

### `gauss_seidel_utils.py` - Utilities

Helper functions for diagonal extraction:
- **`extract_diagonal_info(A_csr)`**: Extract diagonal positions and values
- **`extract_diagonal_info_gpu(A_csr)`**: Extract and transfer to GPU

## Naming Convention

Benchmark methods follow the pattern: `{category}-{method}-{variant}-{params}`

### Examples

**Baselines**:
- `cpu-sequential` - CPU baseline
- `cpu-sequential-diag` - CPU diagonal-optimized
- `amgx-gauss-seidel` - NVIDIA AMGX library
- `gs-wf-base` - GPU wavefront baseline
- `gs-sf-base` - GPU sync-free baseline (128 threads)
- `gs-wf-base-diag` - GPU wavefront diagonal-optimized
- `gs-sf-base-diag` - GPU sync-free diagonal-optimized (128 threads)

**Wavefront Unrolled**:
- `gs-wf-u1-t128` - Wavefront mapping, 128 threads
- `gs-wf-u4-t256` - Wavefront unroll factor 4, 256 threads
- `gs-wf-u16-t128` - Wavefront unroll factor 16, 128 threads
- `gs-wf-u64-t256-diag` - Wavefront unroll 64, 256 threads, diagonal-optimized

**Sync-Free Unrolled**:
- `gs-sf-u4-t128` - Sync-free unroll factor 4, 128 threads
- `gs-sf-u16-t256` - Sync-free unroll factor 16, 256 threads
- `gs-sf-u64-t128-diag` - Sync-free unroll 64, 128 threads, diagonal-optimized

**H-Level**:
- `gs-h-bw8` - H-level bandwidth 8
- `gs-h-bw16-diag` - H-level bandwidth 16, diagonal-optimized


---

# GPU Sparse Solver Benchmarking: Environment Setup

This guide details the installation of NVIDIA AMGX (via `pyamgx`) and PETSc (via `petsc4py`) with full CUDA support. These are required to run the library baselines for the Gauss-Seidel benchmarks.

## Prerequisites

* **CUDA Toolkit** (nvcc must be in PATH)
* **MPI** (OpenMPI or MPICH)
* **C++ Compiler** (gcc/g++)
* **CMake**
* **Python 3.x** with `pip` and `numpy` installed.

---

## 1. Installing PETSc with CUDA (petsc4py)

The standard `pip install petsc4py` installs a CPU-only binary. You must build PETSc from source to enable the `aijcusparse` matrix type.

### Step 1: Download and Configure
Clone the repository and configure it to use NVIDIA compilers.

```bash
git clone -b release [https://gitlab.com/petsc/petsc.git](https://gitlab.com/petsc/petsc.git)
cd petsc

# Configure with CUDA enabled
./configure \
  --with-cuda=1 \
  --with-cudac=nvcc \
  --with-debugging=0 \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=mpifort \
  --download-f2cblaslapack \
  COPTFLAGS="-O3" \
  CXXOPTFLAGS="-O3" \
  CUDAOPTFLAGS="-O3"
````

### Step 2: Compile

Run the build command suggested at the end of the configuration step.

```bash
make all check
```

### Step 3: Link Python Bindings

**Critical:** You must export environment variables so `pip` finds your local GPU build instead of downloading a CPU version from PyPI.

```bash
# Set these based on the output of 'make all check'
export PETSC_DIR=$(pwd)
export PETSC_ARCH=arch-linux-c-opt  # Check your actual arch name

# Force pip to compile bindings against your local source
pip uninstall -y petsc petsc4py
pip install petsc4py --no-binary :all:
```

### Step 4: Runtime Configuration

If you are running on a single node with standard OpenMPI, you must disable GPU-aware MPI checks to prevent `MPI_ABORT error 76`.

```bash
export PETSC_OPTIONS="-use_gpu_aware_mpi 0"
```

-----

## 2\. Installing AMGX (pyamgx)

If you encounter "Invalid/null C wrapper" errors, it usually indicates a binary mismatch between the PyPI wheel and your system drivers. It is safer to build AMGX from source.

### Step 1: Build AMGX Core

```bash
git clone [https://github.com/NVIDIA/AMGX.git](https://github.com/NVIDIA/AMGX.git)
cd AMGX
mkdir build && cd build
cmake ..
make -j16 all
```

### Step 2: Install Python Bindings

Tell `pyamgx` where to find the C++ library you just built.

```bash
export AMGX_DIR=/path/to/AMGX_root_folder
export AMGX_BUILD_DIR=/path/to/AMGX_root_folder/build

pip install pyamgx
```

-----

## 3\. Installing CuPy (Standard Baseline)

CuPy provides a direct interface to `cuSPARSE` and is easier to install than PETSc/AMGX.

```bash
pip install cupy-cuda12x  # Adjust 12x to match your CUDA version (e.g., 11x or 12x)
```

-----

## 4\. Verification

Use these snippets to ensure the libraries are correctly using the GPU.

**Verify PETSc (CUDA):**

```python
from petsc4py import PETSc
try:
    A = PETSc.Mat().create()
    A.setType('aijcusparse') 
    print("Success: PETSc has CUDA support!")
except Exception as e:
    print(f"Failure: {e}")
```

**Verify AMGX:**

```python
import pyamgx
pyamgx.initialize()
cfg = pyamgx.Config()
cfg.create("exception_handling=1")
rsrc = pyamgx.Resources(cfg)
m = pyamgx.Matrix()
try:
    m.create(rsrc, mode='dDDI') # Double Precision, Device, Device, Int32
    print("Success: AMGX handle created.")
except Exception as e:
    print(f"Failure: {e}")
m.destroy()
rsrc.destroy()
cfg.destroy()
pyamgx.finalize()
```
