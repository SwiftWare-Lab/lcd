# SpTRSV (Sparse Triangular Solve) Kernels and Benchmarks

## Directory Structure

```
sptrsv/
├── sptrsv_benchmark.py          # Main benchmark script (real-world matrices)
├── lindep_benchmark.py          # Synthetic banded matrices with linear dependencies
└── kernels/                     # Kernel implementations
    ├── sptrsv_baseline.py       # Baseline kernels (Triton, Numba CSC/CSR)
    ├── sptrsv_unrolled.py       # Unrolled & mapping variants (u1, u4-u64)
    ├── sptrsv_coarsened.py      # Coarsening kernels (thread, block, combined)
    ├── sptrsv_hlevel.py         # H-Level scheduling with CUDA graphs
    └── sptrsv_legacy.py         # Legacy/unused implementations

```

## Kernel Categories

### `sptrsv_baseline.py`
- **Baselines**: cuSPARSE, CSC/CSR wavefront, CSC/CSR sync-free
- **Triton kernels**: CSC/CSR wavefront and sync-free variants
- **Helper functions**: `incoming_edge()`, `build_csc_successors()`, `build_csr_successors()`

### `sptrsv_unrolled.py`
- **Mapping (u1)**: Thread-parallel variants with unroll factor 1
- **Unrolled (u4-u64)**: Code-generated kernels with explicit unrolling
- **Formats**: CSC/CSR for both wavefront and sync-free

### `sptrsv_coarsened.py`
- **Wavefront coarsening**: Block (wc) and thread (tc) coarsening
- **Sync-free coarsening**: Block (bc) and thread (tc) coarsening
- **Combined 3-level**: Wavefront + warp + thread coarsening
- **Helper**: `compute_coarsened_wavefronts()`, `compact_coarsened_wavefronts()`

### `sptrsv_hlevel.py`
- **H-Level scheduling**: Hierarchical level-set method with warp optimization
- **CUDA graphs**: Graph-based execution for reduced overhead
- **Analysis**: `sptrsv_h_level_analyze()` for pre-processing

### `sptrsv_legacy.py`
- Unused/experimental implementations (Triton kernels, older variants)
- Kept for reference only

## Running Benchmarks

### Main Benchmark (Real Matrices)
```bash
cd /home/mad29/projects/gpu-backend/benchmark/sptrsv
python sptrsv_benchmark.py
```

### Linear Dependency Benchmark (Synthetic Banded Matrices)
Tests single-block vs sync-free for highly banded synthetic matrices with linear dependence chains:
```bash
cd /home/mad29/projects/gpu-backend/benchmark/sptrsv
python lindep_benchmark.py
```

## Naming Convention

Benchmark methods follow the pattern: `{format}-{method}-{variant}-{params}`
- **Format**: `csc` or `csr`
- **Method**: `wf` (wavefront), `sf` (sync-free), `h` (h-level)
- **Variant**: `base`, `u{factor}`, `coarse-{type}`
- **Params**: `t{threads}`, `wc{factor}`, `bc{factor}`, `tc{factor}`, `bw{bandwidth}`

Examples:
- `csc-wf-base` - CSC wavefront baseline
- `csr-sf-u4-t128` - CSR sync-free with unroll factor 4, 128 threads
- `csc-wf-coarse-block-wc4` - CSC wavefront block coarsening, factor 4
- `csr-h-bw8` - CSR H-level with bandwidth 8

