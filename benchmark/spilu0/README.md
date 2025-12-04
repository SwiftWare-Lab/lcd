# SpILU0 (Sparse Incomplete LU Factorization) Kernels and Benchmarks

## Directory Structure

```
spilu0_factorization/
├── spilu0_benchmark.py          # Main benchmark script
└── kernels/                     # Kernel implementations
    ├── spilu0_baseline.py       # Baseline kernels (cuSPARSE, Numba CSR)
    ├── spilu0_unrolled.py       # Unrolled & mapping variants (u1, u4-u64)
    ├── spilu0_hlevel.py         # H-Level scheduling with CUDA graphs
    └── spilu0_legacy.py         # Legacy/unused implementations

```

## Kernel Categories

### `spilu0_baseline.py`
- **Baselines**: cuSPARSE, CSR wavefront, CSR sync-free
- **Helper functions**: `compute_ilu_dependency_counts_csr()`, `build_successors_from_csr()`, `find_diagonal_positions()`, `compute_ilu_wavefront_csr()`, `prepare_for_ilu0()`

### `spilu0_unrolled.py`
- **Mapping (u1)**: Thread-parallel variants with unroll factor 1
- **Unrolled (u4-u64)**: Code-generated kernels with explicit unrolling (factors: 4, 8, 16, 32, 64)
- **Formats**: CSR only for both wavefront and sync-free
- **Binary search**: Device function for efficient column matching

### `spilu0_hlevel.py`
- **H-Level scheduling**: Hierarchical level-set method with warp optimization
- **CUDA graphs**: Graph-based execution for reduced overhead
- **Analysis**: `ilu0_h_level_analyze()` for pre-processing
- **Union-Find**: Chain decomposition for independent task identification

### `spilu0_legacy.py`
- Unused/experimental implementations (Triton kernels, older Numba variants)
- Kept for reference only

## Naming Convention

Benchmark methods follow the pattern: `{format}-{method}-{variant}-{params}`
- **Format**: `csr` (CSR only for ILU0)
- **Method**: `wf` (wavefront), `sf` (sync-free), `h` (h-level)
- **Variant**: `base`, `u{factor}`
- **Params**: `t{threads}`, `bw{bandwidth}`

Examples:
- `csr-wf-base` - CSR wavefront baseline
- `csr-sf-u4-t128` - CSR sync-free with unroll factor 4, 128 threads
- `csr-wf-u16-t256` - CSR wavefront with unroll factor 16, 256 threads
- `csr-h-bw16` - CSR H-level with bandwidth 16
