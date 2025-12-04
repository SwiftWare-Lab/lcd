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

## Running Benchmarks

```bash
cd /home/mad29/projects/gpu-backend/benchmark/spilu0_factorization
python spilu0_benchmark.py
```

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

## Key Differences from SpTRSV

1. **CSR Only**: ILU0 factorization is typically performed on CSR format only (no CSC variants)
2. **No Coarsening**: Unlike SpTRSV, ILU0 benchmarks focus on baseline, mapping/unrolling, and H-level methods
3. **In-place Operation**: ILU0 modifies the matrix data in-place, computing L and U factors within the original sparsity pattern
4. **Correctness Testing**: Validates factorization by checking if L@U matches the original matrix at non-zero positions

## Performance Metrics

Benchmarks report:
- **GFLOP/s**: Gigaflops per second (computed as 2 * nnz / time)
- **Median, 20th, and 80th percentile** timings across multiple runs
- **Correctness validation**: Using relative error threshold (default 1e-3)

