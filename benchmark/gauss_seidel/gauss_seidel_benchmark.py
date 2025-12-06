import sys
import os
# Add parent directory to path for bench_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
import triton
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from kernels.gauss_seidel_baseline import (
    gauss_seidel_csr,
    gauss_seidel_diag_csr,
    gs_numba_wavefront,
    gs_numba_syncfree,
    PYAMGX_AVAILABLE,
    AMGXGaussSeidelSolver
)
from kernels.gauss_seidel_hlevel import (
    gs_numba_hlevel_solve,
    gs_h_level_analyze,
    gs_numba_hlevel_syncfree_banded_solve,
    gs_numba_hlevel_syncfree_persistent_solve,
)
from kernels.gauss_seidel_unrolled import (
    gs_numba_wavefront_mapping,
    gs_numba_wavefront_unrolling_4, gs_numba_wavefront_unrolling_8,
    gs_numba_wavefront_unrolling_16, gs_numba_wavefront_unrolling_32,
    gs_numba_wavefront_unrolling_64,
    gs_numba_syncfree_unrolling_4, gs_numba_syncfree_unrolling_8,
    gs_numba_syncfree_unrolling_16, gs_numba_syncfree_unrolling_32,
    gs_numba_syncfree_unrolling_64
)
from kernels.gauss_seidel_baseline_diag import (
    gs_numba_wavefront_diag,
    gs_numba_syncfree_diag
)
from kernels.gauss_seidel_unrolled_diag import (
    gs_numba_wavefront_mapping_diag,
    gs_numba_wavefront_unrolling_4_diag, gs_numba_wavefront_unrolling_8_diag,
    gs_numba_wavefront_unrolling_16_diag, gs_numba_wavefront_unrolling_32_diag,
    gs_numba_wavefront_unrolling_64_diag,
    gs_numba_syncfree_unrolling_4_diag, gs_numba_syncfree_unrolling_8_diag,
    gs_numba_syncfree_unrolling_16_diag, gs_numba_syncfree_unrolling_32_diag,
    gs_numba_syncfree_unrolling_64_diag
)
from kernels.gauss_seidel_hlevel_diag import (
    gs_numba_hlevel_solve_diag,
    gs_h_level_analyze_diag,
    gs_numba_hlevel_syncfree_banded_solve_diag,
    gs_numba_hlevel_syncfree_persistent_solve_diag,
)
from kernels.gauss_seidel_utils import extract_diagonal_info_gpu
from bench_utils import get_matrix_signature

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")

mtx_list = [
"/home/mad29/projects/testdata/matrices/1138_bus.mtx",
# "/home/mad29/projects/testdata/matrices/2cubes_sphere.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk08.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk11.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk12.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk13.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk14.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk15.mtx",
]

# ============================================================================
# Benchmark Configuration
# ============================================================================
unroll_factors = [4, 8, 16, 32, 64]
thread_counts = [32, 64, 128, 256]
band_widths = [1, 2, 4, 8, 16]
num_iterations = 10

# ============================================================================
# Method List Construction
# Comment out any section to disable that category of tests
# ============================================================================
method_list = []

# ----------------------------------------------------------------------------
# BASELINES
# ----------------------------------------------------------------------------
method_list.extend([
    "cpu-sequential",
    "cpu-sequential-diag",
])

# Add AMGX if available
# if PYAMGX_AVAILABLE:
    # method_list.append("amgx-gauss-seidel")

method_list.extend([
    "gs-wf-base",
    "gs-wf-base-diag",
    "gs-sf-base",
    "gs-sf-base-diag",
])

# # ----------------------------------------------------------------------------
# # TEST CANDIDATES - WAVEFRONT - UNROLLED (Regular)
# # ----------------------------------------------------------------------------
# # Mapping (u1)
# for threads in thread_counts:
#     method_list.append(f"gs-wf-u1-t{threads}")

# # Explicit unrolling (u4-u64)
# for factor in unroll_factors:
#     for threads in thread_counts:
#         method_list.append(f"gs-wf-u{factor}-t{threads}")

# # ----------------------------------------------------------------------------
# # TEST CANDIDATES - WAVEFRONT - UNROLLED (Diagonal-Optimized)
# # ----------------------------------------------------------------------------
# # Mapping (u1)
# for threads in thread_counts:
#     method_list.append(f"gs-wf-u1-t{threads}-diag")

# # Explicit unrolling (u4-u64)
# for factor in unroll_factors:
#     for threads in thread_counts:
#         method_list.append(f"gs-wf-u{factor}-t{threads}-diag")

# ----------------------------------------------------------------------------
# TEST CANDIDATES - SYNC-FREE - UNROLLED (Regular)
# ----------------------------------------------------------------------------
# Unrolling (u4-u64) with thread count variations
for factor in unroll_factors:
    for threads in thread_counts:
        method_list.append(f"gs-sf-u{factor}-t{threads}")

# ----------------------------------------------------------------------------
# TEST CANDIDATES - SYNC-FREE - UNROLLED (Diagonal-Optimized)
# ----------------------------------------------------------------------------
# Unrolling (u4-u64) with thread count variations
for factor in unroll_factors:
    for threads in thread_counts:
        method_list.append(f"gs-sf-u{factor}-t{threads}-diag")

# ----------------------------------------------------------------------------
# TEST CANDIDATES - H-LEVEL (Regular + Diagonal-Optimized)
# ----------------------------------------------------------------------------
for bw in band_widths:
    method_list.extend([
        f"gs-h-bw{bw}",
        f"gs-h-bw{bw}-diag",
        f"gs-h-sf-bw{bw}",
        f"gs-h-sf-bw{bw}-diag",
        f"gs-h-sf-persist-bw{bw}",
        f"gs-h-sf-persist-bw{bw}-diag",
    ])

def compute_wavefront_gs(A):
    """Compute wavefront ordering based on lower triangle dependencies."""
    n = A.shape[0]
    A_csr = A.tocsr()
    
    in_degree = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for ptr in range(A_csr.indptr[i], A_csr.indptr[i+1]):
            col = A_csr.indices[ptr]
            if col < i:  # Lower triangle dependency
                in_degree[i] += 1
    
    wavefronts = []
    processed = np.zeros(n, dtype=bool)
    
    while not np.all(processed):
        wave = []
        for i in range(n):
            if not processed[i] and in_degree[i] == 0:
                wave.append(i)
        
        if len(wave) == 0:
            break
            
        wavefronts.append(np.array(wave, dtype=np.int32))
        
        for node in wave:
            processed[node] = True
            # Update dependents (rows that depend on this row)
            for ptr in range(A_csr.indptr[node], A_csr.indptr[node+1]):
                col = A_csr.indices[ptr]
                if col > node and not processed[col]:
                    # Row 'col' depends on row 'node' (via lower triangle)
                    # Actually we need to check if A[col, node] != 0
                    pass
            
            # Better approach: check who depends on me
            A_csc = A.tocsc()
            for ptr in range(A_csc.indptr[node], A_csc.indptr[node+1]):
                row = A_csc.indices[ptr]
                if row > node and not processed[row]:
                    in_degree[row] -= 1
    
    return wavefronts

def compact_wavefronts(wavefronts):
    """Compact wavefronts into flat arrays."""
    wave_start = [0]
    waves = []
    for wave in wavefronts:
        waves.extend(wave.tolist())
        wave_start.append(len(waves))
    return np.array(wave_start, dtype=np.int32), np.array(waves, dtype=np.int32)

def compute_lower_triangle_dependencies(A):
    """Compute incoming edge counts based on lower triangle only."""
    n = A.shape[0]
    A_csr = A.tocsr()
    in_edges = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        for ptr in range(A_csr.indptr[i], A_csr.indptr[i+1]):
            col = A_csr.indices[ptr]
            if col < i:  # Lower triangle
                in_edges[i] += 1
    
    return in_edges

def build_lower_triangle_csc(A):
    """Build CSC structure for lower triangle only (for signaling)."""
    n = A.shape[0]
    A_csr = A.tocsr()
    
    # Extract lower triangle
    L_data = []
    L_row = []
    L_col = []
    
    for i in range(n):
        for ptr in range(A_csr.indptr[i], A_csr.indptr[i+1]):
            col = A_csr.indices[ptr]
            if col < i:  # Lower triangle only
                L_data.append(A_csr.data[ptr])
                L_row.append(i)
                L_col.append(col)
    
    from scipy.sparse import coo_matrix
    L = coo_matrix((L_data, (L_row, L_col)), shape=(n, n)).tocsc()
    
    return L.indices.astype(np.int32), L.indptr.astype(np.int32)

def extract_diagonal(A):
    """Extract diagonal elements from CSR matrix."""
    n = A.shape[0]
    A_csr = A.tocsr()
    diag = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        for ptr in range(A_csr.indptr[i], A_csr.indptr[i+1]):
            if A_csr.indices[ptr] == i:
                diag[i] = A_csr.data[ptr]
                break
    
    return diag

def correctness_test():
    """
    Test a representative subset of methods for correctness.
    Testing all 300+ methods would take too long - we test key representatives.
    """
    matrix_signatures = []
    method_correctness = {}
    
    # Define representative test methods
    test_methods = [
        'cpu-sequential',
        'cpu-sequential-diag',
        'gs-wf-base',
        'gs-wf-base-diag',
        'gs-sf-base',             # Sync-free baseline
        'gs-sf-base-diag',        # Sync-free baseline diag
        'gs-wf-u1-t128',          # Mapping
        'gs-wf-u4-t128',          # Unrolling
        'gs-wf-u16-t128',         # Mid unrolling
        'gs-wf-u64-t128',         # Max unrolling
        'gs-wf-u4-t128-diag',     # Diag variant
        'gs-sf-u4-t128',          # Sync-free unrolling
        'gs-sf-u16-t128',         # Sync-free mid
        'gs-sf-u64-t128',         # Sync-free max
        'gs-sf-u4-t128-diag',     # Sync-free diag
        'gs-h-bw8',               # H-level
        'gs-h-bw8-diag',          # H-level diag
        'gs-h-sf-bw8',            # Sync-free H-level
        'gs-h-sf-bw8-diag',       # Sync-free H-level diag
        'gs-h-sf-persist-bw8',    # Persistent Sync-free H-level
        'gs-h-sf-persist-bw8-diag',# Persistent Sync-free H-level diag
    ]
    
    if PYAMGX_AVAILABLE:
        test_methods.insert(2, 'amgx-gauss-seidel')
    
    for mtx_matrix_path in mtx_list:
        print(f"------------ matrix: {mtx_matrix_path}")
        matrix = spio.mmread(mtx_matrix_path)
        n = matrix.shape[0]
        
        A = matrix.tocsr()
        if A.data.dtype != np.float64:
            print(f"Matrix {mtx_matrix_path} is not double precision")
            A.data = A.data.astype(np.float64)
        
        # Ensure diagonal dominance for convergence
        A.setdiag(A.diagonal() + np.abs(A).sum(axis=1).A1)
        
        mat_name = mtx_matrix_path.split("/")[-1]
        matrix_signatures.append(get_matrix_signature(A.tocsc(), mat_name))
        
        b = np.random.rand(n)
        x_init = np.zeros(n)
        
        method_correctness[mat_name] = {}
        
        # Reference solution (CPU sequential)
        x_ref = x_init.copy()
        x_ref, iters = gauss_seidel_csr(A.data, A.indptr, A.indices, b, x_ref, 
                                        tol=1e-6, max_iter=num_iterations)
        print(f"cpu-sequential: correct (reference, converged in {iters} iterations)")
        method_correctness[mat_name]['cpu-sequential'] = 1
        
        # Test each representative method
        for method in test_methods:
            if method == 'cpu-sequential':
                continue  # Already tested as reference
                
            try:
                x_test = None
                
                if method == 'cpu-sequential-diag':
                    diag = extract_diagonal(A)
                    x_test = x_init.copy()
                    x_test, _ = gauss_seidel_diag_csr(A.data, A.indptr, A.indices, diag, 
                                                     b, x_test, tol=1e-6, max_iter=num_iterations)
                    
                elif method == 'amgx-gauss-seidel':
                    amgx_solver = AMGXGaussSeidelSolver(max_iters=num_iterations, tol=1e-6)
                    amgx_solver.setup(A, b, x_init.copy())
                    amgx_solver.solve()
                    x_test = amgx_solver.download_result()
                    amgx_solver.destroy()
                
                elif method.startswith('gs-wf'):
                    # Wavefront test
                    d_data = cuda.to_device(A.data)
                    d_indices = cuda.to_device(A.indices.astype(np.int32))
                    d_indptr = cuda.to_device(A.indptr.astype(np.int32))
                    d_b = cuda.to_device(b)
                    d_x = cuda.to_device(x_init.copy())
                    
                    wavefronts = compute_wavefront_gs(A)
                    wave_start, waves = compact_wavefronts(wavefronts)
                    d_waves = cuda.to_device(waves)
                    
                    if 'diag' in method:
                        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
                        if 'base' in method:
                            gs_numba_wavefront_diag(d_data, d_indices, d_indptr, d_diag_pos, 
                                                   d_diag_vals, d_b, d_x, wave_start, d_waves, 
                                                   max_iter=num_iterations)
                        elif 'u1' in method:
                            gs_numba_wavefront_mapping_diag(d_data, d_indices, d_indptr, d_diag_pos, 
                                                           d_diag_vals, d_b, d_x, wave_start, d_waves, 
                                                           max_iter=num_iterations, threads_per_block=128)
                        else:
                            uf = int(method.split('-u')[1].split('-')[0])
                            func_map = {4: gs_numba_wavefront_unrolling_4_diag, 
                                      16: gs_numba_wavefront_unrolling_16_diag,
                                      64: gs_numba_wavefront_unrolling_64_diag}
                            func_map[uf](d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                        d_b, d_x, wave_start, d_waves, max_iter=num_iterations, threads_per_block=128)
                    else:
                        if 'base' in method:
                            gs_numba_wavefront(d_data, d_indices, d_indptr, d_b, d_x, 
                                              wave_start, d_waves, max_iter=num_iterations)
                        elif 'u1' in method:
                            gs_numba_wavefront_mapping(d_data, d_indices, d_indptr, d_b, d_x, 
                                                      wave_start, d_waves, max_iter=num_iterations, threads_per_block=128)
                        else:
                            uf = int(method.split('-u')[1].split('-')[0])
                            func_map = {4: gs_numba_wavefront_unrolling_4, 
                                      16: gs_numba_wavefront_unrolling_16,
                                      64: gs_numba_wavefront_unrolling_64}
                            func_map[uf](d_data, d_indices, d_indptr, d_b, d_x, wave_start, d_waves, 
                                        max_iter=num_iterations, threads_per_block=128)
                    
                    x_test = d_x.copy_to_host()
                    
                elif method.startswith('gs-sf'):
                    # Sync-free test
                    d_data = cuda.to_device(A.data)
                    d_indices = cuda.to_device(A.indices.astype(np.int32))
                    d_indptr = cuda.to_device(A.indptr.astype(np.int32))
                    d_b = cuda.to_device(b)
                    d_x = cuda.to_device(x_init.copy())
                    
                    csc_indices_L, csc_indptr_L = build_lower_triangle_csc(A)
                    d_csc_indices_L = cuda.to_device(csc_indices_L)
                    d_csc_indptr_L = cuda.to_device(csc_indptr_L)
                    in_edges = compute_lower_triangle_dependencies(A)
                    d_in_edges = cuda.to_device(in_edges.copy())
                    
                    if 'diag' in method:
                        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
                        if 'base' in method:
                            gs_numba_syncfree_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                                  d_csc_indices_L, d_csc_indptr_L, d_in_edges, 
                                                  d_b, d_x, max_iter=num_iterations)
                        else:
                            uf = int(method.split('-u')[1].replace('-diag', ''))
                            func_map = {4: gs_numba_syncfree_unrolling_4_diag, 
                                      16: gs_numba_syncfree_unrolling_16_diag,
                                      64: gs_numba_syncfree_unrolling_64_diag}
                            func_map[uf](d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                        d_csc_indices_L, d_csc_indptr_L, d_in_edges, 
                                        d_b, d_x, max_iter=num_iterations)
                    else:
                        if 'base' in method:
                            gs_numba_syncfree(d_data, d_indices, d_indptr, d_csc_indices_L, 
                                            d_csc_indptr_L, d_in_edges, d_b, d_x, max_iter=num_iterations)
                        else:
                            uf = int(method.split('-u')[1].replace('-diag', ''))
                            func_map = {4: gs_numba_syncfree_unrolling_4, 
                                      16: gs_numba_syncfree_unrolling_16,
                                      64: gs_numba_syncfree_unrolling_64}
                            func_map[uf](d_data, d_indices, d_indptr, d_csc_indices_L, d_csc_indptr_L,
                                        d_in_edges, d_b, d_x, max_iter=num_iterations)
                    
                    x_test = d_x.copy_to_host()
                    
                elif method.startswith('gs-h'):
                    # H-level test
                    d_data = cuda.to_device(A.data)
                    d_indices = cuda.to_device(A.indices.astype(np.int32))
                    d_indptr = cuda.to_device(A.indptr.astype(np.int32))
                    d_b = cuda.to_device(b)
                    d_x = cuda.to_device(x_init.copy())
                    
                    wavefronts = compute_wavefront_gs(A)
                    wave_start, waves = compact_wavefronts(wavefronts)
                    d_waves = cuda.to_device(waves)
                    
                    if 'gs-h-sf-persist' in method:
                        bw = int(method.replace('gs-h-sf-persist-bw', '').replace('-diag', ''))
                        if 'diag' in method:
                            d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
                            analysis_data = gs_h_level_analyze_diag(d_indptr, d_indices, d_waves, 
                                                                    wave_start, band_width=bw)
                            gs_numba_hlevel_syncfree_persistent_solve_diag(
                                d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                d_b, d_x, analysis_data, max_iter=num_iterations, sub_warp_size=8)
                        else:
                            analysis_data = gs_h_level_analyze(d_indptr, d_indices, d_waves, 
                                                               wave_start, band_width=bw)
                            gs_numba_hlevel_syncfree_persistent_solve(
                                d_data, d_indices, d_indptr, d_b, d_x,
                                analysis_data, max_iter=num_iterations, sub_warp_size=8)
                    elif 'gs-h-sf' in method:
                        bw = int(method.replace('gs-h-sf-bw', '').replace('-diag', ''))
                        if 'diag' in method:
                            d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
                            analysis_data = gs_h_level_analyze_diag(d_indptr, d_indices, d_waves, 
                                                                    wave_start, band_width=bw)
                            gs_numba_hlevel_syncfree_banded_solve_diag(
                                d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                d_b, d_x, analysis_data, max_iter=num_iterations, sub_warp_size=8)
                        else:
                            analysis_data = gs_h_level_analyze(d_indptr, d_indices, d_waves, 
                                                               wave_start, band_width=bw)
                            gs_numba_hlevel_syncfree_banded_solve(
                                d_data, d_indices, d_indptr, d_b, d_x, 
                                analysis_data, max_iter=num_iterations, sub_warp_size=8)
                    else:
                        bw = int(method.replace('gs-h-bw', '').replace('-diag', ''))
                        if 'diag' in method:
                            d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
                            analysis_data = gs_h_level_analyze_diag(d_indptr, d_indices, d_waves, 
                                                                    wave_start, band_width=bw)
                            gs_numba_hlevel_solve_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                                       d_b, d_x, analysis_data, max_iter=num_iterations, sub_warp_size=8)
                        else:
                            analysis_data = gs_h_level_analyze(d_indptr, d_indices, d_waves, 
                                                              wave_start, band_width=bw)
                            gs_numba_hlevel_solve(d_data, d_indices, d_indptr, d_b, d_x, 
                                                analysis_data, max_iter=num_iterations, sub_warp_size=8)
                    
                    x_test = d_x.copy_to_host()
                
                else:
                    print(f"{method}: skipped (not in test categories)")
                    continue
                
                # Check correctness
                if x_test is not None:
                    if np.allclose(x_ref, x_test, atol=1e-4):
                        print(f"{method}: correct")
                        method_correctness[mat_name][method] = 1
                    else:
                        print(f"Warning: {method} incorrect")
                        method_correctness[mat_name][method] = 0
                    
            except Exception as e:
                print(f"{method} failed: {e}")
                method_correctness[mat_name][method] = 0
    
    return matrix_signatures, method_correctness

# mat_signatures, method_correctness = correctness_test()
# import pandas as pd
# df = pd.DataFrame(mat_signatures)
# df.to_csv("gs_matrix_signatures.csv")
# df = pd.DataFrame(method_correctness)
# df.to_csv("gs_method_correctness.csv")

# Generate line names by simplifying method names
line_names = [m.upper().replace("GS-", "").replace("-", " ") for m in method_list]

# Generate styles (simplified - use default colors and line styles)
base_colors = ["black", "gray", "green", "red", "blue", "purple", "orange", "teal", "pink", "brown"]
base_styles = ["-", "--", "-.", ":"]

styles = []
for i, method in enumerate(method_list):
    color = base_colors[i % len(base_colors)]
    style = base_styles[(i // len(base_colors)) % len(base_styles)]
    styles.append((color, style))

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],
        line_arg="provider",
        line_vals=method_list,
        line_names=line_names,
        styles=styles,
        ylabel="GFLOP/S",
        plot_name="gauss-seidel-performance",
        args={}
    ))

@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    matrix = spio.mmread(matrices)
    n = matrix.shape[0]
    
    A = matrix.tocsr()
    if A.data.dtype != np.float64:
        print(f"Matrix {matrices} is not double precision")
        A.data = A.data.astype(np.float64)
    
    # Ensure diagonal dominance for convergence
    A.setdiag(A.diagonal() + np.abs(A).sum(axis=1).A1)
    
    b = np.random.rand(n)
    x_init = np.zeros(n)
    
    quantiles = [0.5, 0.2, 0.8]
    rep, warmup = 10, 5
    
    # ========================================================================
    # CPU BASELINES
    # ========================================================================
    if provider == 'cpu-sequential':
        def run_cpu():
            x = x_init.copy()
            gauss_seidel_csr(A.data, A.indptr, A.indices, b, x, 
                           tol=1e-6, max_iter=num_iterations)
        ms, min_ms, max_ms = triton.testing.do_bench(run_cpu, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    elif provider == 'cpu-sequential-diag':
        diag = extract_diagonal(A)
        def run_cpu_diag():
            x = x_init.copy()
            gauss_seidel_diag_csr(A.data, A.indptr, A.indices, diag, b, x, 
                                 tol=1e-6, max_iter=num_iterations)
        ms, min_ms, max_ms = triton.testing.do_bench(run_cpu_diag, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # AMGX BASELINE
    # ========================================================================
    elif provider == 'amgx-gauss-seidel':
        if not PYAMGX_AVAILABLE:
            print(f"AMGX not available, skipping...")
            return float('nan'), float('nan'), float('nan')
        
        amgx_solver = AMGXGaussSeidelSolver(max_iters=num_iterations, tol=1e-6)
        amgx_solver.setup(A, b, x_init.copy())
        
        def run_amgx():
            amgx_solver.reset_x()
            amgx_solver.solve()
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_amgx, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
        amgx_solver.destroy()
    
    # ========================================================================
    # GPU BASELINES - REGULAR
    # ========================================================================
    elif provider == 'gs-wf-base':
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_wavefront(d_data, d_indices, d_indptr, d_b, d_x, 
                              wave_start, d_waves, max_iter=num_iterations)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    elif provider == 'gs-sf-base':
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        csc_indices_L, csc_indptr_L = build_lower_triangle_csc(A)
        d_csc_indices_L = cuda.to_device(csc_indices_L)
        d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        in_edges = compute_lower_triangle_dependencies(A)
        d_in_edges = cuda.to_device(in_edges.copy())
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_syncfree(d_data, d_indices, d_indptr, d_csc_indices_L, 
                             d_csc_indptr_L, d_in_edges, d_b, d_x, 
                             max_iter=num_iterations, threads_per_block=128)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # GPU BASELINES - DIAGONAL-OPTIMIZED
    # ========================================================================
    elif provider == 'gs-wf-base-diag':
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_wavefront_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                   d_b, d_x, wave_start, d_waves, max_iter=num_iterations)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    elif provider == 'gs-sf-base-diag':
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        csc_indices_L, csc_indptr_L = build_lower_triangle_csc(A)
        d_csc_indices_L = cuda.to_device(csc_indices_L)
        d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        in_edges = compute_lower_triangle_dependencies(A)
        d_in_edges = cuda.to_device(in_edges.copy())
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_syncfree_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                  d_csc_indices_L, d_csc_indptr_L, d_in_edges, 
                                  d_b, d_x, max_iter=num_iterations, threads_per_block=128)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # WAVEFRONT UNROLLED - REGULAR
    # ========================================================================
    elif provider.startswith('gs-wf-u') and '-t' in provider and not provider.endswith('-diag'):
        parts = provider.replace('gs-wf-u', '').replace('-diag', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        # Function mapping
        if unroll_factor == 1:
            func = gs_numba_wavefront_mapping
        else:
            func_map = {
                4: gs_numba_wavefront_unrolling_4,
                8: gs_numba_wavefront_unrolling_8,
                16: gs_numba_wavefront_unrolling_16,
                32: gs_numba_wavefront_unrolling_32,
                64: gs_numba_wavefront_unrolling_64,
            }
            func = func_map[unroll_factor]
        
        def run():
            d_x.copy_to_device(d_x_init)
            func(d_data, d_indices, d_indptr, d_b, d_x, wave_start, d_waves, 
                 max_iter=num_iterations, threads_per_block=threads_per_block)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # WAVEFRONT UNROLLED - DIAGONAL-OPTIMIZED
    # ========================================================================
    elif provider.startswith('gs-wf-u') and '-t' in provider and provider.endswith('-diag'):
        parts = provider.replace('gs-wf-u', '').replace('-diag', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        # Function mapping
        if unroll_factor == 1:
            func = gs_numba_wavefront_mapping_diag
        else:
            func_map = {
                4: gs_numba_wavefront_unrolling_4_diag,
                8: gs_numba_wavefront_unrolling_8_diag,
                16: gs_numba_wavefront_unrolling_16_diag,
                32: gs_numba_wavefront_unrolling_32_diag,
                64: gs_numba_wavefront_unrolling_64_diag,
            }
            func = func_map[unroll_factor]
        
        def run():
            d_x.copy_to_device(d_x_init)
            func(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                 d_b, d_x, wave_start, d_waves, 
                 max_iter=num_iterations, threads_per_block=threads_per_block)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # SYNC-FREE UNROLLED - REGULAR
    # ========================================================================
    elif provider.startswith('gs-sf-u') and not provider.endswith('-diag'):
        # Extract unroll_factor and threads: gs-sf-u{factor}-t{threads}
        parts = provider.split('-')
        unroll_factor = int(parts[2][1:])  # Extract from 'u{factor}'
        threads = int(parts[3][1:])  # Extract from 't{threads}'
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        csc_indices_L, csc_indptr_L = build_lower_triangle_csc(A)
        d_csc_indices_L = cuda.to_device(csc_indices_L)
        d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        in_edges = compute_lower_triangle_dependencies(A)
        d_in_edges = cuda.to_device(in_edges.copy())
        
        func_map = {
            4: gs_numba_syncfree_unrolling_4,
            8: gs_numba_syncfree_unrolling_8,
            16: gs_numba_syncfree_unrolling_16,
            32: gs_numba_syncfree_unrolling_32,
            64: gs_numba_syncfree_unrolling_64,
        }
        func = func_map[unroll_factor]
        
        def run():
            d_x.copy_to_device(d_x_init)
            func(d_data, d_indices, d_indptr, d_csc_indices_L, d_csc_indptr_L,
                 d_in_edges, d_b, d_x, max_iter=num_iterations, threads_per_block=threads)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # SYNC-FREE UNROLLED - DIAGONAL-OPTIMIZED
    # ========================================================================
    elif provider.startswith('gs-sf-u') and provider.endswith('-diag'):
        # Extract unroll_factor and threads: gs-sf-u{factor}-t{threads}-diag
        parts = provider.split('-')
        unroll_factor = int(parts[2][1:])  # Extract from 'u{factor}'
        threads = int(parts[3][1:])  # Extract from 't{threads}'
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        csc_indices_L, csc_indptr_L = build_lower_triangle_csc(A)
        d_csc_indices_L = cuda.to_device(csc_indices_L)
        d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        in_edges = compute_lower_triangle_dependencies(A)
        d_in_edges = cuda.to_device(in_edges.copy())
        
        func_map = {
            4: gs_numba_syncfree_unrolling_4_diag,
            8: gs_numba_syncfree_unrolling_8_diag,
            16: gs_numba_syncfree_unrolling_16_diag,
            32: gs_numba_syncfree_unrolling_32_diag,
            64: gs_numba_syncfree_unrolling_64_diag,
        }
        func = func_map[unroll_factor]
        
        def run():
            d_x.copy_to_device(d_x_init)
            func(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                 d_csc_indices_L, d_csc_indptr_L,
                 d_in_edges, d_b, d_x, max_iter=num_iterations, threads_per_block=threads)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # H-LEVEL - REGULAR
    # ========================================================================
    elif provider.startswith('gs-h-bw') and not provider.endswith('-diag'):
        band_width = int(provider.replace('gs-h-bw', '').replace('-diag', ''))
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        analysis_data = gs_h_level_analyze(d_indptr, d_indices, d_waves, 
                                           wave_start, band_width=band_width)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_hlevel_solve(d_data, d_indices, d_indptr, d_b, d_x, 
                                 analysis_data, max_iter=num_iterations, sub_warp_size=8)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)

    elif provider.startswith('gs-h-sf-bw') and not provider.endswith('-diag'):
        band_width = int(provider.replace('gs-h-sf-bw', '').replace('-diag', ''))
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        analysis_data = gs_h_level_analyze(d_indptr, d_indices, d_waves, 
                                           wave_start, band_width=band_width)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_hlevel_syncfree_banded_solve(d_data, d_indices, d_indptr, d_b, d_x, 
                                                 analysis_data, max_iter=num_iterations, sub_warp_size=8)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)

    elif provider.startswith('gs-h-sf-persist-bw') and not provider.endswith('-diag'):
        band_width = int(provider.replace('gs-h-sf-persist-bw', '').replace('-diag', ''))
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        analysis_data = gs_h_level_analyze(d_indptr, d_indices, d_waves, 
                                           wave_start, band_width=band_width)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_hlevel_syncfree_persistent_solve(d_data, d_indices, d_indptr, d_b, d_x, 
                                                     analysis_data, max_iter=num_iterations, sub_warp_size=8)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    # ========================================================================
    # H-LEVEL - DIAGONAL-OPTIMIZED
    # ========================================================================
    elif provider.startswith('gs-h-bw') and provider.endswith('-diag'):
        band_width = int(provider.replace('gs-h-bw', '').replace('-diag', ''))
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        analysis_data = gs_h_level_analyze_diag(d_indptr, d_indices, d_waves, 
                                                wave_start, band_width=band_width)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_hlevel_solve_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                      d_b, d_x, analysis_data, max_iter=num_iterations, sub_warp_size=8)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)

    elif provider.startswith('gs-h-sf-bw') and provider.endswith('-diag'):
        band_width = int(provider.replace('gs-h-sf-bw', '').replace('-diag', ''))
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        analysis_data = gs_h_level_analyze_diag(d_indptr, d_indices, d_waves, 
                                                wave_start, band_width=band_width)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_hlevel_syncfree_banded_solve_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                                      d_b, d_x, analysis_data, max_iter=num_iterations, sub_warp_size=8)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)

    elif provider.startswith('gs-h-sf-persist-bw') and provider.endswith('-diag'):
        band_width = int(provider.replace('gs-h-sf-persist-bw', '').replace('-diag', ''))
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices.astype(np.int32))
        d_indptr = cuda.to_device(A.indptr.astype(np.int32))
        d_diag_pos, d_diag_vals = extract_diagonal_info_gpu(A)
        d_b = cuda.to_device(b)
        d_x = cuda.device_array(n, dtype=np.float64)
        d_x_init = cuda.to_device(x_init.copy())
        
        wavefronts = compute_wavefront_gs(A)
        wave_start, waves = compact_wavefronts(wavefronts)
        d_waves = cuda.to_device(waves)
        
        analysis_data = gs_h_level_analyze_diag(d_indptr, d_indices, d_waves, 
                                                wave_start, band_width=band_width)
        
        def run():
            d_x.copy_to_device(d_x_init)
            gs_numba_hlevel_syncfree_persistent_solve_diag(d_data, d_indices, d_indptr, d_diag_pos, d_diag_vals,
                                                           d_b, d_x, analysis_data, max_iter=num_iterations, sub_warp_size=8)
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles, 
                                                      warmup=warmup, rep=rep)
    
    else:
        print(f"Unknown provider: {provider}, skipping...")
        return float('nan'), float('nan'), float('nan')
    
    # Performance calculation: 2 * nnz * num_iterations (approximate FLOPs for GS)
    perf = lambda t_ms: (2 * A.nnz * num_iterations) / (t_ms * 1e-3) / 1e9 if not np.isnan(t_ms) else float('nan')
    print(f"{provider} -- {matrices} -- time: {ms:.3f} ms")
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".", show_plots=False)
    
    # Finalize AMGX at the very end
    if PYAMGX_AVAILABLE:
        try:
            import pyamgx
            pyamgx.finalize()
        except:
            pass  # Already finalized or failed

