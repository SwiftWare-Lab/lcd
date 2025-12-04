import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
import triton
import cupy as cp
import cholespy
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from sksparse.cholmod import cholesky

# -------------------------------------------------------------------------
# IMPORTS: BASELINES & KERNELS
# -------------------------------------------------------------------------
# CPU Baselines
from cholesky import (
    cholesky_left_looking_numba,
)

# GPU Row-Cholesky (Right-Looking/Up-Looking) - From cholesky_2.py
from cholesky_2 import (
    cholesky_numba_wavefront as row_wavefront,
    cholesky_numba_syncfree as row_syncfree,
    cholesky_numba_hlevel_solve as row_hlevel,
    cholesky_h_level_analyze as row_analyze,
)

# GPU Left-Looking - From cholesky_left_looking.py
from cholesky_3 import (
    cholesky_left_wavefront_solve,
    cholesky_left_syncfree_solve,
    cholesky_left_hlevel_solve,
    cholesky_left_h_level_analyze
)

from bench_utils import get_matrix_signature

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
mtx_list = [
"/home/mad29/projects/testdata/matrices/1138_bus.mtx",
# "/home/mad29/projects/testdata/matrices/2cubes_sphere.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk08.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk11.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk12.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk13.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk14.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk15.mtx",
]

band_widths = [64] # Focused on 64 for H-Level

method_list = [
    "cholmod",
    "cholespy",
    "numba-cpu",
    # "gpu-row-wavefront",
    # "gpu-left-wavefront",
    "gpu-row-syncfree",
    "gpu-left-syncfree",
    "gpu-row-hlevel",
    "gpu-left-hlevel",
]

# -------------------------------------------------------------------------
# GRAPH & DATA HELPERS
# -------------------------------------------------------------------------

def map_A_into_L_csc(A_coo, L_csc):
    """
    Maps values of A into L (CSC format).
    Used for Left-Looking kernels.
    """
    L_filled = np.zeros_like(L_csc.data)
    
    # Iterate A (lower triangle)
    for i, j, val in zip(A_coo.row, A_coo.col, A_coo.data):
        if i >= j:
            # Find row i in Column j of L
            start = L_csc.indptr[j]
            end = L_csc.indptr[j+1]
            rows = L_csc.indices[start:end]
            
            idx = np.searchsorted(rows, i)
            if idx < len(rows) and rows[idx] == i:
                L_filled[start + idx] = val
    return L_filled

def map_A_into_L_csr(A_coo, L_csr):
    """
    Maps values of A into L (CSR format).
    Used for Row-Cholesky kernels.
    """
    L_filled = np.zeros_like(L_csr.data)
    for i, j, val in zip(A_coo.row, A_coo.col, A_coo.data):
        if i >= j:
            # Find col j in Row i of L
            start = L_csr.indptr[i]
            end = L_csr.indptr[i+1]
            cols = L_csr.indices[start:end]
            idx = np.searchsorted(cols, j)
            if idx < len(cols) and cols[idx] == j:
                L_filled[start + idx] = val
    return L_filled

def compute_wavefront_left(L_csc, L_csr):
    """
    Computes wavefronts for Left-Looking Cholesky.
    Dependency: Column j depends on Column k if L[j, k] != 0.
    In-Degree of j = Number of non-zeros in Row j of L (excluding diagonal).
    Row j of L is exactly Row j of L_csr.
    """
    n = L_csc.shape[0]
    in_degree = np.zeros(n, dtype=np.int32)
    
    # Calculate In-Degree using CSR view (Rows)
    for i in range(n):
        start = L_csr.indptr[i]
        end = L_csr.indptr[i+1]
        for ptr in range(start, end):
            col = L_csr.indices[ptr]
            if col < i: # Strict lower triangle
                in_degree[i] += 1
                
    # Topological Sort
    wavefronts = []
    current_wave = [i for i in range(n) if in_degree[i] == 0]
    
    while current_wave:
        wavefronts.append(np.array(current_wave, dtype=np.int32))
        next_wave = []
        for node in current_wave:
            # Children of 'node' are columns 'j' where L[j, node] != 0.
            # This is Column 'node' in CSC.
            start = L_csc.indptr[node]
            end = L_csc.indptr[node+1]
            for ptr in range(start, end):
                child = L_csc.indices[ptr]
                if child > node:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_wave.append(child)
        current_wave = next_wave
        
    return wavefronts

def compute_in_edges_left(L_csr):
    """
    Compute dependency counts for Left-Looking SyncFree.
    Count = Non-zeros in Row i (parents).
    """
    n = L_csr.shape[0]
    in_edges = np.zeros(n, dtype=np.int32)
    for i in range(n):
        cnt = 0
        for ptr in range(L_csr.indptr[i], L_csr.indptr[i+1]):
            if L_csr.indices[ptr] < i: cnt += 1
        in_edges[i] = cnt
    return in_edges

def compact_wavefronts(wavefronts):
    start = [0]
    waves = []
    for w in wavefronts:
        waves.extend(w.tolist())
        start.append(len(waves))
    return np.array(start, dtype=np.int32), np.array(waves, dtype=np.int32)

# -------------------------------------------------------------------------
# CORRECTNESS TEST
# -------------------------------------------------------------------------
def correctness_test():
    for mtx in mtx_list:
        print(f"--- {mtx} ---")
        Ao = spio.mmread(mtx)
        if not sp.isspmatrix_csc(Ao): Ao = Ao.tocsc()
        if Ao.dtype != np.float64: Ao.data = Ao.data.astype(np.float64)
        
        # Reference
        try:
            factor = cholesky(Ao)
            L_ref = factor.L() # CSC
            P = factor.P()
        except:
            print("Skipping (Not SPD)")
            continue
            
        # Data Prep
        A_p = sp.tril(Ao[P, :][:, P]).tocoo()
        
        # 1. Row-Cholesky Data (CSR)
        L_csr = L_ref.tocsr(); L_csr.sort_indices()
        L_init_row = map_A_into_L_csr(A_p, L_csr)
        
        # 2. Left-Looking Data (CSC)
        L_csc = L_ref # Already sorted CSC usually
        L_init_left = map_A_into_L_csc(A_p, L_csc)
        
        # Common Graph Structures
        # Note: Row-Cholesky and Left-Looking have the SAME dependency graph topology,
        # just viewed differently (Row-based vs Col-based). 
        # Since L is symmetric structure, In-Degree of Row i == In-Degree of Col i.
        # We can reuse the same wavefronts logic.
        wavefronts = compute_wavefront_left(L_csc, L_csr)
        ws, w = compact_wavefronts(wavefronts)
        d_w, d_ws = cuda.to_device(w), cuda.to_device(ws)
        
        # --- TEST CHOLESPY ---
        try:
            # Cholespy returns a solver, we verify it by solving Ax=b
            solver = cholespy.CholeskySolverD(Ao.shape[0], Ao.indptr, Ao.indices, Ao.data, cholespy.MatrixType.CSC)
            x_ref = np.ones(Ao.shape[0])
            b = Ao @ x_ref
            
            b_gpu = cp.array(b, dtype=np.float64)
            x_gpu = cp.zeros(Ao.shape[0], dtype=np.float64)
            solver.solve(b_gpu, x_gpu)
            
            if np.allclose(x_ref, x_gpu.get(), atol=1e-3):
                print("Cholespy: PASS")
            else:
                print("Cholespy: FAIL")
        except Exception as e: print(f"Cholespy err: {e}")

        # --- TEST ROW VARIANTS ---
        try:
            d_data = cuda.to_device(L_init_row)
            d_idx = cuda.to_device(L_csr.indices.astype(np.int32))
            d_ptr = cuda.to_device(L_csr.indptr.astype(np.int32))
            
            # Wavefront
            row_wavefront(d_data, d_idx, d_ptr, d_ws, d_w)
            if np.allclose(L_csr.data, d_data.copy_to_host(), atol=1e-3):
                print("GPU-Row-Wavefront: PASS")
            else: print("GPU-Row-Wavefront: FAIL")
            
            # SyncFree (Row) - Uses CSC for signalling
            d_data.copy_to_device(cuda.to_device(L_init_row))
            d_csc_idx = cuda.to_device(L_csc.indices.astype(np.int32))
            d_csc_ptr = cuda.to_device(L_csc.indptr.astype(np.int32))
            in_edges = compute_in_edges_left(L_csr)
            row_syncfree(d_data, d_idx, d_ptr, d_csc_idx, d_csc_ptr, cuda.to_device(in_edges))
            cuda.synchronize()
            if np.allclose(L_csr.data, d_data.copy_to_host(), atol=1e-3):
                print("GPU-Row-SyncFree: PASS")
            else: print("GPU-Row-SyncFree: FAIL")
            
        except Exception as e: print(f"Row err: {e}")

        # --- TEST LEFT VARIANTS ---
        try:
            d_data = cuda.to_device(L_init_left)
            d_idx = cuda.to_device(L_csc.indices.astype(np.int32))
            d_ptr = cuda.to_device(L_csc.indptr.astype(np.int32))
            
            # Dependency structure for Left is L^T (CSR)
            d_lt_idx = cuda.to_device(L_csr.indices.astype(np.int32))
            d_lt_ptr = cuda.to_device(L_csr.indptr.astype(np.int32))
            
            # Wavefront
            cholesky_left_wavefront_solve(d_data, d_idx, d_ptr, d_lt_idx, d_lt_ptr, d_ws, d_w)
            if np.allclose(L_csc.data, d_data.copy_to_host(), atol=1e-3):
                print("GPU-Left-Wavefront: PASS")
            else: print("GPU-Left-Wavefront: FAIL")
            
            # SyncFree (Left)
            d_data.copy_to_device(cuda.to_device(L_init_left))
            in_edges = compute_in_edges_left(L_csr)
            cholesky_left_syncfree_solve(d_data, d_idx, d_ptr, d_lt_idx, d_lt_ptr, cuda.to_device(in_edges))
            cuda.synchronize()
            if np.allclose(L_csc.data, d_data.copy_to_host(), atol=1e-3):
                print("GPU-Left-SyncFree: PASS")
            else: print("GPU-Left-SyncFree: FAIL")
            
            # H-Level (Left)
            d_data.copy_to_device(cuda.to_device(L_init_left))
            analysis = cholesky_left_h_level_analyze(L_csr.indptr.astype(np.int32), L_csr.indices.astype(np.int32), w, ws, band_width=64)
            cholesky_left_hlevel_solve(d_data, d_idx, d_ptr, d_lt_idx, d_lt_ptr, analysis)
            cuda.synchronize()
            if np.allclose(L_csc.data, d_data.copy_to_host(), atol=1e-3):
                print("GPU-Left-HLevel: PASS")
            else: print("GPU-Left-HLevel: FAIL")
            
        except Exception as e: print(f"Left err: {e}")

# -------------------------------------------------------------------------
# BENCHMARK
# -------------------------------------------------------------------------
configs = [
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=mtx_list,
        line_arg="provider",
        line_vals=method_list,
        line_names=method_list,
        styles=[("blue", "-")]*len(method_list),
        ylabel="GFLOP/S",
        plot_name="chol_perf",
        args={}
    )
]

@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    Ao = spio.mmread(matrices).tocsc()
    if Ao.dtype != np.float64: Ao.data = Ao.data.astype(np.float64)
    
    factor = cholesky(Ao)
    L_ref = factor.L()
    P = factor.P()
    
    # Prep Data
    A_p = sp.tril(Ao[P, :][:, P]).tocoo()
    L_csr = L_ref.tocsr(); L_csr.sort_indices()
    L_csc = L_ref
    
    L_init_row = map_A_into_L_csr(A_p, L_csr)
    L_init_left = map_A_into_L_csc(A_p, L_csc)
    
    # Common pointers
    idx_csr = L_csr.indices.astype(np.int32); ptr_csr = L_csr.indptr.astype(np.int32)
    idx_csc = L_csc.indices.astype(np.int32); ptr_csc = L_csc.indptr.astype(np.int32)
    
    d_idx_csr, d_ptr_csr = cuda.to_device(idx_csr), cuda.to_device(ptr_csr)
    d_idx_csc, d_ptr_csc = cuda.to_device(idx_csc), cuda.to_device(ptr_csc)
    
    d_data_row = cuda.device_array_like(L_init_row)
    d_data_left = cuda.device_array_like(L_init_left)
    d_init_row = cuda.to_device(L_init_row)
    d_init_left = cuda.to_device(L_init_left)
    
    # Quantiles
    quantiles = [0.5, 0.2, 0.8]
    rep, warmup = 10, 5
    
    if provider == 'cholmod':
        func = lambda: cholesky(Ao, mode="simplicial")

    elif provider == 'cholespy':
        A_csc = Ao.tocsc()
        if A_csc.dtype != np.float64:
            A_csc.data = A_csc.data.astype(np.float64)

        b_gpu = cp.ones(A_csc.shape[0], dtype=np.float64)
        x_gpu = cp.zeros(A_csc.shape[0], dtype=np.float64)

        def func():
            solver = cholespy.CholeskySolverF(A_csc.shape[0], A_csc.indptr, A_csc.indices, A_csc.data, cholespy.MatrixType.CSC)
            # solver.solve(b_gpu, x_gpu)
            cp.cuda.Stream.null.synchronize()
    
    elif provider == 'numba-cpu':
        L_copy = L_ref.copy(); Lt = L_ref.T.tocsc()
        prunePtr, pruneSet = Lt.indptr, Lt.indices
        A_p_csc = sp.tril(Ao[P, :][:, P]).tocsc().sorted_indices()
        def func():
            L_copy.data[:] = 0
            cholesky_left_looking_numba(L_copy.shape[0], A_p_csc.indptr, A_p_csc.indices, A_p_csc.data,
                                        L_copy.indptr, L_copy.indices, L_copy.data, prunePtr, pruneSet)
    
    else:
        # GPU KERNEL PROVIDERS
        # Use Custom Loop to avoid measuring Memory Reset Time
        
        if provider == 'gpu-row-wavefront':
            wf = compute_wavefront_left(L_csc, L_csr)
            ws, w = compact_wavefronts(wf)
            d_w, d_ws = cuda.to_device(w), cuda.to_device(ws)
            def run_kernel():
                row_wavefront(d_data_row, d_idx_csr, d_ptr_csr, d_ws, d_w)
                
        elif provider == 'gpu-row-syncfree':
            in_edges = compute_in_edges_left(L_csr)
            d_edges = cuda.to_device(in_edges)
            d_work_edges = cuda.device_array_like(d_edges)
            def run_kernel():
                d_work_edges.copy_to_device(d_edges) # Reset signals
                row_syncfree(d_data_row, d_idx_csr, d_ptr_csr, d_idx_csc, d_ptr_csc, d_work_edges)

        elif provider == 'gpu-row-hlevel':
            wf = compute_wavefront_left(L_csc, L_csr)
            ws, w = compact_wavefronts(wf)
            analysis = row_analyze(ptr_csr, idx_csr, w, ws, band_width=64)
            def run_kernel():
                row_hlevel(d_data_row, d_idx_csr, d_ptr_csr, analysis)

        elif provider == 'gpu-left-wavefront':
            wf = compute_wavefront_left(L_csc, L_csr)
            ws, w = compact_wavefronts(wf)
            d_w, d_ws = cuda.to_device(w), cuda.to_device(ws)
            def run_kernel():
                cholesky_left_wavefront_solve(d_data_left, d_idx_csc, d_ptr_csc, d_idx_csr, d_ptr_csr, d_ws, d_w)

        elif provider == 'gpu-left-syncfree':
            in_edges = compute_in_edges_left(L_csr)
            d_edges = cuda.to_device(in_edges)
            d_work_edges = cuda.device_array_like(d_edges)
            def run_kernel():
                d_work_edges.copy_to_device(d_edges) # Reset signals
                cholesky_left_syncfree_solve(d_data_left, d_idx_csc, d_ptr_csc, d_idx_csr, d_ptr_csr, d_work_edges)

        elif provider == 'gpu-left-hlevel':
            wf = compute_wavefront_left(L_csc, L_csr)
            ws, w = compact_wavefronts(wf)
            analysis = cholesky_left_h_level_analyze(ptr_csr, idx_csr, w, ws, band_width=64)
            def run_kernel():
                cholesky_left_hlevel_solve(d_data_left, d_idx_csc, d_ptr_csc, d_idx_csr, d_ptr_csr, analysis)
        
        else:
            return float('nan'), float('nan'), float('nan')

        # Custom Bench Loop
        # Warmup
        for _ in range(warmup):
            if 'row' in provider: d_data_row.copy_to_device(d_init_row)
            else: d_data_left.copy_to_device(d_init_left)
            run_kernel()
            cuda.synchronize()
            
        # Measurement
        times = []
        start_event = cuda.event()
        end_event = cuda.event()
        
        for _ in range(rep):
            # RESET DATA (Excluded from timer)
            if 'row' in provider: d_data_row.copy_to_device(d_init_row)
            else: d_data_left.copy_to_device(d_init_left)
            cuda.synchronize() # Ensure copy done
            
            start_event.record()
            run_kernel()
            end_event.record()
            end_event.synchronize()
            times.append(cuda.event_elapsed_time(start_event, end_event)) # Returns milliseconds
            
        ms = np.median(times)
        min_ms = np.min(times)
        max_ms = np.max(times)
        
        perf = lambda t: (2 * L_ref.nnz) / (t * 1e-3) / 1e9
        print(f"{provider} : {ms:.3f} ms")
        return perf(ms), perf(min_ms), perf(max_ms)

    # Standard Triton Bench for CPU/External Libs
    ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles, warmup=warmup, rep=rep)
    perf = lambda t: (2 * L_ref.nnz) / (t * 1e-3) / 1e9
    print(f"{provider} : {ms:.3f} ms")
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    # correctness_test()
    benchmark.run(print_data=True, save_path=".", show_plots=False)