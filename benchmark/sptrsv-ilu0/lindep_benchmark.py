import time
import numpy as np
import pandas as pd
from numba import cuda, float64, int32
import matplotlib.pyplot as plt
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Filter warnings
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

from sptrsv_kernels.sptrsv_baseline import *
from sptrsv_kernels.sptrsv_unrolled import (
    sptrsv_numba_csc_sync_free_mapping,
    sptrsv_numba_csr_sync_free_mapping,
)
from sptrsv_kernels.sptrsv_legacy import (
    sptrsv_numba_csc_sync_free_unrolling,
    sptrsv_numba_csr_sync_free_unrolling,
)

# --- BASELINE 1: CSR Single Block (Gather - Generic) ---
@cuda.jit
def sptrsv_single_block_csr_kernel(vals, cols, ptrs, b, x, N):
    tid = cuda.threadIdx.x
    sh_sum = cuda.shared.array(1024, dtype=float64) 

    for i in range(N):
        row_start = ptrs[i]
        row_end   = ptrs[i+1]
        
        partial_sum = 0.0
        diag_val = 1.0
        
        k = row_start + tid
        while k < row_end:
            col_idx = cols[k]
            val = vals[k]
            if col_idx == i:
                diag_val = val
            else:
                partial_sum += val * x[col_idx]
            k += cuda.blockDim.x
            
        sh_sum[tid] = partial_sum
        cuda.syncthreads()
        
        s = cuda.blockDim.x // 2
        while s > 0:
            if tid < s:
                sh_sum[tid] += sh_sum[tid + s]
            cuda.syncthreads()
            s //= 2
            
        if tid == 0:
            x[i] = (b[i] - sh_sum[0]) / diag_val
        cuda.syncthreads()

# --- BASELINE 1b: CSR Single Block (Warp-Level Optimized) ---
@cuda.jit
def sptrsv_single_block_csr_warp_kernel(vals, cols, ptrs, b, x, N):
    """
    Lightweight CSR using Warp-Level Primitives.
    REQUIRES: threads_per_block=32 (1 Warp).
    """
    tid = cuda.threadIdx.x
    lane = tid % 32
    mask = 0xFFFFFFFF # Active mask for the whole warp

    # Iterate rows strictly sequentially
    for i in range(N):
        row_start = ptrs[i]
        row_end = ptrs[i+1]
        
        # 1. Thread-Local Load & Math
        # Each thread grabs different non-zeros of the row
        partial_sum = 0.0
        diag_val = 0.0 
        
        k = row_start + lane
        while k < row_end:
            col_idx = cols[k]
            val = vals[k]
            if col_idx == i:
                diag_val = val # Found diagonal
            else:
                partial_sum += val * x[col_idx] # Gather x[k]
            k += 32 # Stride if row > 32 (unlikely for you)
            
        # 2. Warp-Level Reduction (Registers Only)
        # We simultaneously reduce the partial sums AND find the diagonal
        # (Since only 1 thread holds the diag, summing it brings it to Lane 0)
        
        # Offset 16 down to 1
        partial_sum += cuda.shfl_down_sync(mask, partial_sum, 16)
        diag_val    += cuda.shfl_down_sync(mask, diag_val, 16)
        partial_sum += cuda.shfl_down_sync(mask, partial_sum, 8)
        diag_val    += cuda.shfl_down_sync(mask, diag_val, 8)
        partial_sum += cuda.shfl_down_sync(mask, partial_sum, 4)
        diag_val    += cuda.shfl_down_sync(mask, diag_val, 4)
        partial_sum += cuda.shfl_down_sync(mask, partial_sum, 2)
        diag_val    += cuda.shfl_down_sync(mask, diag_val, 2)
        partial_sum += cuda.shfl_down_sync(mask, partial_sum, 1)
        diag_val    += cuda.shfl_down_sync(mask, diag_val, 1)

        # 3. Finalize
        if lane == 0:
            x[i] = (b[i] - partial_sum) / diag_val
            
        # 4. Memory Fence / Barrier
        # We still need this so the next row (i+1) sees the updated x[i]
        # Since we are 1 warp, this is very cheap.
        cuda.syncthreads() 

# --- BASELINE 2: CSC Single Block (Scatter) ---
@cuda.jit
def sptrsv_single_block_csc_kernel(vals, rows, ptrs, b, x, N):
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x

    # Init x = b
    for i in range(tid, N, stride):
        x[i] = b[i]
    cuda.syncthreads()

    for j in range(N):
        col_start = ptrs[j]
        col_end = ptrs[j+1]
        
        # Diag is first element
        diag_val = vals[col_start]
        
        if tid == 0:
            x[j] = x[j] / diag_val
        cuda.syncthreads()
        
        val_j = x[j]

        # Scatter to off-diagonals
        k = col_start + 1 + tid
        while k < col_end:
            row_idx = rows[k]
            val = vals[k]
            cuda.atomic.add(x, row_idx, -val * val_j) 
            k += stride
            
        cuda.syncthreads()

# --- Data Gen ---
def generate_bidiagonal_csr(N):
    nnz = 2 * N - 1
    vals = np.zeros(nnz, dtype=np.float64)
    cols = np.zeros(nnz, dtype=np.int32)
    ptrs = np.zeros(N + 1, dtype=np.int32)
    
    ptr = 0
    for i in range(N):
        ptrs[i] = ptr
        if i > 0:
            cols[ptr] = i - 1; vals[ptr] = -0.5; ptr += 1
        cols[ptr] = i; vals[ptr] = 2.0; ptr += 1
    ptrs[N] = ptr
    b = np.ones(N, dtype=np.float64)
    return vals, cols, ptrs, b

def csr_to_csc_bidiagonal(N, csr_vals, csr_cols, csr_ptrs):
    nnz = 2 * N - 1
    csc_vals = np.zeros(nnz, dtype=np.float64)
    csc_rows = np.zeros(nnz, dtype=np.int32)
    csc_ptrs = np.zeros(N + 1, dtype=np.int32)
    
    ptr = 0
    for j in range(N):
        csc_ptrs[j] = ptr
        csc_rows[ptr] = j; csc_vals[ptr] = 2.0; ptr += 1
        if j < N - 1:
            csc_rows[ptr] = j + 1; csc_vals[ptr] = -0.5; ptr += 1
            
    csc_ptrs[N] = ptr
    return csc_vals, csc_rows, csc_ptrs

def prep_sync_free_meta(N, fmt="CSR"):
    if fmt == "CSR":
        dep = np.ones(N, dtype=np.int32); dep[0] = 0
    else:
        dep = np.full(N, 2, dtype=np.int32); dep[0] = 1

    succ_ind = np.arange(1, N, dtype=np.int32)
    succ_off = np.arange(N + 1, dtype=np.int32); succ_off[N] = N - 1
    
    return dep, succ_off, succ_ind

# --- Runner ---
def run_benchmark():
    Ns = [1024, 4096, 16384, 65536, 131072]
    results = []
    
    print(f"{'Variant':<40} | {'N':<8} | {'Time (ms)':<10} | {'Speedup':<8}")
    print("-" * 75)

    for N in Ns:
        # Data
        h_vals_r, h_cols, h_ptrs_r, h_b = generate_bidiagonal_csr(N)
        h_vals_c, h_rows, h_ptrs_c = csr_to_csc_bidiagonal(N, h_vals_r, h_cols, h_ptrs_r)
        
        d_vals_r = cuda.to_device(h_vals_r); d_cols = cuda.to_device(h_cols); d_ptrs_r = cuda.to_device(h_ptrs_r)
        d_vals_c = cuda.to_device(h_vals_c); d_rows = cuda.to_device(h_rows); d_ptrs_c = cuda.to_device(h_ptrs_c)
        d_b = cuda.to_device(h_b)
        d_x = cuda.device_array(N, dtype=np.float64)

        # 1. Baseline CSR
        sptrsv_single_block_csr_kernel[1, 256](d_vals_r, d_cols, d_ptrs_r, d_b, d_x, N) 
        cuda.synchronize()
        start = time.time()
        for _ in range(10): sptrsv_single_block_csr_kernel[1, 256](d_vals_r, d_cols, d_ptrs_r, d_b, d_x, N)
        cuda.synchronize()
        t_base_csr = (time.time() - start) / 10.0 * 1000
        results.append({"Variant": "Single Block CSR", "N": N, "Time": t_base_csr, "Speedup": 1.0})
        print(f"{'Single Block CSR':<40} | {N:<8} | {t_base_csr:<10.4f} | {1.0:<8.2f}")

        # 1b. Baseline CSR (Warp-Level)
        # Note: Must run with exactly 32 threads
        sptrsv_single_block_csr_warp_kernel[1, 32](d_vals_r, d_cols, d_ptrs_r, d_b, d_x, N)
        cuda.synchronize()
        start = time.time()
        for _ in range(10): sptrsv_single_block_csr_warp_kernel[1, 32](d_vals_r, d_cols, d_ptrs_r, d_b, d_x, N)
        cuda.synchronize()
        t_base_csr_warp = (time.time() - start) / 10.0 * 1000
        results.append({"Variant": "Single Block CSR (Warp)", "N": N, "Time": t_base_csr_warp, "Speedup": t_base_csr/t_base_csr_warp})
        print(f"{'Single Block CSR (Warp)':<40} | {N:<8} | {t_base_csr_warp:<10.4f} | {t_base_csr/t_base_csr_warp:<8.2f}")

        # 2. Baseline CSC
        sptrsv_single_block_csc_kernel[1, 256](d_vals_c, d_rows, d_ptrs_c, d_b, d_x, N) 
        cuda.synchronize()
        start = time.time()
        for _ in range(10): sptrsv_single_block_csc_kernel[1, 256](d_vals_c, d_rows, d_ptrs_c, d_b, d_x, N)
        cuda.synchronize()
        t_base_csc = (time.time() - start) / 10.0 * 1000
        results.append({"Variant": "Single Block CSC", "N": N, "Time": t_base_csc, "Speedup": t_base_csr/t_base_csc})
        print(f"{'Single Block CSC':<40} | {N:<8} | {t_base_csc:<10.4f} | {t_base_csr/t_base_csc:<8.2f}")

        # 3. Sync Free Variants
        variants = [
            ("SyncFree CSR Map", sptrsv_numba_csr_sync_free_mapping, "CSR"),
            ("SyncFree CSR Unroll", sptrsv_numba_csr_sync_free_unrolling, "CSR"),
            ("SyncFree CSC Map", sptrsv_numba_csc_sync_free_mapping, "CSC"),
            ("SyncFree CSC Unroll", sptrsv_numba_csc_sync_free_unrolling, "CSC"),
        ]

        for name, func, fmt in variants:
            try:
                if fmt == "CSR":
                    h_dep, h_s_off, h_s_ind = prep_sync_free_meta(N, "CSR")
                    args = (d_vals_r, d_cols, d_ptrs_r, d_b)
                else:
                    h_dep, h_s_off, h_s_ind = prep_sync_free_meta(N, "CSC")
                    args = (d_vals_c, d_rows, d_ptrs_c, d_b)
                
                func(*args, M=N, N=N, nnz=len(h_vals_r), in_edges=h_dep.copy(), succ_offsets=h_s_off, succ_indices=h_s_ind)
                cuda.synchronize()
                
                start = time.time()
                for _ in range(5):
                    func(*args, M=N, N=N, nnz=len(h_vals_r), in_edges=h_dep.copy(), succ_offsets=h_s_off, succ_indices=h_s_ind)
                cuda.synchronize()
                t_sf = (time.time() - start) / 5.0 * 1000
                
                results.append({"Variant": name, "N": N, "Time": t_sf, "Speedup": t_base_csr/t_sf})
                print(f"{name:<40} | {N:<8} | {t_sf:<10.4f} | {t_base_csr/t_sf:<8.4f}")
            except Exception as e:
                print(f"Skipped {name}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_benchmark()
    df.to_csv("sptrsv_full_benchmark.csv", index=False)
    
    print("\nGenerating clustered bar chart...")
    
    pivot_df = df.pivot(index='N', columns='Variant', values='Time')
    
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8, logy=True, rot=0)
    
    ax.set_title("") 
    ax.grid(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel("Matrix Dimension (N)", fontsize=12, labelpad=10)
    ax.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=12, labelpad=10)
    
    plt.legend(title="Test Candidate", title_fontsize=10, fontsize=9, 
               bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plot_filename = "sptrsv_benchmark_clustered_bar.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")
    plt.show()