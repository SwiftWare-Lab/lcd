"""
Supernodal SpTRSV (Sparse Triangular Solve) implementations:
- Python CPU sequential
- Numba CPU sequential  
- Numba CPU wavefront (parallel)
- Numba GPU sync-free
- Numba GPU wavefront
"""

import numpy as np
import cupy as cp
from numba import njit, cuda, config, prange, void, int32, float64

from utils.dense_blas import (
    lsolve_dense_col_order,
    dense_matvec_col_order,
    lsolve_BLAS_python,
    matvec_BLAS_python
)

config.CUDA_ENABLE_PYNVJITLINK = 1


# =============================================================================
# Python CPU Sequential Implementation
# =============================================================================

def sptrv_csc_supernodal(sn_begin, Lip, Li, Lp, Lx, b, x):
    """
    Python implementation of supernodal SpTRSV (sequential).
    
    Args:
        sn_begin: Supernode column boundaries
        Lip: Supernode row index pointers
        Li: Row indices
        Lp: Column pointers
        Lx: Non-zero values
        b: Input vector
        x: Output vector (in-place)
    """
    for i in range(len(sn_begin) - 1):
        col_b = sn_begin[i]
        col_e = sn_begin[i + 1]
        sup_width = col_e - col_b
        sup_len = Lip[i + 1] - Lip[i]
        val_beg = Lp[col_b]
        
        # Solve diagonal block
        lsolve_dense_col_order(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        
        # Update off-diagonal contributions
        if sup_len > sup_width:
            off_diag_val = Lx[val_beg + sup_width:]
            sol = dense_matvec_col_order(sup_len, sup_len - sup_width, sup_width,
                                         off_diag_val, x[col_b:col_e])
            cnt = 0
            for j in range(Lip[i] + sup_width, Lip[i + 1]):
                x[Li[j]] -= sol[cnt]
                sol[cnt] = 0
                cnt += 1


# =============================================================================
# Numba CPU Sequential Implementation
# =============================================================================

@njit(void(int32[:], int32[:], int32[:], int32[:], float64[:],
           float64[:], float64[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_numba(sn_begin, Lip, Li, Lp, Lx, b, x):
    """
    Numba-accelerated CPU implementation of supernodal SpTRSV (sequential).
    
    Args:
        sn_begin: Supernode column boundaries
        Lip: Supernode row index pointers
        Li: Row indices
        Lp: Column pointers
        Lx: Non-zero values
        b: Input vector
        x: Output vector (in-place)
    """
    sol = np.zeros(len(b), dtype=b.dtype)
    for i in range(len(sn_begin) - 1):
        col_b = sn_begin[i]
        col_e = sn_begin[i + 1]
        sup_width = col_e - col_b
        sup_len = Lip[i + 1] - Lip[i]
        val_beg = Lp[col_b]
        
        # Solve diagonal block
        lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        
        # Matrix-vector product for off-diagonal
        off_diag_val = Lx[val_beg + sup_width:]
        matvec_BLAS_python(sup_len, sup_len - sup_width, sup_width,
                           off_diag_val, x[col_b:col_e], sol)
        
        # Scatter updates
        cnt = 0
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            x[Li[j]] -= sol[cnt]
            sol[cnt] = 0
            cnt += 1


# =============================================================================
# Numba CPU Wavefront Implementation (Parallel)
# =============================================================================

@njit(void(int32[:], int32[:], int32[:], int32[:], float64[:],
           float64[:], float64[:], int32[:], int32[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_wavefront_numba(sn_begin, Lip, Li, Lp, Lx, b, x, wave_ptr, waves):
    """
    Numba CPU wavefront implementation - processes supernodes in parallel within wavefronts.
    
    Args:
        sn_begin: Supernode column boundaries
        Lip: Supernode row index pointers
        Li: Row indices
        Lp: Column pointers
        Lx: Non-zero values
        b: Input vector
        x: Output vector (in-place)
        wave_ptr: Wavefront boundaries
        waves: Supernode IDs grouped by wavefront
    """
    sol = np.zeros(len(b), dtype=b.dtype)
    
    # Process wavefronts sequentially
    for w0 in range(len(wave_ptr) - 1):
        # Process supernodes in parallel within each wavefront
        for w1 in prange(wave_ptr[w0], wave_ptr[w0 + 1]):
            i = waves[w1]
            col_b = sn_begin[i]
            col_e = sn_begin[i + 1]
            sup_width = col_e - col_b
            sup_len = Lip[i + 1] - Lip[i]
            val_beg = Lp[col_b]
            
            # Solve diagonal block
            lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
            
            # Matrix-vector product for off-diagonal
            off_diag_val = Lx[val_beg + sup_width:]
            matvec_BLAS_python(sup_len, sup_len - sup_width, sup_width,
                               off_diag_val, x[col_b:col_e], sol)
            
            # Scatter updates
            cnt = 0
            for j in range(Lip[i] + sup_width, Lip[i + 1]):
                x[Li[j]] -= sol[cnt]
                sol[cnt] = 0
                cnt += 1


# =============================================================================
# Numba GPU Sync-Free Implementation
# =============================================================================

@cuda.jit
def sptrsv_csc_supernodal_numba_cuda_kernel(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, 
                                            tmp_mem, max_width, flag_thread):
    """
    GPU kernel for sync-free supernodal SpTRSV.
    Uses atomic operations and flags for lock-free synchronization.
    """
    i = cuda.grid(1)
    M = len(sn_begin) - 1
    
    if i < M:
        # Wait until all dependencies are satisfied
        while cuda.atomic.add(flag_thread, i, 0) > 0:
            pass
        
        sol = cuda.shared.array(shape=(1000,), dtype=float64)
        col_b = sn_begin[i]
        col_e = sn_begin[i + 1]
        sup_width = col_e - col_b
        sup_len = Lip[i + 1] - Lip[i]
        val_beg = Lp[col_b]
        
        # Solve diagonal block
        lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        
        # Matrix-vector product for off-diagonal
        off_diag_val = Lx[val_beg + sup_width:]
        matvec_BLAS_python(sup_len, sup_len - sup_width, sup_width,
                           off_diag_val, x[col_b:col_e], sol)
        
        # Scatter updates with atomic operations and flag decrements
        cnt = 0
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            tmp = -sol[cnt]
            idx = Li[j]
            cuda.atomic.add(x, idx, tmp)
            idxs = col_to_sup[idx]
            cuda.atomic.add(flag_thread, idxs, -1)
            sol[cnt] = 0
            cnt += 1


def sptrsv_csc_supernodal_numba_cuda(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, 
                                     flag_thread, max_width):
    """
    Wrapper for GPU sync-free supernodal SpTRSV.
    
    Args:
        sn_begin: Supernode column boundaries (device array)
        col_to_sup: Column to supernode mapping (device array)
        Lip: Supernode row index pointers (device array)
        Li: Row indices (device array)
        Lp: Column pointers (device array)
        Lx: Non-zero values (device array)
        x: Solution vector (device array, in-place)
        flag_thread: Dependency flags (device array)
        max_width: Maximum supernode width
    """
    tmp_mem = 1
    threads_per_block = 1
    blocks_per_grid = (len(sn_begin) + threads_per_block - 1) // threads_per_block
    sptrsv_csc_supernodal_numba_cuda_kernel[blocks_per_grid, threads_per_block](
        sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, tmp_mem, max_width, flag_thread
    )
    cuda.synchronize()


# =============================================================================
# Numba GPU Wavefront Implementation
# =============================================================================

@cuda.jit
def sptrsv_csc_supernodal_wavefront_cuda_kernel(sn_begin, Lip, Li, Lp, Lx, x, 
                                                waves, wave_start, wave_end):
    """
    GPU kernel for wavefront-based supernodal SpTRSV.
    Processes supernodes within a wavefront in parallel.
    """
    i = cuda.grid(1)
    wave_size = wave_end - wave_start
    
    if i < wave_size:
        sn_id = waves[wave_start + i]
        sol = cuda.shared.array(shape=(1000,), dtype=float64)
        
        col_b = sn_begin[sn_id]
        col_e = sn_begin[sn_id + 1]
        sup_width = col_e - col_b
        sup_len = Lip[sn_id + 1] - Lip[sn_id]
        val_beg = Lp[col_b]
        
        # Solve diagonal block
        lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        
        # Matrix-vector product for off-diagonal
        off_diag_val = Lx[val_beg + sup_width:]
        matvec_BLAS_python(sup_len, sup_len - sup_width, sup_width,
                           off_diag_val, x[col_b:col_e], sol)
        
        # Scatter updates with atomic operations
        cnt = 0
        for j in range(Lip[sn_id] + sup_width, Lip[sn_id + 1]):
            tmp = -sol[cnt]
            idx = Li[j]
            cuda.atomic.add(x, idx, tmp)
            sol[cnt] = 0
            cnt += 1


def sptrsv_csc_supernodal_wavefront_cuda(sn_begin, Lip, Li, Lp, Lx, x, wave_ptr, waves):
    """
    Wrapper for GPU wavefront supernodal SpTRSV.
    Launches one kernel per wavefront with synchronization between wavefronts.
    
    Args:
        sn_begin: Supernode column boundaries (device array)
        Lip: Supernode row index pointers (device array)
        Li: Row indices (device array)
        Lp: Column pointers (device array)
        Lx: Non-zero values (device array)
        x: Solution vector (device array, in-place)
        wave_ptr: Wavefront boundaries (device array)
        waves: Supernode IDs grouped by wavefront (device array)
    """
    threads_per_block = 32
    
    for w in range(len(wave_ptr) - 1):
        wave_start = wave_ptr[w]
        wave_end = wave_ptr[w + 1]
        wave_size = wave_end - wave_start
        blocks_per_grid = (wave_size + threads_per_block - 1) // threads_per_block
        
        sptrsv_csc_supernodal_wavefront_cuda_kernel[blocks_per_grid, threads_per_block](
            sn_begin, Lip, Li, Lp, Lx, x, waves, wave_start, wave_end
        )
        cuda.synchronize()
