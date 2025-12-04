"""
Legacy SpILU0 implementations that are not currently used in benchmarks.
These implementations may be useful for reference or future experiments.
"""

import sys
import os
# Add project root to path for compiler module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# Add benchmark directory to path for bench_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import numba
from numba import njit, cuda
from compiler.loop_carried_compiler import MyCompiler

from scipy.sparse.linalg import spilu
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
import time
from scipy.sparse import csr_matrix, csc_matrix
import torch
import triton
import triton.language as tl

from numba import config
from numba import int32, float64

import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.linalg import spilu as cp_spilu
from cupyx.scipy.sparse import tril as cp_tril, triu as cp_triu

config.CUDA_ENABLE_PYNVJITLINK = 1

REAL_TYPE = np.float64
REAL_TYPE_TL = tl.float64
NUM_RUNS = 1
DEVICE = torch.device("cuda:0")

# ============================================================================
# CPU Reference Implementation
# ============================================================================

@njit(numba.void(numba.float64[:], numba.int32[:], numba.int32[:]), pipeline_class=MyCompiler)
def ilu0_csr_numba(A_data, A_indices, A_indptr):
    n = len(A_indptr) - 1
    for i in range(n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]
        for k_ptr in range(row_start, row_end):
            k = A_indices[k_ptr]
            if k >= i:
                break
            diag_val = 0.0
            for diag_ptr in range(A_indptr[k], A_indptr[k + 1]):
                if A_indices[diag_ptr] == k:
                    diag_val = A_data[diag_ptr]
                    break
            if diag_val != 0.0:
                A_data[k_ptr] = A_data[k_ptr] / diag_val
                l_ik = A_data[k_ptr]
                for j_ptr in range(k_ptr + 1, row_end):
                    j = A_indices[j_ptr]
                    for kj_ptr in range(A_indptr[k], A_indptr[k + 1]):
                        if A_indices[kj_ptr] == j:
                            u_kj = A_data[kj_ptr]
                            A_data[j_ptr] = A_data[j_ptr] - l_ik * u_kj
                            break


def simple_ilu0_csr(A):
    A_copy = A.copy()
    data = A_copy.data
    indices = A_copy.indices
    indptr = A_copy.indptr
    n = A.shape[0]
    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k_ptr in range(row_start, row_end):
            k = indices[k_ptr]
            if k >= i:
                break
            diag_val = 0.0
            for diag_ptr in range(indptr[k], indptr[k + 1]):
                if indices[diag_ptr] == k:
                    diag_val = data[diag_ptr]
                    break
            if diag_val != 0.0:
                data[k_ptr] = data[k_ptr] / diag_val
                l_ik = data[k_ptr]
                for j_ptr in range(k_ptr + 1, row_end):
                    j = indices[j_ptr]
                    for kj_ptr in range(indptr[k], indptr[k + 1]):
                        if indices[kj_ptr] == j:
                            u_kj = data[kj_ptr]
                            data[j_ptr] = data[j_ptr] - l_ik * u_kj
                            break
    return A_copy


def cpu_ilu0_solver(A):
    A = prepare_for_ilu0(A)
    t = []
    ilu = None
    for _ in range(NUM_RUNS):
        start_time = time.time()
        ilu = spilu(A.tocsc(), drop_tol=0.0, fill_factor=1.0)
        t.append(time.time() - start_time)
    return ilu, float(np.median(t))

def gpu_ilu0_solver(A):
    A = prepare_for_ilu0(A)
    A_gpu = cp_csr_matrix(A)
    t = []
    A_result = None
    for _ in range(NUM_RUNS):
        start = time.time()
        A_cpu = A_gpu.get()
        ilu_result = simple_ilu0_csr(A_cpu)
        A_result = cp.asarray(ilu_result.toarray())
        cp.cuda.Stream.null.synchronize()
        t.append(time.time() - start)
    return A_result, float(np.median(t))


def prepare_for_ilu0(A):
    A = A.tocsr(copy=True)
    A.sort_indices()
    n = A.shape[0]
    d = A.diagonal()
    missing = np.where(d == 0)[0]
    if len(missing) > 0:
        eps = 0.0
        A = A + csr_matrix((np.full(len(missing), eps, dtype=A.dtype), (missing, missing)), shape=A.shape)
    return A

# ============================================================================
# Triton Kernels - Wavefront
# ============================================================================

@triton.jit
def ilu0_csr_triton_wavefront_kernel(
    data_ptr, indices_ptr, indptr_ptr, M, wavefront, diag_pos_ptr
):
    row_loc = tl.program_id(axis=0)
    row = tl.load(wavefront + row_loc)
    
    if row < M:
        row_start = tl.load(indptr_ptr + row)
        row_end = tl.load(indptr_ptr + row + 1)
        
        for ptr in range(row_start, row_end):
            col = tl.load(indices_ptr + ptr)
            if col < row:
                diag_ptr = tl.load(diag_pos_ptr + col)
                diag_val = tl.load(data_ptr + diag_ptr)
                col_start = tl.load(indptr_ptr + col)
                col_end = tl.load(indptr_ptr + col + 1)
                
                if diag_val != 0.0:
                    l_val = tl.load(data_ptr + ptr) / diag_val
                    tl.store(data_ptr + ptr, l_val)
                    
                    for j_ptr in range(ptr + 1, row_end):
                        j = tl.load(indices_ptr + j_ptr)
                        found = tl.cast(0, tl.int1)
                        for cj_ptr in range(col_start, col_end):
                            col_j = tl.load(indices_ptr + cj_ptr)
                            if col_j == j and not found:
                                old_val = tl.load(data_ptr + j_ptr)
                                new_val = old_val - l_val * tl.load(data_ptr + cj_ptr)
                                tl.store(data_ptr + j_ptr, new_val)
                                found = tl.cast(1, tl.int1)


def ilu0_csr_triton_wavefront(data, indices, indptr, M, wave_start, wavefronts, diag_pos):
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        start = wave_start[i]
        end = wave_start[i + 1]
        grid_len = end - start
        grid = (grid_len,)
        ilu0_csr_triton_wavefront_kernel[grid](
            data, indices, indptr, M, wavefronts[start:end], diag_pos
        )


# ============================================================================
# Triton Kernels - Sync-Free
# ============================================================================

@triton.jit
def ilu0_csr_triton_sync_free_kernel(
    data_ptr, indices_ptr, indptr_ptr, M, dep_counts, 
    succ_offsets_ptr, succ_indices_ptr, diag_pos_ptr
):
    row = tl.program_id(axis=0)
    
    if row < M:
        while tl.atomic_add(dep_counts + row, 0) > 0:
            pass
        
        row_start = tl.load(indptr_ptr + row)
        row_end = tl.load(indptr_ptr + row + 1)
        
        for ptr in range(row_start, row_end):
            col = tl.load(indices_ptr + ptr)
            if col < row:
                diag_ptr = tl.load(diag_pos_ptr + col)
                diag_val = tl.load(data_ptr + diag_ptr)
                col_start = tl.load(indptr_ptr + col)
                col_end = tl.load(indptr_ptr + col + 1)
                
                if diag_val != 0.0:
                    l_val = tl.load(data_ptr + ptr) / diag_val
                    tl.store(data_ptr + ptr, l_val)
                    
                    for j_ptr in range(ptr + 1, row_end):
                        j = tl.load(indices_ptr + j_ptr)
                        found = tl.cast(0, tl.int1)
                        for cj_ptr in range(col_start, col_end):
                            col_j = tl.load(indices_ptr + cj_ptr)
                            if col_j == j and not found:
                                old_val = tl.load(data_ptr + j_ptr)
                                new_val = old_val - l_val * tl.load(data_ptr + cj_ptr)
                                tl.store(data_ptr + j_ptr, new_val)
                                found = tl.cast(1, tl.int1)

        s0 = tl.load(succ_offsets_ptr + row)
        s1 = tl.load(succ_offsets_ptr + row + 1)
        for p in range(s0, s1):
            s = tl.load(succ_indices_ptr + p)
            tl.atomic_add(dep_counts + s, -1)


def ilu0_csr_triton_sync_free(data, indices, indptr, M, dep_counts, succ_offsets, succ_indices, diag_pos):
    d_dep_counts = dep_counts.clone().detach()
    grid = (M,)
    ilu0_csr_triton_sync_free_kernel[grid](
        data, indices, indptr, M, d_dep_counts,
        succ_offsets, succ_indices, diag_pos
    )


# ============================================================================
# Triton Kernels - Wavefront Mapping
# ============================================================================

@triton.jit
def ilu0_csr_triton_wavefront_mapping_kernel(
    data_ptr, indices_ptr, indptr_ptr, M, wavefront, diag_pos_ptr, BLOCK_SIZE: tl.constexpr
):
    row_loc = tl.program_id(axis=0)
    row = tl.load(wavefront + row_loc)
    
    if row < M:
        row_start = tl.load(indptr_ptr + row)
        row_end = tl.load(indptr_ptr + row + 1)
        
        for ptr in range(row_start, row_end):
            col = tl.load(indices_ptr + ptr)
            if col < row:
                diag_ptr = tl.load(diag_pos_ptr + col)
                diag_val = tl.load(data_ptr + diag_ptr)
                col_start = tl.load(indptr_ptr + col)
                col_end = tl.load(indptr_ptr + col + 1)
                
                if diag_val != 0.0:
                    l_val = tl.load(data_ptr + ptr) / diag_val
                    tl.store(data_ptr + ptr, l_val)
                    
                    # Vectorized update: process BLOCK_SIZE elements at a time
                    for j_base in range(ptr + 1, row_end, BLOCK_SIZE):
                        j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
                        j_mask = j_offsets < row_end
                        
                        j_cols = tl.load(indices_ptr + j_offsets, mask=j_mask, other=0)
                        
                        for cj_ptr in range(col_start, col_end):
                            col_j = tl.load(indices_ptr + cj_ptr)
                            match_mask = (j_cols == col_j) & j_mask
                            if tl.sum(match_mask.to(tl.int32)) > 0:
                                old_vals = tl.load(data_ptr + j_offsets, mask=match_mask, other=0.0)
                                col_val = tl.load(data_ptr + cj_ptr)
                                new_vals = old_vals - l_val * col_val
                                tl.store(data_ptr + j_offsets, new_vals, mask=match_mask)


def ilu0_csr_triton_wavefront_mapping(data, indices, indptr, M, wave_start, wavefronts, diag_pos, block_size=32):
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        start = wave_start[i]
        end = wave_start[i + 1]
        grid_len = end - start
        grid = (grid_len,)
        ilu0_csr_triton_wavefront_mapping_kernel[grid](
            data, indices, indptr, M, wavefronts[start:end], diag_pos, BLOCK_SIZE=block_size
        )


# ============================================================================
# Triton Kernels - Wavefront Unrolling
# ============================================================================

@triton.jit
def ilu0_csr_triton_wavefront_unrolling_kernel(
    data_ptr, indices_ptr, indptr_ptr, M, wavefront, diag_pos_ptr
):
    row_loc = tl.program_id(axis=0)
    row = tl.load(wavefront + row_loc)
    
    if row < M:
        row_start = tl.load(indptr_ptr + row)
        row_end = tl.load(indptr_ptr + row + 1)
        
        for ptr in range(row_start, row_end):
            col = tl.load(indices_ptr + ptr)
            if col < row:
                diag_ptr = tl.load(diag_pos_ptr + col)
                diag_val = tl.load(data_ptr + diag_ptr)
                col_start = tl.load(indptr_ptr + col)
                col_end = tl.load(indptr_ptr + col + 1)
                
                if diag_val != 0.0:
                    l_val = tl.load(data_ptr + ptr) / diag_val
                    tl.store(data_ptr + ptr, l_val)
                    
                    # Unrolled loop: process 4 elements at a time
                    j_ptr = ptr + 1
                    while j_ptr + 3 < row_end:
                        # Process 4 iterations
                        for unroll_i in range(4):
                            j = tl.load(indices_ptr + j_ptr + unroll_i)
                            found = tl.cast(0, tl.int1)
                            for cj_ptr in range(col_start, col_end):
                                col_j = tl.load(indices_ptr + cj_ptr)
                                if col_j == j and not found:
                                    old_val = tl.load(data_ptr + j_ptr + unroll_i)
                                    new_val = old_val - l_val * tl.load(data_ptr + cj_ptr)
                                    tl.store(data_ptr + j_ptr + unroll_i, new_val)
                                    found = tl.cast(1, tl.int1)
                        j_ptr += 4
                    
                    # Remainder
                    while j_ptr < row_end:
                        j = tl.load(indices_ptr + j_ptr)
                        found = tl.cast(0, tl.int1)
                        for cj_ptr in range(col_start, col_end):
                            col_j = tl.load(indices_ptr + cj_ptr)
                            if col_j == j and not found:
                                old_val = tl.load(data_ptr + j_ptr)
                                new_val = old_val - l_val * tl.load(data_ptr + cj_ptr)
                                tl.store(data_ptr + j_ptr, new_val)
                                found = tl.cast(1, tl.int1)
                        j_ptr += 1


def ilu0_csr_triton_wavefront_unrolling(data, indices, indptr, M, wave_start, wavefronts, diag_pos):
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        start = wave_start[i]
        end = wave_start[i + 1]
        grid_len = end - start
        grid = (grid_len,)
        ilu0_csr_triton_wavefront_unrolling_kernel[grid](
            data, indices, indptr, M, wavefronts[start:end], diag_pos
        )


# ============================================================================
# Triton Kernels - Sync-Free Mapping
# ============================================================================

@triton.jit
def ilu0_csr_triton_sync_free_mapping_kernel(
    data_ptr, indices_ptr, indptr_ptr, M, dep_counts,
    succ_offsets_ptr, succ_indices_ptr, diag_pos_ptr, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    
    if row < M:
        while tl.atomic_add(dep_counts + row, 0) > 0:
            pass
        
        row_start = tl.load(indptr_ptr + row)
        row_end = tl.load(indptr_ptr + row + 1)
        
        for ptr in range(row_start, row_end):
            col = tl.load(indices_ptr + ptr)
            if col < row:
                diag_ptr = tl.load(diag_pos_ptr + col)
                diag_val = tl.load(data_ptr + diag_ptr)
                col_start = tl.load(indptr_ptr + col)
                col_end = tl.load(indptr_ptr + col + 1)
                
                if diag_val != 0.0:
                    l_val = tl.load(data_ptr + ptr) / diag_val
                    tl.store(data_ptr + ptr, l_val)
                    
                    # Vectorized update: process BLOCK_SIZE elements at a time
                    for j_base in range(ptr + 1, row_end, BLOCK_SIZE):
                        j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
                        j_mask = j_offsets < row_end
                        
                        j_cols = tl.load(indices_ptr + j_offsets, mask=j_mask, other=0)
                        
                        for cj_ptr in range(col_start, col_end):
                            col_j = tl.load(indices_ptr + cj_ptr)
                            match_mask = (j_cols == col_j) & j_mask
                            if tl.sum(match_mask.to(tl.int32)) > 0:
                                old_vals = tl.load(data_ptr + j_offsets, mask=match_mask, other=0.0)
                                col_val = tl.load(data_ptr + cj_ptr)
                                new_vals = old_vals - l_val * col_val
                                tl.store(data_ptr + j_offsets, new_vals, mask=match_mask)
        
        s0 = tl.load(succ_offsets_ptr + row)
        s1 = tl.load(succ_offsets_ptr + row + 1)
        for p in range(s0, s1):
            s = tl.load(succ_indices_ptr + p)
            tl.atomic_add(dep_counts + s, -1)


def ilu0_csr_triton_sync_free_mapping(data, indices, indptr, M, dep_counts, succ_offsets, succ_indices, diag_pos, block_size=32):
    d_dep_counts = dep_counts.clone().detach()
    grid = (M,)
    ilu0_csr_triton_sync_free_mapping_kernel[grid](
        data, indices, indptr, M, d_dep_counts,
        succ_offsets, succ_indices, diag_pos, BLOCK_SIZE=block_size
    )


# ============================================================================
# Triton Kernels - Sync-Free Unrolling
# ============================================================================

@triton.jit
def ilu0_csr_triton_sync_free_unrolling_kernel(
    data_ptr, indices_ptr, indptr_ptr, M, dep_counts,
    succ_offsets_ptr, succ_indices_ptr, diag_pos_ptr
):
    row = tl.program_id(axis=0)
    
    if row < M:
        while tl.atomic_add(dep_counts + row, 0) > 0:
            pass
        
        row_start = tl.load(indptr_ptr + row)
        row_end = tl.load(indptr_ptr + row + 1)
        
        for ptr in range(row_start, row_end):
            col = tl.load(indices_ptr + ptr)
            if col < row:
                diag_ptr = tl.load(diag_pos_ptr + col)
                diag_val = tl.load(data_ptr + diag_ptr)
                col_start = tl.load(indptr_ptr + col)
                col_end = tl.load(indptr_ptr + col + 1)
                
                if diag_val != 0.0:
                    l_val = tl.load(data_ptr + ptr) / diag_val
                    tl.store(data_ptr + ptr, l_val)
                    
                    # Unrolled loop: process 4 elements at a time
                    j_ptr = ptr + 1
                    while j_ptr + 3 < row_end:
                        # Process 4 iterations
                        for unroll_i in range(4):
                            j = tl.load(indices_ptr + j_ptr + unroll_i)
                            found = tl.cast(0, tl.int1)
                            for cj_ptr in range(col_start, col_end):
                                col_j = tl.load(indices_ptr + cj_ptr)
                                if col_j == j and not found:
                                    old_val = tl.load(data_ptr + j_ptr + unroll_i)
                                    new_val = old_val - l_val * tl.load(data_ptr + cj_ptr)
                                    tl.store(data_ptr + j_ptr + unroll_i, new_val)
                                    found = tl.cast(1, tl.int1)
                        j_ptr += 4
                    
                    # Remainder
                    while j_ptr < row_end:
                        j = tl.load(indices_ptr + j_ptr)
                        found = tl.cast(0, tl.int1)
                        for cj_ptr in range(col_start, col_end):
                            col_j = tl.load(indices_ptr + cj_ptr)
                            if col_j == j and not found:
                                old_val = tl.load(data_ptr + j_ptr)
                                new_val = old_val - l_val * tl.load(data_ptr + cj_ptr)
                                tl.store(data_ptr + j_ptr, new_val)
                                found = tl.cast(1, tl.int1)
                        j_ptr += 1
        
        s0 = tl.load(succ_offsets_ptr + row)
        s1 = tl.load(succ_offsets_ptr + row + 1)
        for p in range(s0, s1):
            s = tl.load(succ_indices_ptr + p)
            tl.atomic_add(dep_counts + s, -1)


def ilu0_csr_triton_sync_free_unrolling(data, indices, indptr, M, dep_counts, succ_offsets, succ_indices, diag_pos):
    d_dep_counts = dep_counts.clone().detach()
    grid = (M,)
    ilu0_csr_triton_sync_free_unrolling_kernel[grid](
        data, indices, indptr, M, d_dep_counts,
        succ_offsets, succ_indices, diag_pos
    )


# ============================================================================
# Numba Unrolling (older implementation kept in spilu0.py, not codegen)
# ============================================================================

@cuda.jit
def ilu0_csr_wavefront_unrolling_kernel(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            # Unrolled loop: process 4 elements at a time
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 3 * cuda.blockDim.x < row_end:
                for unroll_idx in range(4):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 4 * cuda.blockDim.x
            
            # Remainder
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x

        cuda.syncthreads()
        k_ptr += 1


def ilu0_csr_numba_wavefront_unrolling(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_unrolling_kernel[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


@cuda.jit
def ilu0_csr_sync_free_unrolling_kernel(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    while cuda.atomic.add(dep_counts, i, 0) > 0:
        pass

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            
            # Unrolled loop: process 4 elements at a time
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 3 * cuda.blockDim.x < row_end:
                for unroll_idx in range(4):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 4 * cuda.blockDim.x
            
            # Remainder
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x
        
        cuda.syncthreads()
        k_ptr += 1

    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0, s1):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


def ilu0_csr_numba_sync_free_unrolling(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_unrolling_kernel[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


# Helper function for binary search (needed by legacy unrolling)
@cuda.jit(device=True, inline=True)
def bin_search_col(colind, start, end, key):
    lo = start
    hi = end - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        c = colind[mid]
        if c < key:
            lo = mid + 1
        elif c > key:
            hi = mid - 1
        else:
            return mid
    return -1

