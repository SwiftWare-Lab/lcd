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
# Baseline Numba CUDA Kernels
# ============================================================================

@cuda.jit
def ilu0_csr_numba_wavefront_kernel(data, indices, indptr, M, wavefronts, wave_start, wave_end, diag_pos):
    gid = cuda.grid(1)
    span = wave_end - wave_start
    if gid >= span:
        return
    row = wavefronts[wave_start + gid]
    if row >= M:
        return
    row_start = indptr[row]
    row_end = indptr[row + 1]
    for ptr in range(row_start, row_end):
        col = indices[ptr]
        if col < row:
            diag_ptr = diag_pos[col]
            diag_val = data[diag_ptr]
            col_start = indptr[col]
            col_end = indptr[col + 1]
            if diag_val != 0.0:
                l_val = data[ptr] / diag_val
                data[ptr] = l_val
                for j_ptr in range(ptr + 1, row_end):
                    j = indices[j_ptr]
                    for cj_ptr in range(col_start, col_end):
                        if indices[cj_ptr] == j:
                            data[j_ptr] = data[j_ptr] - l_val * data[cj_ptr]
                            break


@cuda.jit
def ilu0_csr_numba_sync_free_kernel(data, indices, indptr, M, dep_counts, succ_offsets, succ_indices, diag_pos):
    row = cuda.grid(1)
    if row >= M:
        return
    while cuda.atomic.add(dep_counts, row, 0) > 0:
        pass
    row_start = indptr[row]
    row_end = indptr[row + 1]
    for ptr in range(row_start, row_end):
        col = indices[ptr]
        if col < row:
            diag_ptr = diag_pos[col]
            diag_val = data[diag_ptr]
            col_start = indptr[col]
            col_end = indptr[col + 1]
            if diag_val != 0.0:
                l_val = data[ptr] / diag_val
                data[ptr] = l_val
                for j_ptr in range(ptr + 1, row_end):
                    j = indices[j_ptr]
                    for cj_ptr in range(col_start, col_end):
                        if indices[cj_ptr] == j:
                            data[j_ptr] = data[j_ptr] - l_val * data[cj_ptr]
                            break
    s0 = succ_offsets[row]
    s1 = succ_offsets[row + 1]
    for p in range(s0, s1):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


def ilu0_csr_numba_wavefront(data, indices, indptr, M, wave_start, wavefronts, diag_pos, threads_per_block=1):
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        start = wave_start[i]
        end = wave_start[i + 1]
        grid_len = end - start
        blocks_per_grid = (grid_len + threads_per_block - 1) // threads_per_block
        ilu0_csr_numba_wavefront_kernel[blocks_per_grid, threads_per_block](
            data, indices, indptr, M, wavefronts, start, end, diag_pos
        )
        cuda.synchronize()


def ilu0_csr_numba_sync_free(data, indices, indptr, M, dep_counts, succ_offsets, succ_indices, diag_pos, threads_per_block=1):
    blocks_per_grid = (M + threads_per_block - 1) // threads_per_block
    ilu0_csr_numba_sync_free_kernel[blocks_per_grid, threads_per_block](
        data, indices, indptr, M, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


# ============================================================================
# Helper Functions
# ============================================================================

def compute_ilu_dependency_counts_csr(A):
    n = A.shape[0]
    dep_counts = np.zeros(n, dtype=np.int32)
    indptr = A.indptr
    indices = A.indices
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            k = indices[p]
            if k < i:
                dep_counts[i] += 1
    return dep_counts

def build_successors_from_csr(A):
    Ac = A.tocsc()
    n = A.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for col in range(n):
        s = Ac.indptr[col]
        e = Ac.indptr[col + 1]
        for r in Ac.indices[s:e]:
            if r > col:
                counts[col] += 1
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]
    succ = np.empty(offsets[-1], dtype=np.int32)
    write = offsets.copy()
    for col in range(n):
        s = Ac.indptr[col]
        e = Ac.indptr[col + 1]
        for r in Ac.indices[s:e]:
            if r > col:
                pos = write[col]
                succ[pos] = r
                write[col] += 1
    return offsets, succ


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

def find_diagonal_positions(A):
    n = A.shape[0]
    diag_pos = np.zeros(n, dtype=np.int32)
    for i in range(n):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        for ptr in range(row_start, row_end):
            if A.indices[ptr] == i:
                diag_pos[i] = ptr
                break
    return diag_pos

def compute_ilu_dependency_graph(A):
    return compute_ilu_dependency_counts_csr(A)

def compute_ilu_wavefront_csr(A):
    n = A.shape[0]
    dep_counts = compute_ilu_dependency_counts_csr(A)
    
    A_csc = A.tocsc()
    
    wavefront_list = []
    remaining = set(range(n))
    
    while remaining:
        wavefront = [i for i in remaining if dep_counts[i] == 0]
        if not wavefront:
            break
        
        wavefront_list.append(np.array(wavefront, dtype=np.int32))
        
        for i in wavefront:
            remaining.remove(i)
        
        for i in wavefront:
            col_start = A_csc.indptr[i]
            col_end = A_csc.indptr[i + 1]
            for ptr in range(col_start, col_end):
                j = A_csc.indices[ptr]
                if j > i and j in remaining:
                    dep_counts[j] -= 1
    
    return wavefront_list


def cusparse_ilu0(A_gpu):
    ilu_result = cp_spilu(A_gpu, fill_factor=1.0, drop_tol=0.0)

    L_gpu = ilu_result.L.tocsr()
    U_gpu = ilu_result.U.tocsr()
    
    result_gpu = cp_tril(L_gpu, k=-1) + cp_triu(U_gpu, k=0)
    
    return result_gpu
