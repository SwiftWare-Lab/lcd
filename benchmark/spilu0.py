import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

from numba import config
from numba import int32, float64

config.CUDA_ENABLE_PYNVJITLINK = 1

REAL_TYPE = np.float64
NUM_RUNS = 1

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
    """Pre-compute diagonal positions for each row to avoid linear search"""
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

########################################################
########################################################


# many alternatives to this, may choose for tradeoffs
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

@cuda.jit
def ilu0_csr_wavefront_block_kernel(data, indices, indptr, rows, diag_pos):
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
            # only thread 0 does l_ik
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            # Each thread updates a strided subset
            for j_ptr in range(k_ptr + 1 + tid, row_end, cuda.blockDim.x):
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]

        cuda.syncthreads()
        k_ptr += 1


def ilu0_csr_numba_wavefront_block(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_block_kernel[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


@cuda.jit
def ilu0_csr_sync_free_block_kernel(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x  # one row per block
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
            for j_ptr in range(k_ptr + 1 + tid, row_end, cuda.blockDim.x):
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
        cuda.syncthreads()
        k_ptr += 1

    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


def ilu0_csr_numba_sync_free_block(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_block_kernel[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()
