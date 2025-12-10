import numpy as np
import numba
from numba import njit
import sys
import os
# Add project root to path for compiler module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# Add benchmark directory to path for bench_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from compiler.loop_carried_compiler import MyCompiler

import numpy as np
from scipy.sparse.linalg import spsolve_triangular  # CPU sparse solver
import cupy as cp  # For GPU operations
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix  # GPU sparse matrix
# import csr_matrix as cp_csr_matrix  # GPU sparse matrix
from cupyx.scipy.sparse.linalg import spsolve_triangular as cp_spsolve  # GPU sparse solver
import time
from scipy.sparse import csc_matrix
import torch
import triton
import triton.language as tl

from numba import cuda
from numba import config
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase

config.CUDA_ENABLE_PYNVJITLINK = 1

from bench_utils import incoming_edge
from bench_utils import compute_wavefront


REAL_TYPE = torch.float64
REAL_TYPE_TL = tl.float64

NUM_RUNS=1



DEVICE = torch.device("cuda:0")



def _to_device_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr
    return cuda.to_device(np.asarray(arr, dtype=np.int32))


def _to_host_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr.copy_to_host()
    return np.asarray(arr, dtype=np.int32)


def _select_block_threads(work_items, *, min_threads=32, max_threads=128):
    if work_items <= 0:
        return 1
    rounded = 1 << (work_items - 1).bit_length()
    if rounded < min_threads:
        return min_threads
    if rounded > max_threads:
        return max_threads
    return rounded


_SMALL_WORK_THRESHOLD = 8


# Upper triangular CSR solve (backward substitution).
@njit(numba.float64[:](numba.float32[:], numba.int32[:], numba.int32[:], numba.float32[:]), pipeline_class=MyCompiler)
def sptrsv_csr_numba(A, IA, JA, b):
    n = len(IA) - 1
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(IA[i] + 1, IA[i + 1]):
             x[i] -= A[j] * x[JA[j]]
        x[i] /= A[IA[i]] 
    return x


# Triton sync-free CSR solver for upper triangular matrices.
@triton.jit
def sptrsv_sync_free_csr(
    data_ptr, indices_ptr, indptr_ptr, map_ptr, map_vals, rhs_vec, y_ptr,
    M, N, nnz, count_dep
):
    row = tl.program_id(axis=0)  # Get the row id
    # For backward solve, we might logically process rows M-1 down to 0.
    # But if the dependency graph `count_dep` is set up correctly (waiting for rows > row),
    # the thread index `row` works as an ID.
    
    is_valid_row = row < M
    # Wait for dependencies (rows j > row that 'row' depends on)
    while tl.atomic_add(count_dep + row, 0) != 1:
        pass  # Busy-wait

    # Load start and end pointers
    start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
    
    rhs_val = tl.load(rhs_vec + row)
    
    # Upper Triangular: Diagonal is typically at 'start' (first element)
    diag = tl.load(data_ptr + start) 
    
    y_val =  tl.cast(0.0, REAL_TYPE_TL)
    
    # Process off-diagonal elements (start + 1 to end)
    # x[i] = (b[i] - sum(A_ij * x_j)) / A_ii  where j > i
    for i in range(start + 1, end, 32):
        nnz_indx = i + tl.arange(0, 32)
        mask = nnz_indx < end
        indices_vec = tl.load(indices_ptr + nnz_indx, mask=mask, other=0)
        # Access x[j] (y_ptr) which should be ready
        y_vec = tl.load(y_ptr + indices_vec, mask=is_valid_row, other=0)
        val_vec = tl.load(data_ptr + nnz_indx, mask=mask, other=0)
        product = y_vec * val_vec
        y_val += tl.sum(product, axis=0)

    y_val = (rhs_val - y_val) / diag
    tl.store(y_ptr + row, y_val)

    # Signal dependents: rows k < row that need 'row' (Backward dependency)
    start_col = tl.load(map_ptr + row, mask=is_valid_row, other=0)
    end_col = tl.load(map_ptr + row + 1, mask=is_valid_row, other=0)

    for i in range(start_col, end_col, 32):
        nnz_indx = i + tl.arange(0, 32)
        values = tl.load(map_vals + nnz_indx, mask=nnz_indx < end_col, other=0)
        # Signal upstream rows (which are waiting for this result)
        tl.atomic_add(count_dep + values, -1, mask=values < M)


@triton.jit
def sptrsv_sync_free_csc2(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    col = tl.program_id(axis=0)  # Logic ID
    is_valid_row = col < M
    if is_valid_row:
        while tl.atomic_add(count_dep + col, 0) != 1:
            pass   # Busy-wait

        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1 , mask=is_valid_row, other=0)

        # Upper Triangular CSC: Diagonal is typically the LAST element (end-1)
        # Or if transposed CSR: Diagonal is first.
        # Assuming standard Upper Triangular CSC:
        # Col j contains A_ij for i <= j. Diagonal A_jj is at the end.
        
        diag_idx = end - 1
        diag_val = tl.load(data_ptr + diag_idx)
        
        # Calculate x[col]
        # x[j] = (b[j] - accumulated_updates) / A_jj
        val_res = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / diag_val
        tl.store(y_ptr + col, val_res)
        
        # Scatter updates to rows i < col (where A_ij != 0)
        # Iterate from start to end-1 (exclude diagonal)
        for j in range(start, end - 1):
            val = tl.load(data_ptr + j) # A[i, j]
            # r_i is row index i < col
            r_i = tl.load(indices_ptr + j)
            
            # Atomic update x[i] += A_ij * x_j
            # Note: backward substitution subtraction handled by initial b[i] - ... structure or 
            # here we accumulate A_ij*x_j to be subtracted later.
            # Usually: x[i] -= val * val_res. Assuming y_ptr stores updates.
            tl.atomic_add(y_ptr + r_i, val * val_res)
            
            # Signal row i that one of its dependencies (col j) is ready
            tl.atomic_add(count_dep + r_i, -1)


@triton.jit
def sptrsv_sync_free_csc(
        data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
        M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    col = tl.program_id(axis=0)
    is_valid_row = col < M
    if is_valid_row:
        while tl.atomic_add(count_dep + col, 0) != 1:
            pass  # Busy-wait
        
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
        
        # Upper Triangular CSC: Diagonal is last element (end-1)
        diag_idx = end - 1 
        diag_val = tl.load(data_ptr + diag_idx)
        
        y_col_sol = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / diag_val
        tl.store(y_ptr + col, y_col_sol)

        # Thread-level processing for scatter
        # Process elements start to end-1 (Off-diagonal elements A_ij where i < j)
        thread_idx = tl.arange(0, BLOCK_SIZE)
        block_start = start + thread_idx  # Start from beginning

        # Exclude the last element (diagonal)
        mask = block_start < (end - 1)
        
        data_col = tl.load(data_ptr + block_start, mask=mask, other=0)
        partial_sum = data_col * y_col_sol

        indices_col = tl.load(indices_ptr + block_start, mask=mask, other=0)

        valid_indices = mask

        # Scatter-add contributions to y_ptr (upstream rows i < col)
        tl.atomic_add(y_ptr + indices_col, partial_sum, mask=valid_indices)

        # Decrement count_dep for upstream rows
        tl.atomic_add(count_dep + indices_col, -1, mask=valid_indices)


def sptrsv_sync_free_triton(data, indices, indptr, x, M, N, nnz, dep_counts, max_col_length):
    # Allocate result vector on GPU
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)  # One thread per row
    
    # For Upper Triangular, sync-free logic relies on dependency graph.
    # Ensure dep_counts are initialized for backward solve.
    sptrsv_sync_free_csc[grid](
        data,
        indices,
        indptr,
        x,
        y,
        M,
        N,
        nnz,
        BLOCK_SIZE= max_col_length,
        count_dep=d_dep_count)
    return y


def sptrsv_sync_free_triton_csr(data, indices, indptr, x, M, N, nnz, dep_counts, map_ptr, map_vals):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)  # One thread per row
    sptrsv_sync_free_csr[grid](
        data,
        indices,
        indptr,
        map_ptr, map_vals,
        x,
        y,
        M,
        N,
        nnz,
        count_dep=d_dep_count)
    return y


# Triton kernel for upper triangular CSR wavefront-based solve.
@triton.jit
def sptrsv_kernel_csr(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, wavefront
):
    row_loc = tl.program_id(axis=0)
    row = tl.load(wavefront + row_loc)
    is_valid_row = row < M

    if is_valid_row:
        start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
        
        rhs_val = tl.load(rhs_vec + row)
        diag = tl.load(data_ptr + start)
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        
        for j in range(start + 1, end):
            val = tl.load(data_ptr + j)
            col_idx = tl.load(indices_ptr + j)
            x_val = tl.load(y_ptr + col_idx)
            y_val += val * x_val
        
        y_val = (rhs_val - y_val) / diag
        tl.store(y_ptr + row, y_val)


# Triton kernel for upper triangular CSC wavefront-based solve.
@triton.jit
def sptrsv_kernel_csc(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, wavefront
):
    row_loc = tl.program_id(axis=0)
    col = tl.load(wavefront + row_loc)
    is_valid_row = col < M

    if is_valid_row:
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
        
        diag_idx = end - 1
        diag_val = tl.load(data_ptr + diag_idx)
        
        tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)  )/ diag_val)
        
        for j in range(start, end - 1):
            val = tl.load(data_ptr + j)
            x_val = tl.load(y_ptr + col)
            r_i = tl.load(indices_ptr + j)
            tl.atomic_add(y_ptr + r_i, val * x_val)


# Triton kernel for upper triangular CSC mapping-based solve.
@triton.jit
def sptrsv_kernel_csc_mapping(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, wavefront
):
    row_loc = tl.program_id(axis=0)
    col = tl.load(wavefront + row_loc)
    is_valid_row = col < M

    if is_valid_row:
        start = tl.load(indptr_ptr + col)
        end = tl.load(indptr_ptr + col + 1)
        
        diag_idx = end - 1
        
        y_col_val = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + diag_idx)
        tl.store(y_ptr + col, y_col_val)
        
        for i in range(start, end - 1, BLOCK_SIZE):
            nnz_idx = i + tl.arange(0, BLOCK_SIZE)
            mask = nnz_idx < (end - 1)
            data_col = tl.load(data_ptr + nnz_idx, mask=mask, other=0)
            row_indices = tl.load(indices_ptr + nnz_idx, mask=mask, other=0)
            tl.atomic_add(y_ptr + row_indices, data_col * y_col_val, mask=mask)


# Triton kernel for upper triangular CSR mapping-based solve.
@triton.jit
def sptrsv_kernel_csr_mapping(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, wavefront
):
    row_loc = tl.program_id(axis=0)
    row = tl.load(wavefront + row_loc)
    is_valid_row = row < M

    if is_valid_row:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        
        rhs_val = tl.load(rhs_vec + row)
        diag = tl.load(data_ptr + start)
        
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        for i in range(start + 1, end, BLOCK_SIZE):
            nnz_idx = i + tl.arange(0, BLOCK_SIZE)
            mask = nnz_idx < end
            col_indices = tl.load(indices_ptr + nnz_idx, mask=mask, other=0)
            y_vec = tl.load(y_ptr + col_indices, mask=mask, other=0)
            val_vec = tl.load(data_ptr + nnz_idx, mask=mask, other=0)
            product = y_vec * val_vec
            y_val += tl.sum(product, axis=0)
        
        y_val = (rhs_val - y_val) / diag
        tl.store(y_ptr + row, y_val)


# Triton kernel for upper triangular CSR sync-free mapping-based solve.
@triton.jit
def sptrsv_kernel_csr_sync_free_mapping(
    data_ptr, indices_ptr, indptr_ptr, map_ptr, map_vals, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    row = tl.program_id(axis=0)
    is_valid_row = row < M
    
    if is_valid_row:
        while tl.atomic_add(count_dep + row, 0) != 1:
            pass
        
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        
        rhs_val = tl.load(rhs_vec + row)
        # Upper CSR: Diagonal at start
        diag = tl.load(data_ptr + start)
        
        # Vectorized reduction
        # Off-diagonals: start + 1 to end
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        for i in range(start + 1, end, BLOCK_SIZE):
            nnz_idx = i + tl.arange(0, BLOCK_SIZE)
            mask = nnz_idx < end
            col_indices = tl.load(indices_ptr + nnz_idx, mask=mask, other=0)
            y_vec = tl.load(y_ptr + col_indices, mask=mask, other=0)
            val_vec = tl.load(data_ptr + nnz_idx, mask=mask, other=0)
            product = y_vec * val_vec
            y_val += tl.sum(product, axis=0)
        
        y_val = (rhs_val - y_val) / diag
        tl.store(y_ptr + row, y_val)
        
        # Signal successors
        start_col = tl.load(map_ptr + row)
        end_col = tl.load(map_ptr + row + 1)
        for i in range(start_col, end_col, 32):
            nnz_idx = i + tl.arange(0, 32)
            values = tl.load(map_vals + nnz_idx, mask=nnz_idx < end_col, other=0)
            tl.atomic_add(count_dep + values, -1, mask=values < M)


def sptrsv_triton_csr(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    # Allocate result vector on GPU
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_kernel_csr[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            wavefront=wavefronts[wave_start[i]])
    return y


def sptrsv_triton_csr_mapping(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_kernel_csr_mapping[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            BLOCK_SIZE=32,
            wavefront=wavefronts[wave_start[i]])
    return y


def sptrsv_triton_csr_sync_free_mapping(data, indices, indptr, x, M, N, nnz, dep_counts, map_ptr, map_vals):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)
    sptrsv_kernel_csr_sync_free_mapping[grid](
        data,
        indices,
        indptr,
        map_ptr,
        map_vals,
        x,
        y,
        M,
        N,
        nnz,
        BLOCK_SIZE=32,
        count_dep=d_dep_count)
    return y


def sptrsv_triton(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    # Allocate result vector on GPU
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    #wavefront = torch.tensor(wavefronts, dtype=torch.int32, device=DEVICE)
    # for wavefront in wavefronts:
    #     # Launch Triton kernel
    #     grid = (len(wavefront),)
    #     # copy wavefront to GPU
    #     wavefront_gpu = torch.tensor(wavefront, dtype=torch.int32, device=DEVICE)
    num_waves = len(wave_start)-1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_kernel_csc[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            BLOCK_SIZE=128,
            wavefront=wavefronts[wave_start[i]])
    return y


def sptrsv_triton_csc_mapping(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_kernel_csc_mapping[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            BLOCK_SIZE=32,
            wavefront=wavefronts[wave_start[i]])
    return y


# Numba CUDA kernel for upper triangular CSR wavefront-based solve.
@cuda.jit
def sptrsv_numba_csr_wavefront_kernel(data, indices, indptr, b, y, M, N, nnz, wavefronts, wave_start):
    wave_loc = cuda.grid(1)
    row = wavefronts[wave_start + wave_loc]
    if row < M:
        start = indptr[row]
        end = indptr[row + 1]
        y_val = b[row]
        for j in range(start + 1, end):
            val = data[j]
            col_idx = indices[j]
            y_val -= val * y[col_idx]
        y[row] = y_val / data[start]


# Numba CUDA kernel for upper triangular CSR sync-free solve.
@cuda.jit
def sptrsv_numba_csr_sync_free_kernel(data, indices, indptr, b, y, M, N, nnz, incom_edges, succ_offsets, succ_indices):
    row = cuda.grid(1)
    if row < M:
        while cuda.atomic.add(incom_edges, row, 0) > 0:
            pass
        start = indptr[row]
        end = indptr[row + 1]
        y_val = b[row]
        for j in range(start + 1, end):
            val = data[j]
            col_idx = indices[j]
            y_val -= val * y[col_idx]
        y[row] = y_val / data[start]
        
        cuda.threadfence()
        
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            succ_row = succ_indices[p]
            cuda.atomic.add(incom_edges, succ_row, -1)


# Numba CUDA kernel for upper triangular CSC sync-free solve.
@cuda.jit
def sptrsv_numba_csc_sync_free_kernel(data, indices, indptr, b, y, M, N, nnz, incom_edges):
    col = cuda.grid(1)
    if col < M:
        while cuda.atomic.add(incom_edges, col, 0) > 0:
            pass

        cuda.threadfence()
        
        start = indptr[col]
        end = indptr[col + 1]
        y[col] = (b[col] - y[col]) / data[end - 1]
        
        for j in range(start, end - 1):
            val = data[j]
            r_i = indices[j]
            cuda.atomic.add(y, r_i, val * y[col])
        
        cuda.threadfence()
        
        for j in range(start, end - 1):
            r_i = indices[j]
            cuda.atomic.add(incom_edges, r_i, -1)


def sptrsv_numba_csc_sync_free(data, indices, indptr, b, M, N, nnz, in_edges):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_in_edges = cuda.to_device(in_edges) # avoid race in multiple kernel calls
    threads_per_block = 1
    blocks_per_grid = M+1 #(M + threads_per_block - 1) // threads_per_block
    sptrsv_numba_csc_sync_free_kernel[blocks_per_grid, threads_per_block](data, indices, indptr, b, d_y, M, N, nnz, d_in_edges)
    return d_y


def sptrsv_numba_csr_wavefront(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        grid_len = wave_start[i + 1] - wave_start[i]
        threads_per_block = 1
        blocks_per_grid = grid_len
        sptrsv_numba_csr_wavefront_kernel[blocks_per_grid, threads_per_block](data, indices, indptr, b, d_y, M, N, nnz, wavefronts, wave_start[i])
        cuda.synchronize()
    return d_y


def sptrsv_numba_csr_sync_free(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_in_edges = cuda.to_device(in_edges)
    d_succ_offsets = cuda.to_device(succ_offsets)
    d_succ_indices = cuda.to_device(succ_indices)
    threads_per_block = 1
    blocks_per_grid = M + 1
    sptrsv_numba_csr_sync_free_kernel[blocks_per_grid, threads_per_block](data, indices, indptr, b, d_y, M, N, nnz, d_in_edges, d_succ_offsets, d_succ_indices)
    return d_y


# Numba CUDA kernel for upper triangular CSC wavefront-based solve.
@cuda.jit
def sptrsv_numba_csc_kernel(data, indices, indptr, b, y, M, N, nnz, wavefronts, wave_start):
    wave_loc = cuda.grid(1)
    col = wavefronts[wave_start + wave_loc]
    if col < M:
        start = indptr[col]
        end = indptr[col + 1]
        y[col] = (b[col] - y[col]) / data[end - 1]
        for j in range(start, end - 1):
            val = data[j]
            x_val = y[col]
            r_i = indices[j]
            cuda.atomic.add(y, r_i, val * x_val)


def sptrsv_numba_csc(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        grid_len = wave_start[i + 1] - wave_start[i]
        #print("grid_len: ", grid_len)
        threads_per_block = 1
        blocks_per_grid = grid_len #(threads_per_block - 1) // threads_per_block
        sptrsv_numba_csc_kernel[blocks_per_grid, threads_per_block](data, indices, indptr, b, d_y, M, N, nnz, wavefronts, wave_start[i])
    return d_y


def generate_sparse_triangular_matrix(n, density=0.2):
    """Generates a sparse lower triangular matrix with given density."""
    rand_matrix = np.random.rand(n, n) < density
    dense_matrix = np.multiply(rand_matrix, np.random.rand(n, n))
    sparse_lower_tri = np.tril(dense_matrix)
    np.fill_diagonal(sparse_lower_tri, sparse_lower_tri.diagonal() + 100)
    return csc_matrix(sparse_lower_tri)


def cpu_solver(matrix, b):
    print("Running CPU-based solver...")
    time_arr = []
    for i in range(NUM_RUNS):
        start_time = time.time()
        x_cpu = spsolve_triangular(matrix, b, lower=False)
        end_time = time.time()
        time_arr.append(end_time - start_time)
    print(f"CPU solution took: {np.median(time_arr):.6f} seconds")
    return x_cpu, np.median(time_arr)


def cusparse_solver(matrix, b):
    x_gpu = cp_spsolve(matrix, b, lower=False)
    return x_gpu


def gpu_solver(matrix, b):
    print("\nRunning GPU-based solver...")
    matrix_gpu_csc = cp_csc_matrix(matrix.tocsc())
    b_gpu = cp.array(b)
    time_measures = []
    for i in range(NUM_RUNS):
        start_time = time.time()
        x_gpu = cp_spsolve(matrix_gpu_csc, b_gpu, lower=False)
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        time_measures.append(end_time - start_time)
    print(f"GPU solution took: {np.median(time_measures):.6f} seconds")
    return cp.asnumpy(x_gpu), np.median(time_measures)


def simple_sptrsv_csc(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[i]) / L.data[L.indptr[i] ]
        for j in range(L.indptr[i]+1, L.indptr[i + 1]):
            x[L.indices[j]] += L.data[j] * x[i]
    return x


def simple_sptrsv_csc_waves(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for wave in compute_wavefront(L):
        for i in wave:
            x[i] = (b[i] - x[i]) / L.data[L.indptr[i]]
            for j in range(L.indptr[i]+1, L.indptr[i + 1]):
                x[L.indices[j]] += L.data[j] * x[i]
    return x


def simple_sptrsv_csr(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(L.indptr[i], L.indptr[i + 1] - 1):
            x[i] -= L.data[j] * x[L.indices[j]]
        x[i] /= L.data[L.indptr[i + 1] - 1]
    return x


def simple_sptrsv_csr_waves(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for wave in compute_wavefront(L.tocsc()):
        for i in wave:
            x[i] = b[i]
            for j in range(L.indptr[i], L.indptr[i + 1] - 1):
                x[i] -= L.data[j] * x[L.indices[j]]
            x[i] /= L.data[L.indptr[i + 1] - 1]
    return x


# Count incoming dependencies for each row in upper triangular CSR format.
def incoming_edge_csr_rows(L):
    n = L.shape[0]
    in_edges = np.zeros(n, dtype=np.int32)
    for i in range(n):
        row_len = L.indptr[i + 1] - L.indptr[i]
        if row_len > 0:
            in_edges[i] = row_len - 1
    return in_edges


# Build successor lists for upper triangular CSR backward substitution.
def build_csr_successors(L):
    n = L.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for row in range(n):
        for j_idx in range(L.indptr[row] + 1, L.indptr[row + 1]):
            col = L.indices[j_idx]
            if col > row:
                counts[col] += 1
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]
    succ = np.empty(offsets[-1], dtype=np.int32)
    write_pos = offsets.copy()
    for row in range(n):
        for j_idx in range(L.indptr[row] + 1, L.indptr[row + 1]):
            col = L.indices[j_idx]
            if col > row:
                pos = write_pos[col]
                succ[pos] = row
                write_pos[col] += 1
    
    return offsets, succ


# Build successor lists for upper triangular CSC backward substitution.
def build_csc_successors(L):
    n = L.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for col in range(n):
        for i_idx in range(L.indptr[col], L.indptr[col + 1] - 1):
            row = L.indices[i_idx]
            if row < col:
                counts[col] += 1

    offsets = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]

    succ = np.empty(offsets[-1], dtype=np.int32)
    write_pos = offsets.copy()
    for col in range(n):
        for i_idx in range(L.indptr[col], L.indptr[col + 1] - 1):
            row = L.indices[i_idx]
            if row < col:
                pos = write_pos[col]
                succ[pos] = row
                write_pos[col] += 1
    
    return offsets, succ


def simple_sptrsv_csc_syncfree(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    in_edges = incoming_edge(L)
    ready_cols = np.where(in_edges == 1)[0]

    while ready_cols.size > 0:
        i = ready_cols[0]
        # remove i from ready_cols
        ready_cols = ready_cols[1:]
        x[i] = (b[i] - x[i]) / L.data[L.indptr[i] ]
        for j in range(L.indptr[i]+1, L.indptr[i + 1]):
            x[L.indices[j]] += L.data[j] * x[i]
            in_edges[L.indices[j]] -= 1
            if in_edges[L.indices[j]] == 1:
                ready_cols = np.append(ready_cols, L.indices[j])
    return x