"""
Legacy SpTRSV implementations that are not currently used in benchmarks.
These implementations may be useful for reference or future experiments.
"""

import numpy as np
import torch
import triton
import triton.language as tl
from numba import cuda
import numba

REAL_TYPE = torch.float64
REAL_TYPE_TL = tl.float64
DEVICE = torch.device("cuda:0")


# ============================================================================
# Triton Kernels - Sync-Free CSR
# ============================================================================

@triton.jit
def sptrsv_sync_free_csr(
    data_ptr, indices_ptr, indptr_ptr, map_ptr, map_vals, rhs_vec, y_ptr,
    M, N, nnz, count_dep
):
    row = tl.program_id(axis=0)
    is_valid_row = row < M
    while tl.atomic_add(count_dep + row, 0) != 1:
        pass
    start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
    rhs_val = tl.load(rhs_vec + row)
    diag = tl.load(data_ptr + end-1)
    y_val =  tl.cast(0.0, REAL_TYPE_TL)
    for i in range(start, end-1, 32):
        nnz_indx = i + tl.arange(0, 32)
        mask = nnz_indx < end - 1
        indices_vec = tl.load(indices_ptr + nnz_indx, mask=mask, other=0)
        y_vec = tl.load(y_ptr + indices_vec, mask=is_valid_row, other=0)
        val_vec = tl.load(data_ptr + nnz_indx, mask=mask, other=0)
        product = y_vec * val_vec
        y_val += tl.sum(product, axis=0)
    y_val = (rhs_val - y_val) / diag
    tl.store(y_ptr + row, y_val)
    start_col = tl.load(map_ptr + row, mask=is_valid_row, other=0)
    end_col = tl.load(map_ptr + row + 1, mask=is_valid_row, other=0)

    for i in range(start_col, end_col, 32):
        nnz_indx = i + tl.arange(0, 32)
        values = tl.load(map_vals + nnz_indx, mask=nnz_indx < end_col, other=0)
        tl.atomic_add(count_dep + values, -1, mask=values < M)


def sptrsv_sync_free_triton_csr(data, indices, indptr, x, M, N, nnz, dep_counts, map_ptr, map_vals):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)
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


# ============================================================================
# Triton Kernels - Sync-Free CSC (Alternative Versions)
# ============================================================================

@triton.jit
def sptrsv_sync_free_csc2(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    col = tl.program_id(axis=0)
    is_valid_row = col < M
    if is_valid_row:
        while tl.atomic_add(count_dep + col, 0) != 1:
            pass
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1 , mask=is_valid_row, other=0)

        tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)  )/ tl.load(data_ptr + start))
        for j in range(start+1, end):
            val = tl.load(data_ptr + j)
            x_val = tl.load(y_ptr + col)
            r_i = tl.load(indices_ptr + j)
            tl.atomic_add(y_ptr + r_i, val * x_val)
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
            pass
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
        col_length = end - start

        y_col_sol = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + start)
        tl.store(y_ptr + col, y_col_sol)

        # Thread-level processing
        thread_idx = tl.arange(0, BLOCK_SIZE)
        block_start = start + 1 + thread_idx

        mask = block_start < end
        data_col = tl.load(data_ptr + block_start, mask=mask, other=0)

        partial_sum = data_col * y_col_sol

        indices_col = tl.load(indices_ptr + block_start, mask=mask, other=0)

        valid_indices = mask

        # Scatter-add contributions to y_ptr
        tl.atomic_add(y_ptr + indices_col, partial_sum, mask=valid_indices)

        tl.atomic_add(count_dep + indices_col, -1, mask=valid_indices)


def sptrsv_sync_free_triton(data, indices, indptr, x, M, N, nnz, dep_counts, max_col_length):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)
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


# ============================================================================
# Triton Kernels - Wavefront CSR
# ============================================================================

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
        diag = tl.load(data_ptr + end - 1)
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        
        for j in range(start, end - 1):
            val = tl.load(data_ptr + j)
            col_idx = tl.load(indices_ptr + j)
            x_val = tl.load(y_ptr + col_idx)
            y_val += val * x_val
        
        y_val = (rhs_val - y_val) / diag
        tl.store(y_ptr + row, y_val)


def sptrsv_triton_csr(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
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


# ============================================================================
# Triton Kernels - Wavefront CSC
# ============================================================================

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
        tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)  )/ tl.load(data_ptr + start))
        for j in range(start+1, end):
            val = tl.load(data_ptr + j)
            x_val = tl.load(y_ptr + col)
            r_i = tl.load(indices_ptr + j)
            tl.atomic_add(y_ptr + r_i, val * x_val)


def sptrsv_triton(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
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


# ============================================================================
# Triton Kernels - Mapping Variants with Vectorization
# ============================================================================

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
        
        # Compute y[col]
        y_col_val = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + start)
        tl.store(y_ptr + col, y_col_val)
        
        # Vectorized scatter: process BLOCK_SIZE elements at a time
        for i in range(start + 1, end, BLOCK_SIZE):
            nnz_idx = i + tl.arange(0, BLOCK_SIZE)
            mask = nnz_idx < end
            data_col = tl.load(data_ptr + nnz_idx, mask=mask, other=0)
            row_indices = tl.load(indices_ptr + nnz_idx, mask=mask, other=0)
            tl.atomic_add(y_ptr + row_indices, data_col * y_col_val, mask=mask)


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
        diag = tl.load(data_ptr + end - 1)
        
        # Vectorized reduction: process BLOCK_SIZE elements at a time
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        for i in range(start, end - 1, BLOCK_SIZE):
            nnz_idx = i + tl.arange(0, BLOCK_SIZE)
            mask = nnz_idx < end - 1
            col_indices = tl.load(indices_ptr + nnz_idx, mask=mask, other=0)
            y_vec = tl.load(y_ptr + col_indices, mask=mask, other=0)
            val_vec = tl.load(data_ptr + nnz_idx, mask=mask, other=0)
            product = y_vec * val_vec
            y_val += tl.sum(product, axis=0)
        
        y_val = (rhs_val - y_val) / diag
        tl.store(y_ptr + row, y_val)


@triton.jit
def sptrsv_kernel_csr_sync_free_mapping(
    data_ptr, indices_ptr, indptr_ptr, map_ptr, map_vals, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    row = tl.program_id(axis=0)
    is_valid_row = row < M
    
    if is_valid_row:
        # Busy-wait for dependencies
        while tl.atomic_add(count_dep + row, 0) != 1:
            pass
        
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        
        rhs_val = tl.load(rhs_vec + row)
        diag = tl.load(data_ptr + end - 1)
        
        # Vectorized reduction
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        for i in range(start, end - 1, BLOCK_SIZE):
            nnz_idx = i + tl.arange(0, BLOCK_SIZE)
            mask = nnz_idx < end - 1
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


# ============================================================================
# Numba CUDA Kernels - Unrolling Variants (Use sptrsv_unrolled.py instead)
# ============================================================================
# NOTE: The following unrolling variants have been moved to sptrsv_unrolled.py
# and are kept here only for reference. Use the versions in sptrsv_unrolled.py
# for actual benchmarking.

def _to_device_int32(arr):
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
    if isinstance(arr, DeviceNDArrayBase):
        return arr
    return cuda.to_device(np.asarray(arr, dtype=np.int32))


def _to_host_int32(arr):
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
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


@cuda.jit
def sptrsv_numba_csc_wavefront_unrolling_kernel(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len, unroll_factor):
    """
    Thread-level parallelism with loop unrolling for SpTRSV CSC wavefront.
    Each block processes one column; multiple threads cooperate on scatter with unrolled loop.
    """
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    col = wavefronts[wave_offset + block_id]
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    # Thread 0 computes y[col]
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    
    # Unrolled scatter
    j = col_start + 1 + tid
    stride = unroll_factor * block_size
    while j + (unroll_factor - 1) * block_size < col_end:
        # Explicit unrolling to avoid magic numbers
        idx = j
        row_idx = indices[idx]
        val = data[idx]
        cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 1:
            idx = j + block_size
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 2:
            idx = j + 2 * block_size
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 3:
            idx = j + 3 * block_size
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 4:
            for k in range(4, 64):
                if k < unroll_factor:
                    idx = j + k * block_size
                    row_idx = indices[idx]
                    val = data[idx]
                    cuda.atomic.add(y, row_idx, val * y_col)
        j += stride
    
    # Remainder
    while j < col_end:
        row_idx = indices[j]
        val = data[j]
        cuda.atomic.add(y, row_idx, val * y_col)
        j += block_size


def sptrsv_numba_csc_wavefront_unrolling(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128, unroll_factor=4):
    from sptrsv import sptrsv_numba_csc_kernel
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wavefronts = _to_device_int32(wavefronts)
    wavefronts_host = _to_host_int32(wavefronts)
    indptr_host = _to_host_int32(indptr)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = wave_start[i + 1] - wave_start[i]
        if wave_len <= 0:
            continue
        blocks_per_grid = int(wave_len)
        max_work = 0
        for col in wavefronts_host[offset:offset + wave_len]:
            col_start = int(indptr_host[col])
            col_end = int(indptr_host[col + 1])
            max_work = max(max_work, col_end - col_start - 1)
        if max_work <= _SMALL_WORK_THRESHOLD:
            sptrsv_numba_csc_kernel[blocks_per_grid, 1](
                data, indices, indptr, b, d_y, M, N, nnz, d_wavefronts, np.int32(offset)
            )
            continue
        block_threads = _select_block_threads(max_work, max_threads=threads_per_block)
        sptrsv_numba_csc_wavefront_unrolling_kernel[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len), np.int32(unroll_factor)
        )
        cuda.synchronize()
    return d_y


@cuda.jit
def sptrsv_numba_csc_sync_free_unrolling_kernel(data, indices, indptr, b, y, M, dep_counts, unroll_factor):
    """
    Thread-level parallelism with loop unrolling for SpTRSV CSC sync-free.
    Each block processes one column with unrolled scatter when dependencies are satisfied.
    """
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    
    # Busy-wait for dependencies
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    # Thread 0 computes y[col]
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    
    # Unrolled scatter
    j = col_start + 1 + tid
    stride = unroll_factor * block_size
    while j + (unroll_factor - 1) * block_size < col_end:
        # Explicit unrolling to avoid magic numbers
        idx = j
        row_idx = indices[idx]
        val = data[idx]
        cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 1:
            idx = j + block_size
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 2:
            idx = j + 2 * block_size
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 3:
            idx = j + 3 * block_size
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        if unroll_factor > 4:
            for k in range(4, 64):
                if k < unroll_factor:
                    idx = j + k * block_size
                    row_idx = indices[idx]
                    val = data[idx]
                    cuda.atomic.add(y, row_idx, val * y_col)
        j += stride
    
    # Remainder
    while j < col_end:
        row_idx = indices[j]
        val = data[j]
        cuda.atomic.add(y, row_idx, val * y_col)
        j += block_size
    
    # Ensure all threads complete scatter operations before signaling
    cuda.syncthreads()
    
    # Thread 0 signals rows
    if tid == 0:
        cuda.threadfence()
        
        # Signal each row that was updated
        for j in range(col_start + 1, col_end):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


def sptrsv_numba_csc_sync_free_unrolling(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128, unroll_factor=4):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_unrolling_kernel[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, np.int32(unroll_factor)
    )
    return d_y


@cuda.jit
def sptrsv_numba_csr_wavefront_unrolling_kernel(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len, unroll_factor):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    row = wavefronts[wave_offset + block_id]
    row_start = indptr[row]
    row_end = indptr[row + 1]
    shared_sum = cuda.shared.array(128, dtype=numba.float64)
    partial_sum = 0.0
    j = row_start + tid
    stride = unroll_factor * block_size

    while j + (unroll_factor - 1) * block_size < row_end - 1:
        # Explicit unrolling to avoid magic numbers
        idx = j
        partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 1:
            idx = j + block_size
            partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 2:
            idx = j + 2 * block_size
            partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 3:
            idx = j + 3 * block_size
            partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 4:
            for k in range(4, 64):
                if k < unroll_factor:
                    idx = j + k * block_size
                    partial_sum += data[idx] * y[indices[idx]]
        j += stride

    while j < row_end - 1:
        partial_sum += data[j] * y[indices[j]]
        j += block_size
    
    shared_sum[tid] = partial_sum
    cuda.syncthreads()

    s = block_size // 2
    while s > 0:
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
        cuda.syncthreads()
        s //= 2
        
    if tid == 0:
        diag = data[row_end - 1]
        y[row] = (b[row] - shared_sum[0]) / diag


def sptrsv_numba_csr_wavefront_unrolling(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128, unroll_factor=4):
    from sptrsv import sptrsv_numba_csr_wavefront_kernel
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wavefronts = _to_device_int32(wavefronts)
    wavefronts_host = _to_host_int32(wavefronts)
    indptr_host = _to_host_int32(indptr)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = wave_start[i + 1] - wave_start[i]
        if wave_len <= 0:
            continue
        blocks_per_grid = int(wave_len)
        max_work = 0
        for row in wavefronts_host[offset:offset + wave_len]:
            row_start = int(indptr_host[row])
            row_end = int(indptr_host[row + 1])
            max_work = max(max_work, row_end - row_start - 1)
        if max_work <= _SMALL_WORK_THRESHOLD:
            sptrsv_numba_csr_wavefront_kernel[blocks_per_grid, 1](
                data, indices, indptr, b, d_y, M, N, nnz, d_wavefronts, np.int32(offset)
            )
            continue
        block_threads = _select_block_threads(max_work, max_threads=threads_per_block)
        sptrsv_numba_csr_wavefront_unrolling_kernel[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len), np.int32(unroll_factor)
        )
        cuda.synchronize()
    return d_y


@cuda.jit
def sptrsv_numba_csr_sync_free_unrolling_kernel(data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices, unroll_factor):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    row = block_id

    while cuda.atomic.add(dep_counts, row, 0) > 0:
        pass

    cuda.threadfence()
    
    row_start = indptr[row]
    row_end = indptr[row + 1]

    shared_sum = cuda.shared.array(128, dtype=numba.float64)

    partial_sum = 0.0
    j = row_start + tid
    stride = unroll_factor * block_size
    
    # Unrolled loop
    while j + (unroll_factor - 1) * block_size < row_end - 1:
        # Explicit unrolling to avoid magic numbers
        idx = j
        partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 1:
            idx = j + block_size
            partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 2:
            idx = j + 2 * block_size
            partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 3:
            idx = j + 3 * block_size
            partial_sum += data[idx] * y[indices[idx]]
        if unroll_factor > 4:
            for k in range(4, 64):
                if k < unroll_factor:
                    idx = j + k * block_size
                    partial_sum += data[idx] * y[indices[idx]]
        j += stride

    while j < row_end - 1:
        partial_sum += data[j] * y[indices[j]]
        j += block_size
    
    shared_sum[tid] = partial_sum
    cuda.syncthreads()

    s = block_size // 2
    while s > 0:
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        diag = data[row_end - 1]
        y[row] = (b[row] - shared_sum[0]) / diag

        cuda.threadfence()

        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            succ_row = succ_indices[p]
            cuda.atomic.add(dep_counts, succ_row, -1)


def sptrsv_numba_csr_sync_free_unrolling(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128, unroll_factor=4):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csr_sync_free_unrolling_kernel[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, succ_offsets, succ_indices, np.int32(unroll_factor)
    )
    return d_y


# ============================================================================
# CUDA Graph Variants
# ============================================================================

@cuda.jit
def sptrsv_numba_csr_wavefront_cudagraph_kernel(data, indices, indptr, b, y, rows):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= rows.size:
        return

    row = rows[block_id]
    row_start = indptr[row]
    row_end = indptr[row + 1]
    
    shared_sum = cuda.shared.array(128, dtype=numba.float64)
    
    partial_sum = 0.0
    for j in range(row_start + tid, row_end - 1, block_size):
        partial_sum += data[j] * y[indices[j]]
    
    shared_sum[tid] = partial_sum
    cuda.syncthreads()

    s = block_size // 2
    while s > 0:
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        diag = data[row_end - 1]
        y[row] = (b[row] - shared_sum[0]) / diag


def sptrsv_numba_csr_wavefront_cudagraph(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
    import cupy as cp
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_waves = len(wave_start) - 1
    s1 = cp.cuda.Stream(non_blocking=True)
    graphs = []
    nb_stream = cuda.external_stream(s1.ptr)
    s1.begin_capture()
    for i in range(num_waves):
        wave_rows = wavefronts[wave_start[i]:wave_start[i + 1]]
        d_wave_rows = cuda.to_device(wave_rows)
        blocks_per_grid = len(wave_rows)
        sptrsv_numba_csr_wavefront_cudagraph_kernel[blocks_per_grid, threads_per_block,nb_stream](
            data, indices, indptr, b, d_y, d_wave_rows
        )
    g = s1.end_capture()
    graphs.append(g)

    for _ in range(1):
        for g in graphs:
            g.launch(stream=s1)
            cuda.synchronize()
    return d_y


# ============================================================================
# Simple CPU Reference Implementations
# ============================================================================

def simple_sptrsv_csc_syncfree(L, b):
    """Simple CPU reference for sync-free SpTRSV (CSC format)"""
    from bench_utils import incoming_edge
    n = L.shape[0]
    x = np.zeros(n)
    in_edges = incoming_edge(L)
    ready_cols = np.where(in_edges == 1)[0]

    while ready_cols.size > 0:
        i = ready_cols[0]
        ready_cols = ready_cols[1:]
        x[i] = (b[i] - x[i]) / L.data[L.indptr[i]]
        for j in range(L.indptr[i]+1, L.indptr[i + 1]):
            x[L.indices[j]] += L.data[j] * x[i]
            in_edges[L.indices[j]] -= 1
            if in_edges[L.indices[j]] == 1:
                ready_cols = np.append(ready_cols, L.indices[j])
    return x

