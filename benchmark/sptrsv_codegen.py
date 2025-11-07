"""
Code-generated SpTRSV kernels with different unroll factors.
This file contains specialized kernels for unroll factors: 4, 8, 16, 32, 64
to avoid runtime overhead from conditional checks.
"""

import numpy as np
from numba import cuda
import numba
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase


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


# ============================================================================
# CSC Wavefront Unrolling Kernels (5 variants)
# ============================================================================

@cuda.jit
def sptrsv_numba_csc_wavefront_unrolling_kernel_4(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    col = wavefronts[wave_offset + block_id]
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 3 * block_size < col_end:
        for k in range(4):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 4 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size


@cuda.jit
def sptrsv_numba_csc_wavefront_unrolling_kernel_8(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    col = wavefronts[wave_offset + block_id]
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 7 * block_size < col_end:
        for k in range(8):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 8 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size


@cuda.jit
def sptrsv_numba_csc_wavefront_unrolling_kernel_16(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    col = wavefronts[wave_offset + block_id]
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 15 * block_size < col_end:
        for k in range(16):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 16 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size


@cuda.jit
def sptrsv_numba_csc_wavefront_unrolling_kernel_32(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    col = wavefronts[wave_offset + block_id]
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 31 * block_size < col_end:
        for k in range(32):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 32 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size


@cuda.jit
def sptrsv_numba_csc_wavefront_unrolling_kernel_64(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= wave_len:
        return
    
    col = wavefronts[wave_offset + block_id]
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 63 * block_size < col_end:
        for k in range(64):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 64 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size


# CSC Wavefront Launch Functions

def sptrsv_numba_csc_wavefront_unrolling_4(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csc_wavefront_unrolling_kernel_4[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csc_wavefront_unrolling_8(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csc_wavefront_unrolling_kernel_8[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csc_wavefront_unrolling_16(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csc_wavefront_unrolling_kernel_16[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csc_wavefront_unrolling_32(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csc_wavefront_unrolling_kernel_32[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csc_wavefront_unrolling_64(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csc_wavefront_unrolling_kernel_64[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


# ============================================================================
# CSC Sync-Free Unrolling Kernels (5 variants)
# ============================================================================

@cuda.jit
def sptrsv_numba_csc_sync_free_unrolling_kernel_4(data, indices, indptr, b, y, M, dep_counts):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 3 * block_size < col_end:
        for k in range(4):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 4 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size
    
    cuda.syncthreads()
    
    if tid == 0:
        cuda.threadfence()
        for j in range(col_start + 1, col_end):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


@cuda.jit
def sptrsv_numba_csc_sync_free_unrolling_kernel_8(data, indices, indptr, b, y, M, dep_counts):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 7 * block_size < col_end:
        for k in range(8):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 8 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size
    
    cuda.syncthreads()
    
    if tid == 0:
        cuda.threadfence()
        for j in range(col_start + 1, col_end):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


@cuda.jit
def sptrsv_numba_csc_sync_free_unrolling_kernel_16(data, indices, indptr, b, y, M, dep_counts):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 15 * block_size < col_end:
        for k in range(16):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 16 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size
    
    cuda.syncthreads()
    
    if tid == 0:
        cuda.threadfence()
        for j in range(col_start + 1, col_end):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


@cuda.jit
def sptrsv_numba_csc_sync_free_unrolling_kernel_32(data, indices, indptr, b, y, M, dep_counts):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 31 * block_size < col_end:
        for k in range(32):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 32 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size
    
    cuda.syncthreads()
    
    if tid == 0:
        cuda.threadfence()
        for j in range(col_start + 1, col_end):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


@cuda.jit
def sptrsv_numba_csc_sync_free_unrolling_kernel_64(data, indices, indptr, b, y, M, dep_counts):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_start]
    cuda.syncthreads()
    
    y_col = y[col]
    j = col_start + 1 + tid
    
    while j + 63 * block_size < col_end:
        for k in range(64):
            idx = j + k * block_size
            cuda.atomic.add(y, indices[idx], data[idx] * y_col)
        j += 64 * block_size
    
    while j < col_end:
        cuda.atomic.add(y, indices[j], data[j] * y_col)
        j += block_size
    
    cuda.syncthreads()
    
    if tid == 0:
        cuda.threadfence()
        for j in range(col_start + 1, col_end):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


# CSC Sync-Free Launch Functions

def sptrsv_numba_csc_sync_free_unrolling_4(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_unrolling_kernel_4[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts
    )
    return d_y


def sptrsv_numba_csc_sync_free_unrolling_8(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_unrolling_kernel_8[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts
    )
    return d_y


def sptrsv_numba_csc_sync_free_unrolling_16(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_unrolling_kernel_16[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts
    )
    return d_y


def sptrsv_numba_csc_sync_free_unrolling_32(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_unrolling_kernel_32[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts
    )
    return d_y


def sptrsv_numba_csc_sync_free_unrolling_64(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_unrolling_kernel_64[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts
    )
    return d_y


# ============================================================================
# CSR Wavefront Unrolling Kernels (5 variants)
# ============================================================================

@cuda.jit
def sptrsv_numba_csr_wavefront_unrolling_kernel_4(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
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

    while j + 3 * block_size < row_end - 1:
        for k in range(4):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 4 * block_size

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


@cuda.jit
def sptrsv_numba_csr_wavefront_unrolling_kernel_8(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
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

    while j + 7 * block_size < row_end - 1:
        for k in range(8):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 8 * block_size

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


@cuda.jit
def sptrsv_numba_csr_wavefront_unrolling_kernel_16(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
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

    while j + 15 * block_size < row_end - 1:
        for k in range(16):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 16 * block_size

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


@cuda.jit
def sptrsv_numba_csr_wavefront_unrolling_kernel_32(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
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

    while j + 31 * block_size < row_end - 1:
        for k in range(32):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 32 * block_size

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


@cuda.jit
def sptrsv_numba_csr_wavefront_unrolling_kernel_64(data, indices, indptr, b, y, wavefronts, wave_offset, wave_len):
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

    while j + 63 * block_size < row_end - 1:
        for k in range(64):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 64 * block_size

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


# CSR Wavefront Launch Functions

def sptrsv_numba_csr_wavefront_unrolling_4(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csr_wavefront_unrolling_kernel_4[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csr_wavefront_unrolling_8(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csr_wavefront_unrolling_kernel_8[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csr_wavefront_unrolling_16(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csr_wavefront_unrolling_kernel_16[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csr_wavefront_unrolling_32(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csr_wavefront_unrolling_kernel_32[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


def sptrsv_numba_csr_wavefront_unrolling_64(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, threads_per_block=128):
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
        sptrsv_numba_csr_wavefront_unrolling_kernel_64[blocks_per_grid, block_threads](
            data, indices, indptr, b, d_y, d_wavefronts, np.int32(offset), np.int32(wave_len)
        )
        cuda.synchronize()
    return d_y


# ============================================================================
# CSR Sync-Free Unrolling Kernels (5 variants)
# ============================================================================

@cuda.jit
def sptrsv_numba_csr_sync_free_unrolling_kernel_4(data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices):
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
    
    while j + 3 * block_size < row_end - 1:
        for k in range(4):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 4 * block_size

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


@cuda.jit
def sptrsv_numba_csr_sync_free_unrolling_kernel_8(data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices):
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
    
    while j + 7 * block_size < row_end - 1:
        for k in range(8):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 8 * block_size

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


@cuda.jit
def sptrsv_numba_csr_sync_free_unrolling_kernel_16(data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices):
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
    
    while j + 15 * block_size < row_end - 1:
        for k in range(16):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 16 * block_size

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


@cuda.jit
def sptrsv_numba_csr_sync_free_unrolling_kernel_32(data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices):
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
    
    while j + 31 * block_size < row_end - 1:
        for k in range(32):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 32 * block_size

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


@cuda.jit
def sptrsv_numba_csr_sync_free_unrolling_kernel_64(data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices):
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
    
    while j + 63 * block_size < row_end - 1:
        for k in range(64):
            idx = j + k * block_size
            partial_sum += data[idx] * y[indices[idx]]
        j += 64 * block_size

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


# CSR Sync-Free Launch Functions

def sptrsv_numba_csr_sync_free_unrolling_4(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csr_sync_free_unrolling_kernel_4[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, succ_offsets, succ_indices
    )
    return d_y


def sptrsv_numba_csr_sync_free_unrolling_8(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csr_sync_free_unrolling_kernel_8[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, succ_offsets, succ_indices
    )
    return d_y


def sptrsv_numba_csr_sync_free_unrolling_16(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csr_sync_free_unrolling_kernel_16[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, succ_offsets, succ_indices
    )
    return d_y


def sptrsv_numba_csr_sync_free_unrolling_32(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csr_sync_free_unrolling_kernel_32[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, succ_offsets, succ_indices
    )
    return d_y


def sptrsv_numba_csr_sync_free_unrolling_64(data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, threads_per_block=128):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    blocks_per_grid = M
    sptrsv_numba_csr_sync_free_unrolling_kernel_64[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, succ_offsets, succ_indices
    )
    return d_y

