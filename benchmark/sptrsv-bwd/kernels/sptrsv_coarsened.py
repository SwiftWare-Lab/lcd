"""SpTRSV with Coarsening: Thread, Block, and Wavefront Coarsening Strategies (Upper Triangular)"""

import numpy as np
import numba
import sys
import os
import warnings
# Add benchmark directory to path for bench_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from numba import cuda
from numba import config
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
from numba.core.errors import NumbaPerformanceWarning

config.CUDA_ENABLE_PYNVJITLINK = 1
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from bench_utils import compute_wavefront


def _validate_threads(threads_per_block):
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
    max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    if threads_per_block > max_threads:
        raise ValueError(f"threads_per_block must be <= {max_threads}")


def _to_device_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr
    return cuda.to_device(np.asarray(arr, dtype=np.int32))


def coarsen_wavefronts(wavefront_list, coarsening_factor):
    """Coarsen wavefronts by merging consecutive dependency levels."""
    if coarsening_factor <= 1:
        return wavefront_list
    
    coarsened = []
    for i in range(0, len(wavefront_list), coarsening_factor):
        end_idx = min(i + coarsening_factor, len(wavefront_list))
        merged_wave = np.concatenate(wavefront_list[i:end_idx])
        coarsened.append(merged_wave)
    
    return coarsened


def compute_coarsened_wavefronts(L, coarsening_factor):
    """Compute wavefronts and apply dependency level coarsening."""
    wavefront_list = compute_wavefront(L)
    return coarsen_wavefronts(wavefront_list, coarsening_factor)


def compact_2d_list(wavefront_list):
    """Convert list of wavefronts to flat array with offset array."""
    waves = [element for sublist in wavefront_list for element in sublist]
    waves = np.array(waves, dtype=np.int32)
    wave_start_loc = np.zeros(len(wavefront_list) + 1, dtype=np.int32)
    wave_start_loc[0] = 0
    for i in range(1, len(wavefront_list)+1):
        wave_start_loc[i] = wave_start_loc[i - 1] + len(wavefront_list[i - 1])
    return wave_start_loc, waves


def compact_coarsened_wavefronts(original_wavefront_list, coarsening_factor):
    """Compact coarsened wavefronts with level boundary information."""
    if coarsening_factor <= 1:
        wave_start_loc, waves = compact_2d_list(original_wavefront_list)
        level_boundaries = [np.array([0, len(wf)], dtype=np.int32) for wf in original_wavefront_list]
        return wave_start_loc, waves, level_boundaries
    
    coarsened_waves = []
    level_boundaries = []
    
    for i in range(0, len(original_wavefront_list), coarsening_factor):
        end_idx = min(i + coarsening_factor, len(original_wavefront_list))
        levels_to_merge = original_wavefront_list[i:end_idx]
        merged_wave = np.concatenate(levels_to_merge)
        coarsened_waves.append(merged_wave)
        
        level_offsets = np.zeros(len(levels_to_merge) + 1, dtype=np.int32)
        level_offsets[0] = 0
        for j in range(len(levels_to_merge)):
            level_offsets[j + 1] = level_offsets[j] + len(levels_to_merge[j])
        level_boundaries.append(level_offsets)
    
    wave_start_loc, waves = compact_2d_list(coarsened_waves)
    
    return wave_start_loc, waves, level_boundaries


# CSC wavefront combined coarsening kernel for upper triangular matrices.
@cuda.jit
def sptrsv_numba_csc_wavefront_coarsening_single_level_kernel(
    data, indices, indptr, b, y, M, wavefronts, level_offset, level_size, 
    warps_per_block, thread_coarsening_factor
):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    warp_id = tid // 32
    lane_id = tid % 32
    
    col_idx = block_id * warps_per_block + warp_id
    
    if col_idx >= level_size:
        return
    
    col = wavefronts[level_offset + col_idx]
    if col >= M:
        return
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    if lane_id == 0:
        diag = data[col_end - 1]
        y[col] = (b[col] - y[col]) / diag
    
    y_col = y[col]
    
    stride = 32 * thread_coarsening_factor
    for chunk_start in range(col_start + lane_id * thread_coarsening_factor, col_end - 1, stride):
        for local_j in range(thread_coarsening_factor):
            j = chunk_start + local_j
            if j >= col_end - 1:
                break
            row_idx = indices[j]
            val = data[j]
            cuda.atomic.add(y, row_idx, val * y_col)


# CSR wavefront combined coarsening kernel for upper triangular matrices.
@cuda.jit
def sptrsv_numba_csr_wavefront_coarsening_single_level_kernel(
    data, indices, indptr, b, y, M, wavefronts, level_offset, level_size, 
    warps_per_block, thread_coarsening_factor
):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    warp_id = tid // 32
    lane_id = tid % 32
    
    row_idx = block_id * warps_per_block + warp_id
    
    if row_idx >= level_size:
        return
    
    row = wavefronts[level_offset + row_idx]
    if row >= M:
        return
    
    row_start = indptr[row]
    row_end = indptr[row + 1]
    
    partial_sum = 0.0
    stride = 32 * thread_coarsening_factor
    for chunk_start in range(row_start + 1 + lane_id * thread_coarsening_factor, row_end, stride):
        for local_j in range(thread_coarsening_factor):
            j = chunk_start + local_j
            if j >= row_end:
                break
            partial_sum += data[j] * y[indices[j]]
    
    for offset in [16, 8, 4, 2, 1]:
        partial_sum += cuda.shfl_down_sync(0xffffffff, partial_sum, offset)
    
    if lane_id == 0:
        diag = data[row_start]
        y[row] = (b[row] - partial_sum) / diag


# CSC wavefront combined coarsening for upper triangular matrices.
def sptrsv_numba_csc_wavefront_block_coarsening(
    data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, level_boundaries,
    block_coarsening_factor=1, thread_coarsening_factor=1, threads_per_block=128
):
    _validate_threads(threads_per_block)
    
    if threads_per_block % 32 != 0:
        threads_per_block = ((threads_per_block + 31) // 32) * 32
    
    warps_per_block = threads_per_block // 32
    
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wavefronts = _to_device_int32(wavefronts)
    num_waves = len(wave_start) - 1
    
    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0:
            continue
        
        level_offsets = level_boundaries[i]
        num_levels = len(level_offsets) - 1
        
        # Process each original level separately to enforce dependencies
        for level_idx in range(num_levels):
            level_start = int(level_offsets[level_idx])
            level_end = int(level_offsets[level_idx + 1])
            level_size = level_end - level_start
            
            if level_size <= 0:
                continue
            
            blocks_per_grid = (level_size + warps_per_block - 1) // warps_per_block
            sptrsv_numba_csc_wavefront_coarsening_single_level_kernel[blocks_per_grid, threads_per_block](
                data, indices, indptr, b, d_y, M, d_wavefronts, 
                np.int32(offset + level_start), np.int32(level_size),
                np.int32(warps_per_block), np.int32(thread_coarsening_factor)
            )
            cuda.synchronize()  # Grid-level barrier between dependency levels
    
    return d_y


# CSC wavefront thread coarsening wrapper.
def sptrsv_numba_csc_wavefront_thread_coarsening(
    data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, 
    thread_coarsening_factor=1, threads_per_block=128
):
    # Generate level_boundaries for single-level wavefronts
    wave_start_arr = np.asarray(wave_start, dtype=np.int32)
    num_waves = len(wave_start_arr) - 1
    level_boundaries = [np.array([0, wave_start_arr[i+1] - wave_start_arr[i]], dtype=np.int32) 
                        for i in range(num_waves)]
    
    return sptrsv_numba_csc_wavefront_block_coarsening(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, level_boundaries,
        block_coarsening_factor=1, thread_coarsening_factor=thread_coarsening_factor, 
        threads_per_block=threads_per_block
    )


# CSR wavefront combined coarsening for upper triangular matrices.
def sptrsv_numba_csr_wavefront_block_coarsening(
    data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, level_boundaries,
    block_coarsening_factor=1, thread_coarsening_factor=1, threads_per_block=128
):
    _validate_threads(threads_per_block)
    
    # Ensure threads_per_block is a multiple of 32 (warp size)
    if threads_per_block % 32 != 0:
        threads_per_block = ((threads_per_block + 31) // 32) * 32
    
    # Calculate warps per block
    # Each warp handles one row, so warps_per_block = rows processed per block
    warps_per_block = threads_per_block // 32
    
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wavefronts = _to_device_int32(wavefronts)
    num_waves = len(wave_start) - 1
    
    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0:
            continue
        
        level_offsets = level_boundaries[i]
        num_levels = len(level_offsets) - 1
        
        for level_idx in range(num_levels):
            level_start = int(level_offsets[level_idx])
            level_end = int(level_offsets[level_idx + 1])
            level_size = level_end - level_start
            
            if level_size <= 0:
                continue
            
            blocks_per_grid = (level_size + warps_per_block - 1) // warps_per_block
            shared_mem_bytes = threads_per_block * 8
            sptrsv_numba_csr_wavefront_coarsening_single_level_kernel[blocks_per_grid, threads_per_block, 0, shared_mem_bytes](
                data, indices, indptr, b, d_y, M, d_wavefronts, 
                np.int32(offset + level_start), np.int32(level_size),
                np.int32(warps_per_block), np.int32(thread_coarsening_factor)
            )
            cuda.synchronize()  # Sync after each level to enforce dependencies
    
    return d_y


# CSR wavefront thread coarsening wrapper.
def sptrsv_numba_csr_wavefront_thread_coarsening(
    data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, 
    thread_coarsening_factor=1, threads_per_block=128
):
    # Generate level_boundaries for single-level wavefronts
    wave_start_arr = np.asarray(wave_start, dtype=np.int32)
    num_waves = len(wave_start_arr) - 1
    level_boundaries = [np.array([0, wave_start_arr[i+1] - wave_start_arr[i]], dtype=np.int32) 
                        for i in range(num_waves)]
    
    return sptrsv_numba_csr_wavefront_block_coarsening(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, level_boundaries,
        block_coarsening_factor=1, thread_coarsening_factor=thread_coarsening_factor, 
        threads_per_block=threads_per_block
    )


# CSC sync-free block coarsening kernel for upper triangular matrices.
@cuda.jit
def sptrsv_numba_csc_sync_free_block_coarsening_kernel(
    data, indices, indptr, b, y, M, dep_counts, block_coarsening_factor
):
    block_id = cuda.blockIdx.x
    base_col = block_id * block_coarsening_factor
    
    # Upper triangular: process columns from right to left within block
    # E.g., if bcf=2 and base_col=0, process columns 1, 0 (higher first)
    for local_idx in range(block_coarsening_factor - 1, -1, -1):
        col = base_col + local_idx
        if col >= M:
            continue
        
        # Wait until all dependencies are satisfied
        while cuda.atomic.add(dep_counts, col, 0) > 0:
            pass

        cuda.threadfence()
        
        col_start = indptr[col]
        col_end = indptr[col + 1]
        
        # Upper CSC: Diagonal at col_end - 1
        y[col] = (b[col] - y[col]) / data[col_end - 1]
        
        # Scatter: start to col_end - 1
        for j in range(col_start, col_end - 1):
            row_idx = indices[j]
            val = data[j]
            cuda.atomic.add(y, row_idx, val * y[col])
        
        cuda.threadfence()
        # Decrement dependencies for upstream rows
        for j in range(col_start, col_end - 1):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


# CSC sync-free thread coarsening kernel for upper triangular matrices.
@cuda.jit
def sptrsv_numba_csc_sync_free_thread_coarsening_kernel(
    data, indices, indptr, b, y, M, dep_counts, thread_coarsening_factor
):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if block_id >= M:
        return
    
    col = block_id
    
    # Thread 0 checks dependency
    if tid == 0:
        while cuda.atomic.add(dep_counts, col, 0) > 0:
            pass
    cuda.syncthreads()

    cuda.threadfence()
    
    col_start = indptr[col]
    col_end = indptr[col + 1]
    
    # Upper CSC: Diagonal at col_end - 1
    if tid == 0:
        y[col] = (b[col] - y[col]) / data[col_end - 1]
    cuda.syncthreads()
    
    y_col = y[col]
    # Use strided loop to handle columns with more elements than block_size*thread_coarsening_factor
    # Scatter: start to col_end - 1 (exclude diagonal)
    stride = block_size * thread_coarsening_factor
    for chunk_start in range(col_start + tid * thread_coarsening_factor, col_end - 1, stride):
        for local_j in range(thread_coarsening_factor):
            j = chunk_start + local_j
            if j >= col_end - 1:
                break
            row_idx = indices[j]
            val = data[j]
            cuda.atomic.add(y, row_idx, val * y_col)

    cuda.syncthreads()
    if tid == 0:
        cuda.threadfence()
        for j in range(col_start, col_end - 1):
            row_idx = indices[j]
            cuda.atomic.add(dep_counts, row_idx, -1)


# CSR sync-free block coarsening kernel for upper triangular matrices.
@cuda.jit
def sptrsv_numba_csr_sync_free_block_coarsening_kernel(
    data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices, block_coarsening_factor
):
    block_id = cuda.blockIdx.x
    base_row = block_id * block_coarsening_factor
    
    # Upper triangular: process rows from right to left within block
    # Row n-1 processed first, row 0 last
    for local_idx in range(block_coarsening_factor - 1, -1, -1):
        row = base_row + local_idx
        if row >= M:
            continue
        
        while cuda.atomic.add(dep_counts, row, 0) > 0:
            pass
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        y_val = b[row]
        # Upper CSR: Diagonal at start. Off-diagonals start+1 to end
        for j in range(row_start + 1, row_end):
            y_val -= data[j] * y[indices[j]]
        
        y[row] = y_val / data[row_start]
        cuda.threadfence()
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            succ_row = succ_indices[p]
            cuda.atomic.add(dep_counts, succ_row, -1)


# CSR sync-free thread coarsening kernel for upper triangular matrices.
@cuda.jit
def sptrsv_numba_csr_sync_free_thread_coarsening_kernel(
    data, indices, indptr, b, y, M, dep_counts, succ_offsets, succ_indices, thread_coarsening_factor
):
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
    
    # Use dynamic shared memory allocation based on actual block size
    shared_sum = cuda.shared.array(shape=0, dtype=numba.float64)
    partial_sum = 0.0
    # Use strided loop to handle rows with more elements than block_size*thread_coarsening_factor
    # Upper CSR: Start from row_start + 1
    stride = block_size * thread_coarsening_factor
    for chunk_start in range(row_start + 1 + tid * thread_coarsening_factor, row_end, stride):
        for local_j in range(thread_coarsening_factor):
            j = chunk_start + local_j
            if j >= row_end:
                break
            partial_sum += data[j] * y[indices[j]]
    
    shared_sum[tid] = partial_sum
    cuda.syncthreads()

    # Reduction
    s = block_size // 2
    while s > 0:
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        diag = data[row_start]
        y[row] = (b[row] - shared_sum[0]) / diag
        cuda.threadfence()
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            succ_row = succ_indices[p]
            cuda.atomic.add(dep_counts, succ_row, -1)


# CSC Sync-Free Block Coarsening
def sptrsv_numba_csc_sync_free_block_coarsening(
    data, indices, indptr, b, M, N, nnz, in_edges, block_coarsening_factor=2
):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    
    blocks_per_grid = (M + block_coarsening_factor - 1) // block_coarsening_factor
    sptrsv_numba_csc_sync_free_block_coarsening_kernel[blocks_per_grid, 1](
        data, indices, indptr, b, d_y, M, d_dep_counts, np.int32(block_coarsening_factor)
    )
    
    return d_y


# CSC Sync-Free Thread Coarsening
def sptrsv_numba_csc_sync_free_thread_coarsening(
    data, indices, indptr, b, M, N, nnz, in_edges, 
    thread_coarsening_factor=2, threads_per_block=128
):
    _validate_threads(threads_per_block)
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    
    blocks_per_grid = M
    sptrsv_numba_csc_sync_free_thread_coarsening_kernel[blocks_per_grid, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep_counts, np.int32(thread_coarsening_factor)
    )
    
    return d_y


# CSR Sync-Free Block Coarsening
def sptrsv_numba_csr_sync_free_block_coarsening(
    data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, block_coarsening_factor=2
):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    d_succ_offsets = cuda.to_device(succ_offsets)
    d_succ_indices = cuda.to_device(succ_indices)
    
    blocks_per_grid = (M + block_coarsening_factor - 1) // block_coarsening_factor
    sptrsv_numba_csr_sync_free_block_coarsening_kernel[blocks_per_grid, 1](
        data, indices, indptr, b, d_y, M, d_dep_counts, d_succ_offsets, d_succ_indices, 
        np.int32(block_coarsening_factor)
    )
    
    return d_y


# CSR Sync-Free Thread Coarsening
def sptrsv_numba_csr_sync_free_thread_coarsening(
    data, indices, indptr, b, M, N, nnz, in_edges, succ_offsets, succ_indices, 
    thread_coarsening_factor=2, threads_per_block=128
):
    _validate_threads(threads_per_block)
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep_counts = cuda.to_device(in_edges.copy())
    d_succ_offsets = cuda.to_device(succ_offsets)
    d_succ_indices = cuda.to_device(succ_indices)
    
    blocks_per_grid = M
    shared_mem_bytes = threads_per_block * 8
    sptrsv_numba_csr_sync_free_thread_coarsening_kernel[blocks_per_grid, threads_per_block, 0, shared_mem_bytes](
        data, indices, indptr, b, d_y, M, d_dep_counts, d_succ_offsets, d_succ_indices, 
        np.int32(thread_coarsening_factor)
    )
    
    return d_y


def sptrsv_numba_csc_coarsened_mapping(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor, threads_per_block, thread_coarsening_factor=1, level_boundaries=None):
    # Combined 3-level coarsening: wavefront + warp + thread
    wave_start_arr = np.asarray(wave_start, dtype=np.int32)
    num_waves = len(wave_start_arr) - 1
    
    if level_boundaries is None:
        if coarsening_factor > 1:
            raise ValueError(
                f"level_boundaries is required when coarsening_factor > 1 (got {coarsening_factor}). "
                "Use compact_coarsened_wavefronts() to get proper level boundaries."
            )
        level_boundaries = [np.array([0, wave_start_arr[i+1] - wave_start_arr[i]], dtype=np.int32) 
                            for i in range(num_waves)]
    
    return sptrsv_numba_csc_wavefront_block_coarsening(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, level_boundaries,
        block_coarsening_factor=1, thread_coarsening_factor=thread_coarsening_factor, threads_per_block=threads_per_block
    )


def sptrsv_numba_csr_coarsened_mapping(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor, threads_per_block, thread_coarsening_factor=1, level_boundaries=None):
    # Combined 3-level coarsening: wavefront + warp + thread
    wave_start_arr = np.asarray(wave_start, dtype=np.int32)
    num_waves = len(wave_start_arr) - 1
    
    if level_boundaries is None:
        if coarsening_factor > 1:
            raise ValueError(
                f"level_boundaries is required when coarsening_factor > 1 (got {coarsening_factor}). "
                "Use compact_coarsened_wavefronts() to get proper level boundaries."
            )
        level_boundaries = [np.array([0, wave_start_arr[i+1] - wave_start_arr[i]], dtype=np.int32) 
                            for i in range(num_waves)]
    
    return sptrsv_numba_csr_wavefront_block_coarsening(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, level_boundaries,
        block_coarsening_factor=1, thread_coarsening_factor=thread_coarsening_factor, threads_per_block=threads_per_block
    )