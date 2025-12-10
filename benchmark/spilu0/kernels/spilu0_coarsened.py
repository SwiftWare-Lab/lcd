"""SpILU0 with Coarsening: Register-Tiled, Block-Tiled, and All-Tiled Strategies"""

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

# ----------------------------------------------------------------------------
# Utilities & Device Functions
# ----------------------------------------------------------------------------

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

@cuda.jit(device=True)
def find_index_in_row(indices, row_start, row_end, target_col):
    """Linear search to find a column index in a row (essential for ILU0 update)."""
    for k in range(row_start, row_end):
        if indices[k] == target_col:
            return k
    return -1

# ----------------------------------------------------------------------------
# 1. WAVEFRONT KERNELS (Combined Logic)
# ----------------------------------------------------------------------------

# This unified kernel handles Register-Tiled, Block-Tiled, and All-Tiled
# by parameterizing warps_per_block (Block-Tiling) and stride (Register-Tiling).
@cuda.jit
def spilu0_numba_csr_wavefront_tiled_kernel(
    data, indices, indptr, diag_map, M, wavefronts, level_offset, level_size, 
    warps_per_block, register_tiling_factor
):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    warp_id = tid // 32
    lane_id = tid % 32
    
    # Block-Tiling Logic:
    # If warps_per_block > 1, the block processes multiple rows (one per warp).
    # If warps_per_block == 1, the whole block processes one row (or batch if looped).
    # Here we map 1 Warp -> 1 Row.
    row_idx_in_level = block_id * warps_per_block + warp_id
    
    if row_idx_in_level >= level_size:
        return
    
    row = wavefronts[level_offset + row_idx_in_level]
    if row >= M:
        return
    
    row_start = indptr[row]
    row_end = indptr[row + 1]
    diag_ptr = diag_map[row]
    
    # --- ILU0 Row Processing ---
    # Iterate over the L-part (columns k < row)
    for k in range(row_start, diag_ptr):
        col_k = indices[k]
        
        # 1. Compute Multiplier
        diag_k_ptr = diag_map[col_k]
        diag_k_val = data[diag_k_ptr]
        multiplier = data[k] / diag_k_val
        
        # Thread 0 writes the multiplier
        if lane_id == 0:
            data[k] = multiplier
        
        # Sync warp to ensure multiplier is visible (implicit in some archs, explicit is safer)
        # cuda.syncwarp() # Numba < 0.54 might not have syncwarp exposed easily, rely on nature of warp execution or mem fence
        
        # 2. Update Row (The hot loop)
        # Row_i[j] -= multiplier * Row_k[j]
        row_k_start = diag_k_ptr + 1
        row_k_end = indptr[col_k + 1]
        
        # Register-Tiling Logic:
        # Strided loop allows one thread to handle multiple elements, increasing ILP.
        stride = 32 * register_tiling_factor
        
        for chunk_start in range(row_k_start + lane_id * register_tiling_factor, row_k_end, stride):
            for local_j in range(register_tiling_factor):
                idx_in_k = chunk_start + local_j
                if idx_in_k >= row_k_end:
                    break
                
                col_j = indices[idx_in_k]
                val_k = data[idx_in_k]
                
                # Find matching column in Row i to update
                # Optimization: We search in Row i starting from k+1
                match_idx = find_index_in_row(indices, k + 1, row_end, col_j)
                
                if match_idx != -1:
                    cuda.atomic.add(data, match_idx, -multiplier * val_k)

# ----------------------------------------------------------------------------
# 2. WAVEFRONT HOST FUNCTIONS
# ----------------------------------------------------------------------------

def _spilu0_wavefront_launcher(
    data, indices, indptr, diag_map, M, wave_start, wavefronts, level_boundaries,
    block_tiling_factor=1, register_tiling_factor=1, threads_per_block=128
):
    _validate_threads(threads_per_block)
    
    # Ensure threads_per_block is multiple of 32
    if threads_per_block % 32 != 0:
        threads_per_block = ((threads_per_block + 31) // 32) * 32
    
    # Block-Tiling Mapping:
    # If block_tiling_factor > 1, we pack more warps into a block to handle multiple rows.
    # warps_per_block effectively acts as the "rows per block"
    warps_per_block = threads_per_block // 32
    
    d_data = cuda.to_device(data)
    d_wavefronts = _to_device_int32(wavefronts)
    d_diag_map = _to_device_int32(diag_map)
    
    num_waves = len(wave_start) - 1
    
    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0: continue
        
        level_offsets = level_boundaries[i]
        num_levels = len(level_offsets) - 1
        
        for level_idx in range(num_levels):
            level_start = int(level_offsets[level_idx])
            level_end = int(level_offsets[level_idx + 1])
            level_size = level_end - level_start
            
            if level_size <= 0: continue
            
            blocks_per_grid = (level_size + warps_per_block - 1) // warps_per_block
            
            spilu0_numba_csr_wavefront_tiled_kernel[blocks_per_grid, threads_per_block](
                d_data, indices, indptr, d_diag_map, M, d_wavefronts, 
                np.int32(offset + level_start), np.int32(level_size),
                np.int32(warps_per_block), np.int32(register_tiling_factor)
            )
            cuda.synchronize()
            
    return d_data

def spilu0_numba_csr_wavefront_register_tiled(
    data, indices, indptr, diag_map, M, nnz, wave_start, wavefronts, level_boundaries,
    register_tiling_factor=2, threads_per_block=32
):
    """Wavefront Register-Tiled: 1 Warp/Row, Strided thread execution."""
    # Force block_tiling_factor effectively to 1 by using 1 warp per block or just standard mapping
    return _spilu0_wavefront_launcher(
        data, indices, indptr, diag_map, M, wave_start, wavefronts, level_boundaries,
        block_tiling_factor=1, register_tiling_factor=register_tiling_factor, 
        threads_per_block=32 # 1 warp per block implies 1 row per block
    )

def spilu0_numba_csr_wavefront_block_tiled(
    data, indices, indptr, diag_map, M, nnz, wave_start, wavefronts, level_boundaries,
    block_tiling_factor=4, threads_per_block=128
):
    """Wavefront Block-Tiled: Multiple Warps/Block (multiple rows), Standard stride."""
    # We increase threads_per_block to allow multiple warps (rows) per block
    # block_tiling_factor here is implicitly set by threads_per_block / 32
    return _spilu0_wavefront_launcher(
        data, indices, indptr, diag_map, M, wave_start, wavefronts, level_boundaries,
        block_tiling_factor=block_tiling_factor, register_tiling_factor=1, 
        threads_per_block=32*block_tiling_factor
    )

def spilu0_numba_csr_wavefront_all_tiled(
    data, indices, indptr, diag_map, M, nnz, wave_start, wavefronts, level_boundaries,
    block_tiling_factor=4, register_tiling_factor=2, threads_per_block=128
):
    """Wavefront All-Tiled: Multiple Warps/Block AND Strided thread execution."""
    return _spilu0_wavefront_launcher(
        data, indices, indptr, diag_map, M, wave_start, wavefronts, level_boundaries,
        block_tiling_factor=block_tiling_factor, register_tiling_factor=register_tiling_factor, 
        threads_per_block=32*block_tiling_factor
    )

# ----------------------------------------------------------------------------
# 3. SYNC-FREE KERNELS
# ----------------------------------------------------------------------------

# Sync-Free Block-Tiled: One Block processes a "Batch" of rows sequentially
@cuda.jit
def spilu0_numba_csr_sync_free_block_tiled_kernel(
    data, indices, indptr, diag_map, dep_counts, succ_offsets, succ_indices, M, 
    block_tiling_factor
):
    block_id = cuda.blockIdx.x
    base_row = block_id * block_tiling_factor
    
    # Single thread per block logic (simplified for heavy ILU workloads)
    # or we can use the whole block to process the row, but Block-Tiling 
    # implies the block *loop* is coarsened.
    
    # We use 1 thread to drive the logic to strictly match "Block Coarsening" 
    # semantics where 1 execution entity handles N items.
    tid = cuda.threadIdx.x
    if tid > 0: return

    for local_idx in range(block_tiling_factor):
        row = base_row + local_idx
        if row >= M: break
        
        # 1. Wait for dependencies
        while cuda.atomic.add(dep_counts, row, 0) > 0:
            pass
        cuda.threadfence()
        
        # 2. Process Row (Serial implementation for single thread)
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_ptr = diag_map[row]
        
        for k in range(row_start, diag_ptr):
            col_k = indices[k]
            diag_k_ptr = diag_map[col_k]
            
            multiplier = data[k] / data[diag_k_ptr]
            data[k] = multiplier
            
            # Update part
            row_k_start = diag_k_ptr + 1
            row_k_end = indptr[col_k + 1]
            
            for p in range(row_k_start, row_k_end):
                col_j = indices[p]
                val_k = data[p]
                
                # Linear search in current row
                for q in range(k + 1, row_end):
                    if indices[q] == col_j:
                        data[q] -= multiplier * val_k
                        break
        
        cuda.threadfence()
        
        # 3. Release dependencies
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            succ_row = succ_indices[p]
            cuda.atomic.add(dep_counts, succ_row, -1)


# Sync-Free Register-Tiled: One Block processes ONE row, but threads stride (Register Tile)
@cuda.jit
def spilu0_numba_csr_sync_free_register_tiled_kernel(
    data, indices, indptr, diag_map, dep_counts, succ_offsets, succ_indices, M, 
    register_tiling_factor
):
    block_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    row = block_id
    if row >= M: return
    
    # 1. Thread 0 handles dependency waiting
    if tid == 0:
        while cuda.atomic.add(dep_counts, row, 0) > 0:
            pass
    cuda.syncthreads()
    cuda.threadfence()
    
    row_start = indptr[row]
    row_end = indptr[row + 1]
    diag_ptr = diag_map[row]
    
    # 2. Parallel Row Processing
    for k in range(row_start, diag_ptr):
        col_k = indices[k]
        
        # Compute multiplier (Thread 0)
        if tid == 0:
            diag_k_ptr = diag_map[col_k]
            data[k] = data[k] / data[diag_k_ptr]
            
        cuda.syncthreads()
        multiplier = data[k]
        
        # Update Row i using Row k
        diag_k_ptr = diag_map[col_k] # Re-read for threads
        row_k_start = diag_k_ptr + 1
        row_k_end = indptr[col_k + 1]
        
        # --- Register Tiling (Stride) ---
        stride = block_size * register_tiling_factor
        
        for chunk_start in range(row_k_start + tid * register_tiling_factor, row_k_end, stride):
            for local_j in range(register_tiling_factor):
                idx_in_k = chunk_start + local_j
                if idx_in_k >= row_k_end: break
                
                col_j = indices[idx_in_k]
                val_k = data[idx_in_k]
                
                # Find index in row i
                match_idx = find_index_in_row(indices, k + 1, row_end, col_j)
                if match_idx != -1:
                    cuda.atomic.add(data, match_idx, -multiplier * val_k)
                    
        cuda.syncthreads()
        
    # 3. Release Dependencies
    if tid == 0:
        cuda.threadfence()
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            succ_row = succ_indices[p]
            cuda.atomic.add(dep_counts, succ_row, -1)

# ----------------------------------------------------------------------------
# 4. SYNC-FREE HOST FUNCTIONS
# ----------------------------------------------------------------------------

def spilu0_numba_csr_sync_free_block_tiled(
    data, indices, indptr, diag_map, M, in_edges, succ_offsets, succ_indices, 
    block_tiling_factor=2
):
    d_data = cuda.to_device(data)
    d_dep_counts = cuda.to_device(in_edges.copy())
    d_succ_offsets = cuda.to_device(succ_offsets)
    d_succ_indices = cuda.to_device(succ_indices)
    d_diag_map = cuda.to_device(diag_map)
    
    # M rows, coarsened by factor -> Grid Size
    blocks_per_grid = (M + block_tiling_factor - 1) // block_tiling_factor
    
    spilu0_numba_csr_sync_free_block_tiled_kernel[blocks_per_grid, 1](
        d_data, indices, indptr, d_diag_map, d_dep_counts, d_succ_offsets, d_succ_indices, 
        M, np.int32(block_tiling_factor)
    )
    return d_data

def spilu0_numba_csr_sync_free_register_tiled(
    data, indices, indptr, diag_map, M, in_edges, succ_offsets, succ_indices, 
    register_tiling_factor=2, threads_per_block=128
):
    _validate_threads(threads_per_block)
    
    d_data = cuda.to_device(data)
    d_dep_counts = cuda.to_device(in_edges.copy())
    d_succ_offsets = cuda.to_device(succ_offsets)
    d_succ_indices = cuda.to_device(succ_indices)
    d_diag_map = cuda.to_device(diag_map)
    
    # 1 Block per Row
    blocks_per_grid = M
    
    spilu0_numba_csr_sync_free_register_tiled_kernel[blocks_per_grid, threads_per_block](
        d_data, indices, indptr, d_diag_map, d_dep_counts, d_succ_offsets, d_succ_indices, 
        M, np.int32(register_tiling_factor)
    )
    return d_data