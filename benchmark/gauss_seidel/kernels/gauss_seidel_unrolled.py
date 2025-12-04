import numpy as np
from numba import cuda

# ============================================================================
# Wavefront Mapping (Multiple Threads per Row)
# ============================================================================

@cuda.jit
def gs_wavefront_mapping_kernel(data, indices, indptr, b, x, 
                                 wavefronts, wave_start, wave_idx):
    """
    Wavefront with thread-level parallelism within each row.
    Uses shared memory for reduction.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if bid < num_rows:
        row = wavefronts[start_wave + bid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        # Shared memory for reduction
        shared_sigma = cuda.shared.array(1, dtype=np.float64)
        shared_diag = cuda.shared.array(1, dtype=np.float64)
        
        # Initialize shared memory
        if tid == 0:
            shared_sigma[0] = 0.0
            shared_diag[0] = 1.0
        cuda.syncthreads()
        
        # Each thread processes subset of row elements
        local_sigma = 0.0
        local_diag = 1.0
        
        for i in range(row_start + tid, row_end, cuda.blockDim.x):
            col = indices[i]
            val = data[i]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
        
        # Reduce into shared memory (atomic for simplicity)
        if local_sigma != 0.0:
            cuda.atomic.add(shared_sigma, 0, local_sigma)
        if local_diag != 1.0:
            shared_diag[0] = local_diag
        
        cuda.syncthreads()
        
        # Thread 0 updates x
        if tid == 0:
            sigma = shared_sigma[0]
            diag = shared_diag[0]
            x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_mapping(data, indices, indptr, b, x, wave_start, 
                                wavefronts, max_iter=10, threads_per_block=128):
    """
    Driver for wavefront mapping.
    """
    num_waves = len(wave_start) - 1
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                gs_wavefront_mapping_kernel[grid_len, threads_per_block](
                    data, indices, indptr, b, x, wavefronts, wave_start, i
                )


# ============================================================================
# Wavefront Unrolling (Factor 4)
# ============================================================================

@cuda.jit
def gs_wavefront_unrolling_kernel_4(data, indices, indptr, b, x, 
                                     wavefronts, wave_start, wave_idx):
    """
    Wavefront with explicit unrolling (factor 4).
    Each thread processes multiple elements with unrolled inner loop.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if bid < num_rows:
        row = wavefronts[start_wave + bid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        shared_sigma = cuda.shared.array(1, dtype=np.float64)
        shared_diag = cuda.shared.array(1, dtype=np.float64)
        
        if tid == 0:
            shared_sigma[0] = 0.0
            shared_diag[0] = 1.0
        cuda.syncthreads()
        
        local_sigma = 0.0
        local_diag = 1.0
        
        # Unrolled loop: process 4 elements at a time
        i = row_start + tid
        while i + 3 * cuda.blockDim.x < row_end:
            # Unroll 4 iterations
            for unroll_idx in range(4):
                pos = i + unroll_idx * cuda.blockDim.x
                col = indices[pos]
                val = data[pos]
                
                if col == row:
                    local_diag = val
                else:
                    local_sigma += val * x[col]
            
            i += 4 * cuda.blockDim.x
        
        # Remainder
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += cuda.blockDim.x
        
        if local_sigma != 0.0:
            cuda.atomic.add(shared_sigma, 0, local_sigma)
        if local_diag != 1.0:
            shared_diag[0] = local_diag
        
        cuda.syncthreads()
        
        if tid == 0:
            sigma = shared_sigma[0]
            diag = shared_diag[0]
            x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_unrolling_4(data, indices, indptr, b, x, wave_start, 
                                    wavefronts, max_iter=10, threads_per_block=128):
    num_waves = len(wave_start) - 1
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                gs_wavefront_unrolling_kernel_4[grid_len, threads_per_block](
                    data, indices, indptr, b, x, wavefronts, wave_start, i
                )


# ============================================================================
# Wavefront Unrolling (Factor 8)
# ============================================================================

@cuda.jit
def gs_wavefront_unrolling_kernel_8(data, indices, indptr, b, x, 
                                     wavefronts, wave_start, wave_idx):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if bid < num_rows:
        row = wavefronts[start_wave + bid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        shared_sigma = cuda.shared.array(1, dtype=np.float64)
        shared_diag = cuda.shared.array(1, dtype=np.float64)
        
        if tid == 0:
            shared_sigma[0] = 0.0
            shared_diag[0] = 1.0
        cuda.syncthreads()
        
        local_sigma = 0.0
        local_diag = 1.0
        
        i = row_start + tid
        while i + 7 * cuda.blockDim.x < row_end:
            for unroll_idx in range(8):
                pos = i + unroll_idx * cuda.blockDim.x
                col = indices[pos]
                val = data[pos]
                
                if col == row:
                    local_diag = val
                else:
                    local_sigma += val * x[col]
            
            i += 8 * cuda.blockDim.x
        
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += cuda.blockDim.x
        
        if local_sigma != 0.0:
            cuda.atomic.add(shared_sigma, 0, local_sigma)
        if local_diag != 1.0:
            shared_diag[0] = local_diag
        
        cuda.syncthreads()
        
        if tid == 0:
            sigma = shared_sigma[0]
            diag = shared_diag[0]
            x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_unrolling_8(data, indices, indptr, b, x, wave_start, 
                                    wavefronts, max_iter=10, threads_per_block=128):
    num_waves = len(wave_start) - 1
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                gs_wavefront_unrolling_kernel_8[grid_len, threads_per_block](
                    data, indices, indptr, b, x, wavefronts, wave_start, i
                )


# ============================================================================
# Sync-Free Unrolling (Factor 4)
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_4(data, indices, indptr, 
                                    csc_indices_L, csc_indptr_L,
                                    incom_edges, b, x, num_rows):
    """
    Sync-Free GS with unrolling (factor 4).
    1 Warp per Row with explicit loop unrolling.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(global_warp_id, num_rows, total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        sigma = 0.0
        diag = 1.0
        
        # Unrolled parallel reduction
        i = row_start + lane_id
        while i + 3 * 32 < row_end:
            for unroll_idx in range(4):
                pos = i + unroll_idx * 32
                col = indices[pos]
                val = data[pos]
                
                if col == row:
                    diag = val
                else:
                    sigma += val * x[col]
            
            i += 4 * 32
        
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 32
        
        # Warp Shuffle Reduction
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        # Reduce diag
        temp_diag = diag
        for offset in [16, 8, 4, 2, 1]:
            other_diag = cuda.shfl_down_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        temp_diag = diag
        for offset in [1, 2, 4, 8, 16]:
            other_diag = cuda.shfl_up_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        if lane_id == 0 and diag == 1.0:
            for i in range(row_start, row_end):
                if indices[i] == row:
                    diag = data[i]
                    break
        diag = cuda.shfl_sync(mask, diag, 0)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_L[row]
        sig_end = csc_indptr_L[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_L[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


@cuda.jit
def copy_device_array_kernel(src, dst, n):
    tid = cuda.grid(1)
    if tid < n:
        dst[tid] = src[tid]


def gs_numba_syncfree_unrolling_4(data, indices, indptr, csc_indices_L, csc_indptr_L, 
                                   base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    """
    Driver for sync-free with unrolling factor 4.
    threads_per_block: Number of threads per block (must be multiple of 32).
    """
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_4[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


# ============================================================================
# Sync-Free Unrolling (Factor 8)
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_8(data, indices, indptr, 
                                    csc_indices_L, csc_indptr_L,
                                    incom_edges, b, x, num_rows):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(global_warp_id, num_rows, total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        sigma = 0.0
        diag = 1.0
        
        # Unrolled parallel reduction (factor 8)
        i = row_start + lane_id
        while i + 7 * 32 < row_end:
            for unroll_idx in range(8):
                pos = i + unroll_idx * 32
                col = indices[pos]
                val = data[pos]
                
                if col == row:
                    diag = val
                else:
                    sigma += val * x[col]
            
            i += 8 * 32
        
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 32
        
        # Warp Shuffle Reduction
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        # Reduce diag
        temp_diag = diag
        for offset in [16, 8, 4, 2, 1]:
            other_diag = cuda.shfl_down_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        temp_diag = diag
        for offset in [1, 2, 4, 8, 16]:
            other_diag = cuda.shfl_up_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        if lane_id == 0 and diag == 1.0:
            for i in range(row_start, row_end):
                if indices[i] == row:
                    diag = data[i]
                    break
        diag = cuda.shfl_sync(mask, diag, 0)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_L[row]
        sig_end = csc_indptr_L[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_L[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_8(data, indices, indptr, csc_indices_L, csc_indptr_L, 
                                   base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    """Driver for sync-free with unrolling factor 8."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_8[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()

import numpy as np
from numba import cuda

# ============================================================================
# GENERATED UNROLLING KERNELS - Regular Variants
# ============================================================================
# Generated by codegen_gs_unrolling.py
# DO NOT EDIT MANUALLY - regenerate if needed
# ============================================================================


@cuda.jit
def gs_wavefront_unrolling_kernel_16(data, indices, indptr, b, x, 
                                     wavefronts, wave_start, wave_idx):
    """
    Wavefront with explicit unrolling (factor 16).
    Advanced implementation with 16x unrolling for maximum ILP.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if bid < num_rows:
        row = wavefronts[start_wave + bid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        
        shared_sigma = cuda.shared.array(1, dtype=np.float64)
        shared_diag = cuda.shared.array(1, dtype=np.float64)
        
        if tid == 0:
            shared_sigma[0] = 0.0
            shared_diag[0] = 1.0
        cuda.syncthreads()
        
        local_sigma = 0.0
        local_diag = 1.0
        
        # Main unrolled loop: process 16 elements at a time
        i = row_start + tid
        while i + 15 * cuda.blockDim.x < row_end:
            pos = i + 0 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 1 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 2 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 3 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 4 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 5 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 6 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 7 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 8 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 9 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 10 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 11 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 12 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 13 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 14 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 15 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += 16 * cuda.blockDim.x
        
        # Remainder loop
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += cuda.blockDim.x
        
        if local_sigma != 0.0:
            cuda.atomic.add(shared_sigma, 0, local_sigma)
        if local_diag != 1.0:
            shared_diag[0] = local_diag
        
        cuda.syncthreads()
        
        if tid == 0:
            sigma = shared_sigma[0]
            diag = shared_diag[0]
            x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_unrolling_16(data, indices, indptr, b, x, wave_start, 
                                    wavefronts, max_iter=10, threads_per_block=128):
    """Driver for wavefront unrolling factor 16."""
    num_waves = len(wave_start) - 1
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                gs_wavefront_unrolling_kernel_16[grid_len, threads_per_block](
                    data, indices, indptr, b, x, 
                    wavefronts, wave_start, i
                )



@cuda.jit
def gs_syncfree_unrolling_kernel_16(data, indices, indptr, 
                                         csc_indices_L, csc_indptr_L,
                                         incom_edges, b, x, num_rows):
    """
    Sync-Free GS with unrolling (factor 16).
    1 Warp per Row with 16x explicit loop unrolling for maximum throughput.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(global_warp_id, num_rows, total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        
        sigma = 0.0
        diag = 1.0
        
        # Unrolled parallel reduction (factor 16)
        i = row_start + lane_id
        while i + 15 * 32 < row_end:
            pos = i + 0 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 1 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 2 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 3 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 4 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 5 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 6 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 7 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 8 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 9 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 10 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 11 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 12 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 13 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 14 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 15 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 16 * 32
        
        # Remainder loop
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 32
        
        # Warp Shuffle Reduction
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
                # Reduce diag
        temp_diag = diag
        for offset in [16, 8, 4, 2, 1]:
            other_diag = cuda.shfl_down_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        temp_diag = diag
        for offset in [1, 2, 4, 8, 16]:
            other_diag = cuda.shfl_up_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        if lane_id == 0 and diag == 1.0:
            for i in range(row_start, row_end):
                if indices[i] == row:
                    diag = data[i]
                    break
        diag = cuda.shfl_sync(mask, diag, 0)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_L[row]
        sig_end = csc_indptr_L[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_L[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_16(data, indices, indptr, 
                                        csc_indices_L, csc_indptr_L, 
                                        base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    """Driver for sync-free unrolling factor 16."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_16[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()



@cuda.jit
def gs_wavefront_unrolling_kernel_32(data, indices, indptr, b, x, 
                                     wavefronts, wave_start, wave_idx):
    """
    Wavefront with explicit unrolling (factor 32).
    Advanced implementation with 32x unrolling for maximum ILP.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if bid < num_rows:
        row = wavefronts[start_wave + bid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        
        shared_sigma = cuda.shared.array(1, dtype=np.float64)
        shared_diag = cuda.shared.array(1, dtype=np.float64)
        
        if tid == 0:
            shared_sigma[0] = 0.0
            shared_diag[0] = 1.0
        cuda.syncthreads()
        
        local_sigma = 0.0
        local_diag = 1.0
        
        # Main unrolled loop: process 32 elements at a time
        i = row_start + tid
        while i + 31 * cuda.blockDim.x < row_end:
            pos = i + 0 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 1 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 2 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 3 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 4 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 5 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 6 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 7 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 8 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 9 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 10 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 11 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 12 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 13 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 14 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 15 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 16 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 17 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 18 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 19 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 20 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 21 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 22 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 23 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 24 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 25 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 26 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 27 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 28 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 29 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 30 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 31 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += 32 * cuda.blockDim.x
        
        # Remainder loop
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += cuda.blockDim.x
        
        if local_sigma != 0.0:
            cuda.atomic.add(shared_sigma, 0, local_sigma)
        if local_diag != 1.0:
            shared_diag[0] = local_diag
        
        cuda.syncthreads()
        
        if tid == 0:
            sigma = shared_sigma[0]
            diag = shared_diag[0]
            x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_unrolling_32(data, indices, indptr, b, x, wave_start, 
                                    wavefronts, max_iter=10, threads_per_block=128):
    """Driver for wavefront unrolling factor 32."""
    num_waves = len(wave_start) - 1
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                gs_wavefront_unrolling_kernel_32[grid_len, threads_per_block](
                    data, indices, indptr, b, x, 
                    wavefronts, wave_start, i
                )



@cuda.jit
def gs_syncfree_unrolling_kernel_32(data, indices, indptr, 
                                         csc_indices_L, csc_indptr_L,
                                         incom_edges, b, x, num_rows):
    """
    Sync-Free GS with unrolling (factor 32).
    1 Warp per Row with 32x explicit loop unrolling for maximum throughput.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(global_warp_id, num_rows, total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        
        sigma = 0.0
        diag = 1.0
        
        # Unrolled parallel reduction (factor 32)
        i = row_start + lane_id
        while i + 31 * 32 < row_end:
            pos = i + 0 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 1 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 2 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 3 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 4 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 5 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 6 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 7 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 8 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 9 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 10 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 11 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 12 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 13 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 14 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 15 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 16 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 17 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 18 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 19 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 20 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 21 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 22 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 23 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 24 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 25 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 26 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 27 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 28 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 29 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 30 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 31 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 32 * 32
        
        # Remainder loop
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 32
        
        # Warp Shuffle Reduction
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
                # Reduce diag
        temp_diag = diag
        for offset in [16, 8, 4, 2, 1]:
            other_diag = cuda.shfl_down_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        temp_diag = diag
        for offset in [1, 2, 4, 8, 16]:
            other_diag = cuda.shfl_up_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        if lane_id == 0 and diag == 1.0:
            for i in range(row_start, row_end):
                if indices[i] == row:
                    diag = data[i]
                    break
        diag = cuda.shfl_sync(mask, diag, 0)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_L[row]
        sig_end = csc_indptr_L[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_L[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_32(data, indices, indptr, 
                                        csc_indices_L, csc_indptr_L, 
                                        base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    """Driver for sync-free unrolling factor 32."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_32[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()



@cuda.jit
def gs_wavefront_unrolling_kernel_64(data, indices, indptr, b, x, 
                                     wavefronts, wave_start, wave_idx):
    """
    Wavefront with explicit unrolling (factor 64).
    Advanced implementation with 64x unrolling for maximum ILP.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if bid < num_rows:
        row = wavefronts[start_wave + bid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        
        shared_sigma = cuda.shared.array(1, dtype=np.float64)
        shared_diag = cuda.shared.array(1, dtype=np.float64)
        
        if tid == 0:
            shared_sigma[0] = 0.0
            shared_diag[0] = 1.0
        cuda.syncthreads()
        
        local_sigma = 0.0
        local_diag = 1.0
        
        # Main unrolled loop: process 64 elements at a time
        i = row_start + tid
        while i + 63 * cuda.blockDim.x < row_end:
            pos = i + 0 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 1 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 2 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 3 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 4 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 5 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 6 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 7 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 8 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 9 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 10 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 11 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 12 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 13 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 14 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 15 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 16 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 17 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 18 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 19 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 20 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 21 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 22 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 23 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 24 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 25 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 26 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 27 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 28 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 29 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 30 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 31 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 32 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 33 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 34 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 35 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 36 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 37 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 38 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 39 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 40 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 41 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 42 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 43 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 44 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 45 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 46 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 47 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 48 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 49 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 50 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 51 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 52 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 53 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 54 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 55 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 56 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 57 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 58 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 59 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 60 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 61 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 62 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            pos = i + 63 * cuda.blockDim.x
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += 64 * cuda.blockDim.x
        
        # Remainder loop
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                local_diag = val
            else:
                local_sigma += val * x[col]
            
            i += cuda.blockDim.x
        
        if local_sigma != 0.0:
            cuda.atomic.add(shared_sigma, 0, local_sigma)
        if local_diag != 1.0:
            shared_diag[0] = local_diag
        
        cuda.syncthreads()
        
        if tid == 0:
            sigma = shared_sigma[0]
            diag = shared_diag[0]
            x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_unrolling_64(data, indices, indptr, b, x, wave_start, 
                                    wavefronts, max_iter=10, threads_per_block=128):
    """Driver for wavefront unrolling factor 64."""
    num_waves = len(wave_start) - 1
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                gs_wavefront_unrolling_kernel_64[grid_len, threads_per_block](
                    data, indices, indptr, b, x, 
                    wavefronts, wave_start, i
                )



@cuda.jit
def gs_syncfree_unrolling_kernel_64(data, indices, indptr, 
                                         csc_indices_L, csc_indptr_L,
                                         incom_edges, b, x, num_rows):
    """
    Sync-Free GS with unrolling (factor 64).
    1 Warp per Row with 64x explicit loop unrolling for maximum throughput.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(global_warp_id, num_rows, total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        
        sigma = 0.0
        diag = 1.0
        
        # Unrolled parallel reduction (factor 64)
        i = row_start + lane_id
        while i + 63 * 32 < row_end:
            pos = i + 0 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 1 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 2 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 3 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 4 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 5 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 6 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 7 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 8 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 9 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 10 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 11 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 12 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 13 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 14 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 15 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 16 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 17 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 18 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 19 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 20 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 21 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 22 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 23 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 24 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 25 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 26 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 27 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 28 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 29 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 30 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 31 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 32 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 33 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 34 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 35 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 36 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 37 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 38 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 39 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 40 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 41 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 42 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 43 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 44 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 45 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 46 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 47 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 48 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 49 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 50 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 51 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 52 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 53 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 54 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 55 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 56 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 57 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 58 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 59 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 60 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 61 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 62 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            pos = i + 63 * 32
            col = indices[pos]
            val = data[pos]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 64 * 32
        
        # Remainder loop
        while i < row_end:
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
            
            i += 32
        
        # Warp Shuffle Reduction
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
                # Reduce diag
        temp_diag = diag
        for offset in [16, 8, 4, 2, 1]:
            other_diag = cuda.shfl_down_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        temp_diag = diag
        for offset in [1, 2, 4, 8, 16]:
            other_diag = cuda.shfl_up_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        if lane_id == 0 and diag == 1.0:
            for i in range(row_start, row_end):
                if indices[i] == row:
                    diag = data[i]
                    break
        diag = cuda.shfl_sync(mask, diag, 0)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_L[row]
        sig_end = csc_indptr_L[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_L[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_64(data, indices, indptr, 
                                        csc_indices_L, csc_indptr_L, 
                                        base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    """Driver for sync-free unrolling factor 64."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_64[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


