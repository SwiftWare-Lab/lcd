import numpy as np
from numba import cuda

# ============================================================================
# GPU Baseline Implementations with Diagonal Optimization
# ============================================================================

@cuda.jit
def gs_wavefront_kernel_diag(data, indices, indptr, diag_pos, diag_vals, b, x, 
                              wavefronts, wave_start, wave_idx):
    """
    Standard Wavefront with pre-extracted diagonal. One Thread per Row.
    Eliminates branch in hot loop by using pre-extracted diagonal.
    """
    tid = cuda.grid(1)
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if tid < num_rows:
        row = wavefronts[start_wave + tid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        # Compute Sum (A[i,:] * x[:]) - No branch!
        sigma = 0.0
        
        for i in range(row_start, row_end):
            if i != diag_p:  # Skip diagonal position
                col = indices[i]
                sigma += data[i] * x[col]
        
        # Update x[i] using pre-extracted diagonal
        x[row] = (b[row] - sigma) / diag_vals[row]


def gs_numba_wavefront_diag(data, indices, indptr, diag_pos, diag_vals, b, x, 
                             wave_start, wavefronts, max_iter=10):
    """
    Driver: Loops over iterations, and inside that, loops over wavefronts.
    """
    num_waves = len(wave_start) - 1
    threads_per_block = 128
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                blocks = (grid_len + threads_per_block - 1) // threads_per_block
                gs_wavefront_kernel_diag[blocks, threads_per_block](
                    data, indices, indptr, diag_pos, diag_vals, b, x, 
                    wavefronts, wave_start, i
                )


@cuda.jit
def gs_syncfree_kernel_diag(data, indices, indptr, diag_pos, diag_vals,
                             csc_indices_L, csc_indptr_L,
                             incom_edges, b, x, num_rows):
    """
    Sync-Free GS with pre-extracted diagonal.
    1 Warp per Row. Eliminates diagonal search and branch.
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
    
    # Persistent Threads Loop
    for row in range(global_warp_id, num_rows, total_warps):
        
        # Wait (Dependency Check)
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        # Compute Row
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        # Parallel Reduction over the row - No branch!
        for i in range(row_start + lane_id, row_end, 32):
            if i != diag_p:  # Skip diagonal position
                col = indices[i]
                sigma += data[i] * x[col]
        
        # Warp Shuffle Reduction for sigma
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        # Lane 0 updates using pre-extracted diagonal
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
            
        # Fence
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        # Signal Dependents (Lower Triangle Only)
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


def gs_numba_syncfree_diag(data, indices, indptr, diag_pos, diag_vals, 
                            csc_indices_L, csc_indptr_L, 
                            base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    """
    Driver. Must reset dependencies every iteration.
    threads_per_block: Number of threads per block (must be multiple of 32).
    """
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        # RESET Dependencies for this iteration
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_kernel_diag[num_blocks, threads_per_block](
            data, indices, indptr, diag_pos, diag_vals, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()

