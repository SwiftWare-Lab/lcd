import numpy as np
from numba import cuda

# ============================================================================
# BACKWARD PASS: Sync-Free Unrolling Kernels (Diagonal-Optimized)
# ============================================================================

@cuda.jit
def copy_device_array_kernel(src, dst, n):
    tid = cuda.grid(1)
    if tid < n:
        dst[tid] = src[tid]


# ============================================================================
# Sync-Free Unrolling (Factor 4) - BACKWARD with Diagonal Optimization
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_4_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                            csc_indices_U, csc_indptr_U,
                                            incom_edges, b, x, num_rows):
    """Backward pass with unrolling factor 4 and diagonal optimization."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    # Process rows in REVERSE order for backward pass
    for row in range(num_rows - 1 - global_warp_id, -1, -total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        i = row_start + lane_id
        while i + 3 * 32 < row_end:
            for unroll_idx in range(4):
                pos = i + unroll_idx * 32
                if pos != diag_p:
                    col = indices[pos]
                    sigma += data[pos] * x[col]
            
            i += 4 * 32
        
        while i < row_end:
            if i != diag_p:
                col = indices[i]
                sigma += data[i] * x[col]
            
            i += 32
        
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_U[row]
        sig_end = csc_indptr_U[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_U[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_4_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                           csc_indices_U, csc_indptr_U, 
                                           base_incom_edges, b, x, max_iter=1, threads_per_block=128):
    """Driver for sync-free unrolling factor 4 backward pass with diagonal optimization."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_4_diag_bwd[num_blocks, threads_per_block](
            data, indices, indptr, diag_pos, diag_vals, csc_indices_U, csc_indptr_U, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


# ============================================================================
# Sync-Free Unrolling (Factor 8) - BACKWARD with Diagonal Optimization
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_8_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                            csc_indices_U, csc_indptr_U,
                                            incom_edges, b, x, num_rows):
    """Backward pass with unrolling factor 8 and diagonal optimization."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(num_rows - 1 - global_warp_id, -1, -total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        i = row_start + lane_id
        while i + 7 * 32 < row_end:
            for unroll_idx in range(8):
                pos = i + unroll_idx * 32
                if pos != diag_p:
                    col = indices[pos]
                    sigma += data[pos] * x[col]
            
            i += 8 * 32
        
        while i < row_end:
            if i != diag_p:
                col = indices[i]
                sigma += data[i] * x[col]
            
            i += 32
        
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_U[row]
        sig_end = csc_indptr_U[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_U[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_8_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                           csc_indices_U, csc_indptr_U, 
                                           base_incom_edges, b, x, max_iter=1, threads_per_block=128):
    """Driver for sync-free unrolling factor 8 backward pass with diagonal optimization."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_8_diag_bwd[num_blocks, threads_per_block](
            data, indices, indptr, diag_pos, diag_vals, csc_indices_U, csc_indptr_U, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


# ============================================================================
# Sync-Free Unrolling (Factor 16) - BACKWARD with Diagonal Optimization
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_16_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                             csc_indices_U, csc_indptr_U,
                                             incom_edges, b, x, num_rows):
    """Backward pass with unrolling factor 16 and diagonal optimization."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(num_rows - 1 - global_warp_id, -1, -total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        i = row_start + lane_id
        while i + 15 * 32 < row_end:
            for unroll_idx in range(16):
                pos = i + unroll_idx * 32
                if pos != diag_p:
                    col = indices[pos]
                    sigma += data[pos] * x[col]
            
            i += 16 * 32
        
        while i < row_end:
            if i != diag_p:
                col = indices[i]
                sigma += data[i] * x[col]
            
            i += 32
        
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_U[row]
        sig_end = csc_indptr_U[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_U[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_16_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                            csc_indices_U, csc_indptr_U, 
                                            base_incom_edges, b, x, max_iter=1, threads_per_block=128):
    """Driver for sync-free unrolling factor 16 backward pass with diagonal optimization."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_16_diag_bwd[num_blocks, threads_per_block](
            data, indices, indptr, diag_pos, diag_vals, csc_indices_U, csc_indptr_U, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


# ============================================================================
# Sync-Free Unrolling (Factor 32) - BACKWARD with Diagonal Optimization
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_32_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                             csc_indices_U, csc_indptr_U,
                                             incom_edges, b, x, num_rows):
    """Backward pass with unrolling factor 32 and diagonal optimization."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(num_rows - 1 - global_warp_id, -1, -total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        i = row_start + lane_id
        while i + 31 * 32 < row_end:
            for unroll_idx in range(32):
                pos = i + unroll_idx * 32
                if pos != diag_p:
                    col = indices[pos]
                    sigma += data[pos] * x[col]
            
            i += 32 * 32
        
        while i < row_end:
            if i != diag_p:
                col = indices[i]
                sigma += data[i] * x[col]
            
            i += 32
        
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_U[row]
        sig_end = csc_indptr_U[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_U[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_32_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                            csc_indices_U, csc_indptr_U, 
                                            base_incom_edges, b, x, max_iter=1, threads_per_block=128):
    """Driver for sync-free unrolling factor 32 backward pass with diagonal optimization."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_32_diag_bwd[num_blocks, threads_per_block](
            data, indices, indptr, diag_pos, diag_vals, csc_indices_U, csc_indptr_U, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


# ============================================================================
# Sync-Free Unrolling (Factor 64) - BACKWARD with Diagonal Optimization
# ============================================================================

@cuda.jit
def gs_syncfree_unrolling_kernel_64_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                             csc_indices_U, csc_indptr_U,
                                             incom_edges, b, x, num_rows):
    """Backward pass with unrolling factor 64 and diagonal optimization."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    for row in range(num_rows - 1 - global_warp_id, -1, -total_warps):
        
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        i = row_start + lane_id
        while i + 63 * 32 < row_end:
            for unroll_idx in range(64):
                pos = i + unroll_idx * 32
                if pos != diag_p:
                    col = indices[pos]
                    sigma += data[pos] * x[col]
            
            i += 64 * 32
        
        while i < row_end:
            if i != diag_p:
                col = indices[i]
                sigma += data[i] * x[col]
            
            i += 32
        
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
            
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        sig_start = csc_indptr_U[row]
        sig_end = csc_indptr_U[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_U[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_unrolling_64_diag_bwd(data, indices, indptr, diag_pos, diag_vals,
                                            csc_indices_U, csc_indptr_U, 
                                            base_incom_edges, b, x, max_iter=1, threads_per_block=128):
    """Driver for sync-free unrolling factor 64 backward pass with diagonal optimization."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_unrolling_kernel_64_diag_bwd[num_blocks, threads_per_block](
            data, indices, indptr, diag_pos, diag_vals, csc_indices_U, csc_indptr_U, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()
