import numpy as np
from numba import cuda

@cuda.jit
def copy_device_array_kernel(src, dst, n):
    tid = cuda.grid(1)
    if tid < n:
        dst[tid] = src[tid]


@cuda.jit
def gs_wavefront_kernel_bwd(data, indices, indptr, b, x, 
                            wavefronts, wave_start, wave_idx):
    """Backward pass: Standard Wavefront. One Thread per Row."""
    tid = cuda.grid(1)
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if tid < num_rows:
        row = wavefronts[start_wave + tid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]

        sigma = 0.0
        diag = 1.0
        
        for i in range(row_start, row_end):
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]

        x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront_bwd(data, indices, indptr, b, x, wave_start, wavefronts, max_iter=10):
    """
    Backward pass: Process wavefronts in FORWARD order.
    Note: The wavefront ordering itself is already computed for backward
    dependencies (upper triangle), so we process waves 0->N, not N->0.
    """
    num_waves = len(wave_start) - 1
    threads_per_block = 128
    
    for k in range(max_iter):
        # Process wavefronts in FORWARD order (waves already ordered for backward)
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                blocks = (grid_len + threads_per_block - 1) // threads_per_block
                gs_wavefront_kernel_bwd[blocks, threads_per_block](
                    data, indices, indptr, b, x, wavefronts, wave_start, i
                )


@cuda.jit
def gs_syncfree_kernel_bwd(data, indices, indptr, 
                           csc_indices_U, csc_indptr_U,
                           incom_edges, b, x, num_rows):
    """Backward pass: processes from last row to first (upper triangle dependencies)."""
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
        
        sigma = 0.0
        diag = 1.0

        for i in range(row_start + lane_id, row_end, 32):
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]

        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)

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

        # Signal dependents (upper triangle for backward pass)
        sig_start = csc_indptr_U[row]
        sig_end = csc_indptr_U[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_U[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


def gs_numba_syncfree_bwd(data, indices, indptr, csc_indices_U, csc_indptr_U, 
                          base_incom_edges, b, x, max_iter=1, threads_per_block=128):
    """Backward pass using upper triangle dependencies."""
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_kernel_bwd[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_U, csc_indptr_U, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()
