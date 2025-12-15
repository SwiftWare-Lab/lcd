import numpy as np
from numba import cuda
import cupy as cp

def get_safe_persistent_grid_size(threads_per_block):
    """Calculates a CONSERVATIVE grid size."""
    device = cuda.get_current_device()
    num_sms = device.MULTIPROCESSOR_COUNT
    blocks_per_sm = 2 
    return num_sms * blocks_per_sm


@cuda.jit
def reset_counter_kernel(counter):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        counter[0] = 0


@cuda.jit
def gs_hlevel_syncfree_kernel_bwd(data, indices, indptr, b, x,
                                  h_task_nodes, task_ptr,
                                  done, task_counter,
                                  num_tasks_limit, task_offset,
                                  sub_warp_size, iter_offset):
    """Backward pass: processes rows from high to low (upper triangular dependencies)."""
    tid = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    
    local_group_id = tid // sub_warp_size
    lane = tid % sub_warp_size
    
    warp_lane_id = tid % 32
    warp_group_id = warp_lane_id // sub_warp_size
    
    mask_bits = (1 << sub_warp_size) - 1
    sub_warp_mask = (mask_bits << (warp_group_id * sub_warp_size)) & 0xFFFFFFFF
    root_lane_in_warp = warp_group_id * sub_warp_size
        
    while True:
        local_task_id = -1
        if lane == 0:
            local_task_id = cuda.atomic.add(task_counter, 0, 1)
        
        local_task_id = cuda.shfl_sync(sub_warp_mask, local_task_id, root_lane_in_warp)
        
        if local_task_id >= num_tasks_limit:
            return

        global_task_id = task_offset + local_task_id
        start_node = task_ptr[global_task_id]
        end_node = task_ptr[global_task_id + 1]
        
        for i in range(start_node, end_node):
            row = h_task_nodes[i]
            row_start = indptr[row]
            row_end = indptr[row + 1]
            
            sigma = 0.0
            diag = 1.0
            
            for j in range(row_start + lane, row_end, sub_warp_size):
                col = indices[j]
                val = data[j]
                
                if col == row:
                    diag = val
                else:
                    # Backward: wait if col > row (upper triangle)
                    if col > row:
                        while cuda.atomic.add(done, col, 0) < iter_offset:
                            pass
                        cuda.threadfence()
                    
                    sigma += val * x[col]

            cuda.syncwarp(sub_warp_mask)

            # Reduction for sigma
            if sub_warp_size >= 32: sigma += cuda.shfl_down_sync(sub_warp_mask, sigma, 16)
            if sub_warp_size >= 16: sigma += cuda.shfl_down_sync(sub_warp_mask, sigma, 8)
            if sub_warp_size >= 8:  sigma += cuda.shfl_down_sync(sub_warp_mask, sigma, 4)
            if sub_warp_size >= 4:  sigma += cuda.shfl_down_sync(sub_warp_mask, sigma, 2)
            if sub_warp_size >= 2:  sigma += cuda.shfl_down_sync(sub_warp_mask, sigma, 1)
            
            # Diagonal reduction
            temp_diag = diag
            if sub_warp_size >= 32:
                o = cuda.shfl_down_sync(sub_warp_mask, temp_diag, 16)
                if o != 1.0: temp_diag = o
            if sub_warp_size >= 16:
                o = cuda.shfl_down_sync(sub_warp_mask, temp_diag, 8)
                if o != 1.0: temp_diag = o
            if sub_warp_size >= 8:
                o = cuda.shfl_down_sync(sub_warp_mask, temp_diag, 4)
                if o != 1.0: temp_diag = o
            if sub_warp_size >= 4:
                o = cuda.shfl_down_sync(sub_warp_mask, temp_diag, 2)
                if o != 1.0: temp_diag = o
            if sub_warp_size >= 2:
                o = cuda.shfl_down_sync(sub_warp_mask, temp_diag, 1)
                if o != 1.0: temp_diag = o
            diag = temp_diag

            if lane == 0 and diag == 1.0:
                for j in range(row_start, row_end):
                    if indices[j] == row:
                        diag = data[j]
                        break
            
            if lane == 0:
                x[row] = (b[row] - sigma) / diag
                cuda.threadfence()
                cuda.atomic.exch(done, row, iter_offset)
            
            cuda.syncwarp(sub_warp_mask)


def gs_numba_hlevel_syncfree_persistent_solve_bwd(data, indices, indptr, b, x, analysis_data, 
                                                   max_iter=1, sub_warp_size=8):
    """Backward pass solver."""
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    n = len(b)
    
    d_done = cuda.to_device(np.zeros(n, dtype=np.int32))
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    threads_per_block = 128
    chains_per_block = threads_per_block // sub_warp_size
    num_total_tasks = len(d_task_ptr) - 1
    
    hardware_cap = get_safe_persistent_grid_size(threads_per_block)
    grid_size = min((num_total_tasks + chains_per_block - 1) // chains_per_block, hardware_cap)
    grid_size = max(grid_size, 1)
    
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    gs_hlevel_syncfree_kernel_bwd[grid_size, threads_per_block, nb_stream](
        data, indices, indptr, b, x,
        d_h_nodes, d_task_ptr,
        d_done, d_task_counter,
        num_total_tasks, 0,
        sub_warp_size, 1
    )
    cp_stream.synchronize()
