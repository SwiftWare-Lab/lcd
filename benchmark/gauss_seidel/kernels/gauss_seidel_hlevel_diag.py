import numpy as np
import numba
from numba import cuda, njit
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
import cupy as cp

# -------------------------------------------------------------------------
# 1. HELPER: Device Transfers
# -------------------------------------------------------------------------

def _to_device_int32(arr):
    return cuda.to_device(np.asarray(arr, dtype=np.int32))

def _to_host_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr.copy_to_host().astype(np.int32)
    return np.asarray(arr, dtype=np.int32)

# -------------------------------------------------------------------------
# 2. ANALYSIS PHASE: Chain Decomposition (Union-Find)
# -------------------------------------------------------------------------

@njit
def _find_root(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    curr = i
    while curr != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt
    return root

@njit
def _union(parent, i, j):
    root_i = _find_root(parent, i)
    root_j = _find_root(parent, j)
    if root_i != root_j:
        parent[root_i] = root_j

@njit
def build_gs_h_level_diag(indptr, indices, wavefronts, wave_start, band_width=64):
    """
    Builds chains based on L (Lower Triangle) ONLY.
    Upper triangle entries do not create hard synchronization barriers 
    because we use values from the previous iteration.
    """
    num_waves = len(wave_start) - 1
    n_rows = len(indptr) - 1
    
    h_nodes = np.zeros(len(wavefronts), dtype=np.int32)
    task_ptr = np.zeros(len(wavefronts) + 1, dtype=np.int32)
    h_level_ptr = np.zeros(num_waves + 1, dtype=np.int32)
    global_to_local = np.full(n_rows, -1, dtype=np.int32)
    
    node_cnt = 0
    task_cnt = 0
    level_cnt = 0
    task_ptr[0] = 0
    h_level_ptr[0] = 0
    
    curr_wave = 0
    while curr_wave < num_waves:
        end_wave = min(curr_wave + band_width, num_waves)
        start_idx = wave_start[curr_wave]
        end_idx = wave_start[end_wave]
        band_size = end_idx - start_idx
        
        if band_size == 0:
            curr_wave = end_wave
            continue

        parent = np.arange(band_size, dtype=np.int32)
        
        for i in range(band_size):
            global_to_local[wavefronts[start_idx + i]] = i
            
        # Union-Find: Only consider LOWER triangle dependencies
        for i in range(band_size):
            row = wavefronts[start_idx + i]
            for ptr in range(indptr[row], indptr[row+1]):
                col = indices[ptr]
                
                if col < row:
                    local_col = global_to_local[col]
                    if local_col != -1 and local_col != i:
                        _union(parent, i, local_col)
                    
        for i in range(band_size):
            parent[i] = _find_root(parent, i)
            
        processed = np.zeros(band_size, dtype=np.int8)
        for i in range(band_size):
            if processed[i]: continue
            root = parent[i]
            for k in range(i, band_size):
                if parent[k] == root:
                    h_nodes[node_cnt] = wavefronts[start_idx + k]
                    node_cnt += 1
                    processed[k] = 1
            task_cnt += 1
            task_ptr[task_cnt] = node_cnt
            
        for i in range(band_size):
            global_to_local[wavefronts[start_idx + i]] = -1
            
        level_cnt += 1
        h_level_ptr[level_cnt] = task_cnt
        curr_wave = end_wave
        
    return h_nodes[:node_cnt], task_ptr[:task_cnt+1], h_level_ptr[:level_cnt+1]

def gs_h_level_analyze_diag(indptr, indices, wavefronts, wave_start, band_width=64):
    """
    PERFORM THIS ONCE BEFORE BENCHMARKING.
    Returns a tuple of GPU arrays (h_nodes, task_ptr, h_level_ptr) needed for the solver.
    """
    indptr_h = _to_host_int32(indptr)
    indices_h = _to_host_int32(indices)
    waves_h = _to_host_int32(wavefronts)
    wave_start_h = _to_host_int32(wave_start)
    h_nodes, task_ptr, level_ptr = build_gs_h_level_diag(
        indptr_h, indices_h, waves_h, wave_start_h, band_width
    )
    return (_to_device_int32(h_nodes), _to_device_int32(task_ptr), level_ptr)

# -------------------------------------------------------------------------
# 3. H-LEVEL KERNEL with Diagonal Optimization
# -------------------------------------------------------------------------

@cuda.jit
def gs_hlevel_kernel_diag(data, indices, indptr, diag_pos, diag_vals, b, x,
                           h_task_nodes, task_ptr, 
                           level_start, level_end,
                           sub_warp_size):
    """
    Sub-Warp Adaptive Kernel for Gauss-Seidel with pre-extracted diagonal.
    Eliminates diagonal search and branch in hot loop.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    
    groups_per_block = block_dim // sub_warp_size
    local_group = tid // sub_warp_size
    lane = tid % sub_warp_size
    
    global_group = (bid * groups_per_block) + local_group
    task_idx = level_start + global_group
    
    # Construct mask for this sub-warp
    sub_warp_mask = 0
    if sub_warp_size == 32:
        sub_warp_mask = 0xFFFFFFFF
    else:
        sub_warp_mask = ((1 << sub_warp_size) - 1) << (local_group * sub_warp_size)

    if task_idx >= level_end:
        return

    start_node = task_ptr[task_idx]
    end_node = task_ptr[task_idx + 1]
    
    # Process Chain Sequentially
    for i in range(start_node, end_node):
        row = h_task_nodes[i]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        diag_p = diag_pos[row]
        
        sigma = 0.0
        
        # Parallel Reduction within Sub-Warp - No branch!
        for j in range(row_start + lane, row_end, sub_warp_size):
            if j != diag_p:  # Skip diagonal position
                col = indices[j]
                sigma += data[j] * x[col]
                
        # Sub-Warp Reduction for sigma
        mask = sub_warp_mask
        if sub_warp_size >= 32: sigma += cuda.shfl_down_sync(mask, sigma, 16)
        if sub_warp_size >= 16: sigma += cuda.shfl_down_sync(mask, sigma, 8)
        if sub_warp_size >= 8:  sigma += cuda.shfl_down_sync(mask, sigma, 4)
        if sub_warp_size >= 4:  sigma += cuda.shfl_down_sync(mask, sigma, 2)
        if sub_warp_size >= 2:  sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        # Use pre-extracted diagonal
        if lane == 0:
            x[row] = (b[row] - sigma) / diag_vals[row]
        
        # Sync Sub-Warp before next node (intra-chain dependency)
        cuda.syncwarp(mask)


def gs_numba_hlevel_solve_diag(data, indices, indptr, diag_pos, diag_vals, b, x, 
                                analysis_data, max_iter=10, sub_warp_size=8):
    """
    The actual solver function using h-level scheduling with CUDA graphs.
    analysis_data should be the tuple returned by gs_h_level_analyze_diag.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    threads_per_block = 128
    chains_per_block = threads_per_block // sub_warp_size
    num_h_levels = len(h_level_ptr) - 1
    
    # Capture One Iteration
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    
    for i in range(num_h_levels):
        start_task = h_level_ptr[i]
        end_task = h_level_ptr[i+1]
        num_tasks = end_task - start_task
        if num_tasks > 0:
            grid_size = (num_tasks + chains_per_block - 1) // chains_per_block
            gs_hlevel_kernel_diag[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, diag_pos, diag_vals, b, x,
                d_h_nodes, d_task_ptr,
                start_task, end_task,
                sub_warp_size
            )
            
    graph = cp_stream.end_capture()
    
    # Launch Graph for Max Iterations
    for k in range(max_iter):
        graph.launch(stream=cp_stream)
        
    cp_stream.synchronize()

