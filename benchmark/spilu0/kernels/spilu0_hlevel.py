import sys
import os
# Add project root to path for compiler module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# Add benchmark directory to path for bench_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import numba
from numba import cuda, njit
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
import cupy as cp
import math

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
        next_val = parent[curr]
        parent[curr] = root
        curr = next_val
    return root

@njit
def _union(parent, i, j):
    root_i = _find_root(parent, i)
    root_j = _find_root(parent, j)
    if root_i != root_j:
        parent[root_i] = root_j

@njit
def build_h_level_structure(indptr, indices, wavefronts, wave_start, band_width=64):
    """
    Groups levels into bands, then uses Union-Find to break bands into 
    independent execution chains.
    
    band_width=64 (increased from 16) to maximize parallelism and reduce graph node overhead.
    """
    num_standard_waves = len(wave_start) - 1
    n_rows = len(indptr) - 1
    
    h_task_nodes = np.zeros(len(wavefronts), dtype=np.int32)
    task_ptr = np.zeros(len(wavefronts) + 1, dtype=np.int32)
    h_level_ptr = np.zeros(num_standard_waves + 1, dtype=np.int32)
    
    global_to_local = np.full(n_rows, -1, dtype=np.int32)
    
    current_std_wave = 0
    node_count_global = 0
    task_count_global = 0
    h_level_count = 0
    
    task_ptr[0] = 0
    h_level_ptr[0] = 0
    
    while current_std_wave < num_standard_waves:
        end_wave = min(current_std_wave + band_width, num_standard_waves)
        idx_start = wave_start[current_std_wave]
        idx_end = wave_start[end_wave]
        band_size = idx_end - idx_start
        
        if band_size == 0:
            current_std_wave = end_wave
            continue
            
        parent = np.arange(band_size, dtype=np.int32)
        
        for i in range(band_size):
            global_to_local[wavefronts[idx_start + i]] = i
            
        # Union-Find: Link dependencies within this band
        for i in range(band_size):
            row = wavefronts[idx_start + i]
            for j in range(indptr[row], indptr[row+1]):
                col = indices[j]
                local_col = global_to_local[col]
                # If dependency is within this band, link them
                if local_col != -1 and local_col != i:
                    _union(parent, i, local_col)
        
        for i in range(band_size):
            parent[i] = _find_root(parent, i)
            
        # Extract Chains
        processed = np.zeros(band_size, dtype=np.int8)
        for i in range(band_size):
            if processed[i]: continue
            root = parent[i]
            
            # Form the chain
            for k in range(i, band_size):
                if parent[k] == root:
                    h_task_nodes[node_count_global] = wavefronts[idx_start + k]
                    node_count_global += 1
                    processed[k] = 1
            
            task_count_global += 1
            task_ptr[task_count_global] = node_count_global

        for i in range(band_size):
            global_to_local[wavefronts[idx_start + i]] = -1
            
        h_level_count += 1
        h_level_ptr[h_level_count] = task_count_global
        current_std_wave = end_wave

    return (h_task_nodes[:node_count_global], 
            task_ptr[:task_count_global + 1], 
            h_level_ptr[:h_level_count + 1])

def ilu0_h_level_analyze(indptr, indices, wavefronts, wave_start, band_width=64):
    """
    PERFORM THIS ONCE BEFORE BENCHMARKING.
    Returns a tuple of GPU arrays (h_nodes, task_ptr, h_level_ptr) needed for the solver.
    """
    # Ensure inputs are host arrays for the Numba CPU function
    indptr_h = _to_host_int32(indptr)
    indices_h = _to_host_int32(indices)
    wavefronts_h = _to_host_int32(wavefronts)
    wave_start_h = _to_host_int32(wave_start)
    
    h_nodes, task_ptr, h_level_ptr = build_h_level_structure(
        indptr_h, indices_h, wavefronts_h, wave_start_h, band_width=band_width
    )
    
    # Return device arrays ready for the solver
    return (_to_device_int32(h_nodes), 
            _to_device_int32(task_ptr), 
            h_level_ptr) # Keep level_ptr on host for graph construction

# -------------------------------------------------------------------------
# 3. CSR ILU0 - WARP LEVEL OPTIMIZATION
# -------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def bin_search_col(colind, start, end, key):
    lo = start
    hi = end - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        c = colind[mid]
        if c < key:
            lo = mid + 1
        elif c > key:
            hi = mid - 1
        else:
            return mid
    return -1

@cuda.jit
def ilu0_csr_warp_kernel(data, indices, indptr, diag_pos,
                          h_task_nodes, task_ptr, 
                          level_start_task, level_end_task):
    """
    1 Warp (32 threads) = 1 Independent Chain.
    Performs ILU0 factorization with warp-level parallelism.
    """
    # Global Warp ID calculation
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    warps_per_block = cuda.blockDim.x // 32
    global_warp_id = (cuda.blockIdx.x * warps_per_block) + warp_id
    
    task_idx = level_start_task + global_warp_id
    
    if task_idx >= level_end_task:
        return

    # Load task bounds
    start_node_ptr = task_ptr[task_idx]
    end_node_ptr = task_ptr[task_idx + 1]
    
    # Iterate through the chain
    for i in range(start_node_ptr, end_node_ptr):
        row = h_task_nodes[i]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        # Process each column k < row
        for k_ptr in range(row_start, row_end):
            k = indices[k_ptr]
            if k >= row:
                break
            
            diag_ptr = diag_pos[k]
            u_kk = data[diag_ptr]
            
            if u_kk != 0.0:
                # Lane 0 computes L value
                if lane_id == 0:
                    l_ik = data[k_ptr] / u_kk
                    data[k_ptr] = l_ik
                
                # Broadcast L value to all lanes
                cuda.syncwarp(0xFFFFFFFF)
                l_ik = cuda.shfl_sync(0xFFFFFFFF, data[k_ptr] if lane_id == 0 else 0.0, 0)
                
                k_row_start = indptr[k]
                k_row_end = indptr[k + 1]
                
                # Warp-stride loop for updates
                for j_ptr in range(k_ptr + 1 + lane_id, row_end, 32):
                    jcol = indices[j_ptr]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr] -= l_ik * data[pos]
                
                # Sync before next k iteration
                cuda.syncwarp(0xFFFFFFFF)


def ilu0_numba_csr_h_level_graph(data, indices, indptr, M, analysis_data, diag_pos):
    """
    The actual ILU0 solver function using h-level scheduling with CUDA graphs.
    analysis_data should be the tuple returned by ilu0_h_level_analyze.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    threads_per_block = 128
    warps_per_block = 4
    num_h_levels = len(h_level_ptr) - 1

    # --- CUDA GRAPH CAPTURE ---
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    
    for i in range(num_h_levels):
        start_task = h_level_ptr[i]
        end_task = h_level_ptr[i+1]
        num_tasks = end_task - start_task
        
        if num_tasks > 0:
            grid_size = (num_tasks + warps_per_block - 1) // warps_per_block
            ilu0_csr_warp_kernel[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, diag_pos,
                d_h_nodes, d_task_ptr,
                start_task, end_task
            )

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()


# -------------------------------------------------------------------------
# 5. Compatibility Wrapper
# -------------------------------------------------------------------------

# Global cache to reuse analysis if pointers don't change (Naive memoization)
_ANALYSIS_CACHE = {}

def ilu0_numba_csr_h_level_graph_auto(data, indices, indptr, M, 
                                       wavefronts, wave_start, diag_pos, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = ilu0_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
        
    ilu0_numba_csr_h_level_graph(data, indices, indptr, M, _ANALYSIS_CACHE[key], diag_pos)

# -------------------------------------------------------------------------
# 6. HELPERS FOR SYNC-FREE
# -------------------------------------------------------------------------

@cuda.jit
def reset_counter_kernel(counter):
    """Resets a single-element counter to 0."""
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        counter[0] = 0

@cuda.jit
def set_zero_kernel(arr):
    """Sets all elements of an array to 0."""
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = 0

def get_hardware_grid_cap(threads_per_block):
    """Calculates max resident blocks to prevent deadlock in persistent kernels."""
    device = cuda.get_current_device()
    num_sms = device.MULTIPROCESSOR_COUNT
    # SpILU0 has high register pressure, conservative estimate:
    blocks_per_sm = 8 
    return num_sms * blocks_per_sm

# -------------------------------------------------------------------------
# 7. KERNEL: SYNC-FREE ILU0 H-LEVEL
# -------------------------------------------------------------------------

@cuda.jit
def ilu0_csr_syncfree_warp_kernel(data, indices, indptr, diag_pos,
                                  h_task_nodes, task_ptr,
                                  done, task_counter,
                                  num_tasks_limit, task_offset):
    """
    Sync-Free ILU0 Kernel.
    Uses atomic flags ('done') to manage dependencies between rows.
    """
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    
    while True:
        # 1. Dynamic Task Fetching
        local_task_id = -1
        if lane_id == 0:
            local_task_id = cuda.atomic.add(task_counter, 0, 1)
        local_task_id = cuda.shfl_sync(0xFFFFFFFF, local_task_id, 0)
        
        if local_task_id >= num_tasks_limit:
            return

        global_task_id = task_offset + local_task_id
        
        start_node_ptr = task_ptr[global_task_id]
        end_node_ptr = task_ptr[global_task_id + 1]
        
        # 2. Execute Chain
        for i in range(start_node_ptr, end_node_ptr):
            row = h_task_nodes[i]
            
            row_start = indptr[row]
            row_end = indptr[row + 1]
            
            # Iterate over lower triangular part (columns k < row)
            for k_ptr in range(row_start, row_end):
                k = indices[k_ptr]
                
                # Stop if we reach diagonal or upper
                if k >= row:
                    break
                
                # --- SYNC-FREE WAIT ---
                # Before using row 'k' to update row 'i', row 'k' must be finished.
                if lane_id == 0:
                    while cuda.atomic.add(done, k, 0) == 0:
                        pass
                    cuda.threadfence()
                
                cuda.syncwarp(0xFFFFFFFF)
                # ----------------------
                
                # Proceed with standard ILU0 update logic
                diag_ptr = diag_pos[k]
                u_kk = data[diag_ptr]
                
                if u_kk != 0.0:
                    # Lane 0 computes multiplier L_ik
                    if lane_id == 0:
                        l_ik = data[k_ptr] / u_kk
                        data[k_ptr] = l_ik
                    
                    cuda.syncwarp(0xFFFFFFFF)
                    l_ik = cuda.shfl_sync(0xFFFFFFFF, data[k_ptr] if lane_id == 0 else 0.0, 0)
                    
                    k_row_start = indptr[k]
                    k_row_end = indptr[k + 1]
                    
                    # Parallel Update: A[i, j] -= L[i, k] * U[k, j]
                    for j_ptr in range(k_ptr + 1 + lane_id, row_end, 32):
                        jcol = indices[j_ptr]
                        # Find U[k, j] via binary search
                        pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                        if pos != -1:
                            data[j_ptr] -= l_ik * data[pos]
                    
                    cuda.syncwarp(0xFFFFFFFF)

            # --- MARK COMPLETED ---
            if lane_id == 0:
                cuda.threadfence()
                cuda.atomic.exch(done, row, 1)
            
            cuda.syncwarp(0xFFFFFFFF)

# -------------------------------------------------------------------------
# 8. VARIANT C: BANDED SYNC-FREE
# -------------------------------------------------------------------------

def ilu0_numba_csr_syncfree_h_level_solve(data, indices, indptr, n, analysis_data, diag_pos):
    """
    Banded Sync-Free ILU0. Safely loops over bands to prevent deadlock.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    # Setup Done array (0 = not ready, 1 = ready)
    d_done = cuda.to_device(np.zeros(n, dtype=np.int32))
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    threads_per_block = 128
    warps_per_block = threads_per_block // 32
    
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    
    num_bands = len(h_level_ptr) - 1
    for i in range(num_bands):
        start_task = h_level_ptr[i]
        end_task = h_level_ptr[i+1]
        num_tasks_in_band = end_task - start_task
        
        if num_tasks_in_band > 0:
            # Reset counter for this band
            reset_counter_kernel[1, 1, nb_stream](d_task_counter)
            
            grid_size = (num_tasks_in_band + warps_per_block - 1) // warps_per_block
            grid_size = min(grid_size, 256)
            grid_size = max(grid_size, 1)
            
            ilu0_csr_syncfree_warp_kernel[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, diag_pos,
                d_h_nodes, d_task_ptr,
                d_done, d_task_counter,
                num_tasks_in_band, start_task
            )
            
    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()

def ilu0_numba_csr_syncfree_h_level_graph(data, indices, indptr, n, 
                                          wavefronts, wave_start, diag_pos, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = ilu0_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
    ilu0_numba_csr_syncfree_h_level_solve(data, indices, indptr, n, _ANALYSIS_CACHE[key], diag_pos)

# -------------------------------------------------------------------------
# 9. VARIANT E: PERSISTENT SYNC-FREE
# -------------------------------------------------------------------------

def ilu0_numba_csr_syncfree_persistent_solve(data, indices, indptr, n, analysis_data, diag_pos):
    """
    Persistent Sync-Free ILU0. 
    Launches one grid clamped to hardware capacity. Threads loop until done.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    d_done = cuda.to_device(np.zeros(n, dtype=np.int32))
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    threads_per_block = 128
    warps_per_block = threads_per_block // 32
    
    # Total tasks across the whole matrix
    num_total_tasks = len(d_task_ptr) - 1
    
    # Calculate Grid Size (Clamped)
    needed_blocks = (num_total_tasks + warps_per_block - 1) // warps_per_block
    hardware_cap = get_hardware_grid_cap(threads_per_block)
    
    grid_size = min(needed_blocks, hardware_cap)
    grid_size = max(grid_size, 1)
    
    # Launch Single Persistent Kernel
    ilu0_csr_syncfree_warp_kernel[grid_size, threads_per_block](
        data, indices, indptr, diag_pos,
        d_h_nodes, d_task_ptr,
        d_done, d_task_counter,
        num_total_tasks, 0 # 0 offset
    )

def ilu0_numba_csr_syncfree_persistent_graph(data, indices, indptr, n, 
                                             wavefronts, wave_start, diag_pos, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = ilu0_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
    ilu0_numba_csr_syncfree_persistent_solve(data, indices, indptr, n, _ANALYSIS_CACHE[key], diag_pos)
