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

def sptrsv_h_level_analyze(indptr, indices, wavefronts, wave_start, band_width=64):
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
# 3. VARIANT A: CSR (Pull) - WARP LEVEL OPTIMIZATION
# -------------------------------------------------------------------------

@cuda.jit
def sptrsv_csr_warp_kernel(data, indices, indptr, b, y, 
                           h_task_nodes, task_ptr, 
                           level_start_task, level_end_task):
    """
    1 Warp (32 threads) = 1 Independent Chain.
    Uses register shuffles for reduction.
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
        diag_idx = row_end - 1
        
        val = 0.0
        
        # Warp-stride loop 
        # Note: 'data' and 'indices' access is coalesced-ish (warp stride)
        for j in range(row_start + lane_id, diag_idx, 32):
            col = indices[j]
            data_val = data[j]
            val += data_val * y[col]
            
        # Register Reduction (No Shared Memory)
        mask = 0xFFFFFFFF
        val += cuda.shfl_down_sync(mask, val, 16)
        val += cuda.shfl_down_sync(mask, val, 8)
        val += cuda.shfl_down_sync(mask, val, 4)
        val += cuda.shfl_down_sync(mask, val, 2)
        val += cuda.shfl_down_sync(mask, val, 1)
        
        if lane_id == 0:
            diag_val = data[diag_idx]
            y[row] = (b[row] - val) / diag_val
        
        # Essential sync for the chain dependency
        cuda.syncwarp(mask)


def sptrsv_numba_csr_h_level_solve(data, indices, indptr, b, M, N, nnz, analysis_data):
    """
    The actual solver function. 
    analysis_data should be the tuple returned by sptrsv_h_level_analyze.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    
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
            sptrsv_csr_warp_kernel[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, b, d_y,
                d_h_nodes, d_task_ptr,
                start_task, end_task
            )

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
            
    return d_y


# -------------------------------------------------------------------------
# 4. VARIANT B: CSC (Push) - WARP LEVEL OPTIMIZATION
# -------------------------------------------------------------------------

@cuda.jit
def sptrsv_csc_warp_kernel(data, indices, indptr, b, y, 
                           h_task_nodes, task_ptr, 
                           level_start_task, level_end_task):
    """
    1 Warp = 1 Independent Chain.
    Push-based scatter using 32 threads.
    """
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    warps_per_block = cuda.blockDim.x // 32
    global_warp_id = (cuda.blockIdx.x * warps_per_block) + warp_id
    
    task_idx = level_start_task + global_warp_id
    
    if task_idx >= level_end_task:
        return

    start_node_ptr = task_ptr[task_idx]
    end_node_ptr = task_ptr[task_idx + 1]

    for i in range(start_node_ptr, end_node_ptr):
        col = h_task_nodes[i]
        
        col_start = indptr[col]
        col_end = indptr[col + 1]
        
        # Thread 0 computes self
        if lane_id == 0:
            diag_val = data[col_start]
            y[col] = (b[col] - y[col]) / diag_val
            
        cuda.syncwarp(0xFFFFFFFF)
        
        # Broadcast
        y_val = cuda.shfl_sync(0xFFFFFFFF, y[col], 0)
        
        # Warp-stride Scatter
        # Note: Atomic Add is necessary even with independent chains
        # because "Independent" means read-after-write is safe, 
        # but Write-After-Write (two independent chains pushing to the same future row) 
        # still requires atomics.
        for j in range(col_start + 1 + lane_id, col_end, 32):
            row_idx = indices[j]
            val = data[j]
            cuda.atomic.add(y, row_idx, val * y_val)
            
        cuda.syncwarp(0xFFFFFFFF)


def sptrsv_numba_csc_h_level_solve(data, indices, indptr, b, M, N, nnz, analysis_data):
    """
    The actual solver function for CSC.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    
    threads_per_block = 128
    warps_per_block = 4
    num_h_levels = len(h_level_ptr) - 1
    
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    
    for i in range(num_h_levels):
        start_task = h_level_ptr[i]
        end_task = h_level_ptr[i+1]
        num_tasks = end_task - start_task
        
        if num_tasks > 0:
            grid_size = (num_tasks + warps_per_block - 1) // warps_per_block
            sptrsv_csc_warp_kernel[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, b, d_y,
                d_h_nodes, d_task_ptr,
                start_task, end_task
            )
            
    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
            
    return d_y

# -------------------------------------------------------------------------
# 5. Compatibility Wrapper
# -------------------------------------------------------------------------

_ANALYSIS_CACHE = {}

def sptrsv_numba_csc_h_level_graph(data, indices, indptr, b, M, N, nnz, 
                                   wavefronts, wave_start, band_width=64):
    """
    Compatibility wrapper.
    """
    key = (indptr.device_ctypes_pointer.value, band_width)
    
    if key not in _ANALYSIS_CACHE:
        # print("Performing H-Level Analysis (One-time)...")
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
        
    return sptrsv_numba_csc_h_level_solve(
        data, indices, indptr, b, M, N, nnz, _ANALYSIS_CACHE[key]
    )

def sptrsv_numba_csr_h_level_graph(data, indices, indptr, b, M, N, nnz, 
                                   wavefronts, wave_start, band_width=64):
    """
    Compatibility wrapper for CSR.
    """
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
        
    return sptrsv_numba_csr_h_level_solve(
        data, indices, indptr, b, M, N, nnz, _ANALYSIS_CACHE[key]
    )




@cuda.jit
def reset_counter_kernel(counter):
    """Resets a single-element counter to 0."""
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        counter[0] = 0

@cuda.jit
def set_zero_kernel(arr):
    """Sets all elements of an array to 0.0"""
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = 0.0

# -------------------------------------------------------------------------
# 6. VARIANT C: Sync-Free H-Level (Chain) - Coarsened DAG (CSR)
# -------------------------------------------------------------------------

@cuda.jit
def sptrsv_csr_syncfree_warp_kernel(data, indices, indptr, b, y, 
                                    h_task_nodes, task_ptr, 
                                    done, task_counter, 
                                    num_tasks_in_band, task_offset):
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    
    while True:
        # 1. Dynamic Task Fetching (Local to this Band)
        local_task_id = -1
        if lane_id == 0:
            local_task_id = cuda.atomic.add(task_counter, 0, 1)
        local_task_id = cuda.shfl_sync(0xFFFFFFFF, local_task_id, 0)
        
        if local_task_id >= num_tasks_in_band:
            return

        # 2. Calculate Global Task ID
        global_task_id = task_offset + local_task_id

        # 3. Execute Chain
        start_node_ptr = task_ptr[global_task_id]
        end_node_ptr = task_ptr[global_task_id + 1]
        
        for i in range(start_node_ptr, end_node_ptr):
            row = h_task_nodes[i]
            
            row_start = indptr[row]
            row_end = indptr[row + 1]
            diag_idx = row_end - 1
            
            val = 0.0
            
            # Warp-stride accumulation
            for j in range(row_start + lane_id, diag_idx, 32):
                col = indices[j]
                
                # Busy-wait for dependency
                while cuda.atomic.add(done, col, 0) == 0:
                    pass 
                
                cuda.threadfence()

                data_val = data[j]
                val += data_val * y[col]
            
            cuda.syncwarp(0xFFFFFFFF)

            # Reduction
            mask = 0xFFFFFFFF
            val += cuda.shfl_down_sync(mask, val, 16)
            val += cuda.shfl_down_sync(mask, val, 8)
            val += cuda.shfl_down_sync(mask, val, 4)
            val += cuda.shfl_down_sync(mask, val, 2)
            val += cuda.shfl_down_sync(mask, val, 1)
            
            if lane_id == 0:
                diag_val = data[diag_idx]
                y[row] = (b[row] - val) / diag_val
                
                cuda.threadfence() 
                cuda.atomic.exch(done, row, 1) 
            
            cuda.syncwarp(mask)

def sptrsv_numba_csr_syncfree_h_level_solve(data, indices, indptr, b, M, N, nnz, analysis_data):
    """
    Host wrapper for the sync-free H-level solver.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    # 1. Device Array Robustness
    if cuda.is_cuda_array(b):
        d_y = cuda.device_array_like(b)
        # Initialize y to 0.0 for CSR accumulation safely
        init_blocks = (M + 127) // 128
        set_zero_kernel[init_blocks, 128](d_y)
    else:
        d_y = cuda.to_device(np.zeros(M, dtype=np.float64))

    d_done = cuda.to_device(np.zeros(M, dtype=np.int32))
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    threads_per_block = 128
    warps_per_block = threads_per_block // 32

    # --- CUDA GRAPH CAPTURE ---
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    
    # Iterate over BANDS to enforce safety
    num_bands = len(h_level_ptr) - 1
    for i in range(num_bands):
        start_task = h_level_ptr[i]
        end_task = h_level_ptr[i+1]
        num_tasks_in_band = end_task - start_task
        
        if num_tasks_in_band > 0:
            # 1. Reset Counter
            reset_counter_kernel[1, 1, nb_stream](d_task_counter)
            
            # 2. Launch Kernel for this Band
            grid_size = (num_tasks_in_band + warps_per_block - 1) // warps_per_block
            grid_size = min(grid_size, 256) # Cap grid size to avoid launch overhead
            grid_size = max(grid_size, 1)
            
            sptrsv_csr_syncfree_warp_kernel[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, b, d_y,
                d_h_nodes, d_task_ptr,
                d_done, d_task_counter, 
                num_tasks_in_band, start_task # Pass Offset
            )
    
    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    
    return d_y

def sptrsv_numba_csr_syncfree_h_level_graph(data, indices, indptr, b, M, N, nnz, 
                                        wavefronts, wave_start, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
        
    return sptrsv_numba_csr_syncfree_h_level_solve(
        data, indices, indptr, b, M, N, nnz, _ANALYSIS_CACHE[key]
    )

# -------------------------------------------------------------------------
# 7. VARIANT D: Sync-Free H-Level (Chain) - Coarsened DAG (CSC)
# -------------------------------------------------------------------------

@cuda.jit
def sptrsv_csc_syncfree_warp_kernel(data, indices, indptr, b, y, 
                                    h_task_nodes, task_ptr, 
                                    in_degree, task_counter, 
                                    num_tasks_in_band, task_offset):
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    
    while True:
        # 1. Fetch Chain (Dynamic Schedule Local to Band)
        local_task_id = -1
        if lane_id == 0:
            local_task_id = cuda.atomic.add(task_counter, 0, 1)
        local_task_id = cuda.shfl_sync(0xFFFFFFFF, local_task_id, 0)
        
        if local_task_id >= num_tasks_in_band:
            return

        # 2. Calculate Global Task ID
        global_task_id = task_offset + local_task_id

        # 3. Process Chain
        start_node_ptr = task_ptr[global_task_id]
        end_node_ptr = task_ptr[global_task_id + 1]
        
        for i in range(start_node_ptr, end_node_ptr):
            col = h_task_nodes[i]
            col_start = indptr[col]
            col_end = indptr[col + 1]
            
            # --- WAIT PHASE ---
            if lane_id == 0:
                while cuda.atomic.add(in_degree, col, 0) > 0:
                    pass
                cuda.threadfence()

                diag_val = data[col_start]
                y[col] = y[col] / diag_val
            
            cuda.syncwarp(0xFFFFFFFF)
            
            my_val = cuda.shfl_sync(0xFFFFFFFF, y[col], 0)
            
            # --- PUSH PHASE ---
            for j in range(col_start + 1 + lane_id, col_end, 32):
                target_row = indices[j]
                val = data[j]
                cuda.atomic.add(y, target_row, -val * my_val)
                cuda.threadfence()
                cuda.atomic.add(in_degree, target_row, -1)
                
            cuda.syncwarp(0xFFFFFFFF)

def sptrsv_numba_csc_syncfree_h_level_solve(data, indices, indptr, b, M, N, nnz, 
                                            analysis_data, in_edge_counts):
    """
    Host wrapper for CSC Sync-Free.
    """
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    # 1. Device Array Robustness
    if cuda.is_cuda_array(b):
        d_y = cuda.device_array_like(b)
        d_y.copy_to_device(b)
    else:
        d_y = cuda.to_device(b.copy())
    
    # 2. Dependency Counters FIX: Subtract 1 for the diagonal
    # The kernel waits for in_degree == 0. 
    # incoming_edge() counts ALL nnz (including diagonal).
    # We must subtract 1 so roots start at 0 and others start at (nnz-1).
    if cuda.is_cuda_array(in_edge_counts):
        # Copy to host to perform subtraction safely/easily, then send back
        # (Overhead is negligible compared to solver time)
        counts_host = in_edge_counts.copy_to_host().astype(np.int32)
        d_in_degree = cuda.to_device(counts_host - 1)
    else:
        counts_host = np.asarray(in_edge_counts, dtype=np.int32)
        d_in_degree = cuda.to_device(counts_host - 1)
    
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    threads_per_block = 128
    warps_per_block = threads_per_block // 32
    
    # --- CUDA GRAPH CAPTURE ---
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    
    # Iterate over BANDS to enforce safety
    num_bands = len(h_level_ptr) - 1
    for i in range(num_bands):
        start_task = h_level_ptr[i]
        end_task = h_level_ptr[i+1]
        num_tasks_in_band = end_task - start_task
        
        if num_tasks_in_band > 0:
            # 1. Reset Counter
            reset_counter_kernel[1, 1, nb_stream](d_task_counter)
            
            # 2. Launch Kernel
            grid_size = (num_tasks_in_band + warps_per_block - 1) // warps_per_block
            grid_size = min(grid_size, 256)
            grid_size = max(grid_size, 1)
            
            sptrsv_csc_syncfree_warp_kernel[grid_size, threads_per_block, nb_stream](
                data, indices, indptr, b, d_y,
                d_h_nodes, d_task_ptr,
                d_in_degree, d_task_counter, 
                num_tasks_in_band, start_task # Pass Offset
            )
    
    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    
    return d_y

def sptrsv_numba_csc_syncfree_h_level_graph(data, indices, indptr, b, M, N, nnz, 
                                            wavefronts, wave_start, in_edge_counts, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
        
    return sptrsv_numba_csc_syncfree_h_level_solve(
        data, indices, indptr, b, M, N, nnz, 
        _ANALYSIS_CACHE[key], in_edge_counts
    )



def get_hardware_grid_cap(threads_per_block):
    # Calculates the maximum number of blocks the GPU can execute simultaneously.
    device = cuda.get_current_device()
    num_sms = device.MULTIPROCESSOR_COUNT
    
    # TODO (Da): Can be tuned.
    blocks_per_sm = 8
    
    return num_sms * blocks_per_sm

# -------------------------------------------------------------------------
# 8. VARIANT E: Persistent Sync-Free H-Level (CSR)
# -------------------------------------------------------------------------

def sptrsv_numba_csr_syncfree_persistent_solve(data, indices, indptr, b, M, N, nnz, analysis_data):
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    # 1. Memory Setup
    if cuda.is_cuda_array(b):
        d_y = cuda.device_array_like(b)
        # Initialize y to 0.0
        init_blocks = (M + 127) // 128
        set_zero_kernel[init_blocks, 128](d_y)
    else:
        d_y = cuda.to_device(np.zeros(M, dtype=np.float64))

    d_done = cuda.to_device(np.zeros(M, dtype=np.int32))
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    # 2. Grid Calculation
    threads_per_block = 128
    warps_per_block = threads_per_block // 32
    
    # Total tasks in the entire matrix
    num_total_tasks = len(d_task_ptr) - 1
    
    # Calculate needed blocks vs hardware capacity
    needed_blocks = (num_total_tasks + warps_per_block - 1) // warps_per_block
    hardware_cap = get_hardware_grid_cap(threads_per_block)
    
    # CLAMP: Never launch more than the hardware can hold resident
    grid_size = min(needed_blocks, hardware_cap)
    grid_size = max(grid_size, 1)

    # 3. Launch Persistent Kernel
    # We pass 'num_total_tasks' as the limit, and '0' as the offset
    sptrsv_csr_syncfree_warp_kernel[grid_size, threads_per_block](
        data, indices, indptr, b, d_y,
        d_h_nodes, d_task_ptr,
        d_done, d_task_counter, 
        num_total_tasks, 0
    )
    
    return d_y

def sptrsv_numba_csr_syncfree_persistent_graph(data, indices, indptr, b, M, N, nnz, 
                                               wavefronts, wave_start, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
    return sptrsv_numba_csr_syncfree_persistent_solve(
        data, indices, indptr, b, M, N, nnz, _ANALYSIS_CACHE[key]
    )

# -------------------------------------------------------------------------
# 9. VARIANT F: Persistent Sync-Free H-Level (CSC)
# -------------------------------------------------------------------------

def sptrsv_numba_csc_syncfree_persistent_solve(data, indices, indptr, b, M, N, nnz, 
                                               analysis_data, in_edge_counts):
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    
    # 1. Memory Setup
    if cuda.is_cuda_array(b):
        d_y = cuda.device_array_like(b)
        d_y.copy_to_device(b)
    else:
        d_y = cuda.to_device(b.copy())
    
    # 2. Dependency Counts (Subtract 1 for diagonal)
    if cuda.is_cuda_array(in_edge_counts):
        counts_host = in_edge_counts.copy_to_host().astype(np.int32)
        d_in_degree = cuda.to_device(counts_host - 1)
    else:
        counts_host = np.asarray(in_edge_counts, dtype=np.int32)
        d_in_degree = cuda.to_device(counts_host - 1)
    
    d_task_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    # 3. Grid Calculation
    threads_per_block = 128
    warps_per_block = threads_per_block // 32
    
    num_total_tasks = len(d_task_ptr) - 1
    
    needed_blocks = (num_total_tasks + warps_per_block - 1) // warps_per_block
    hardware_cap = get_hardware_grid_cap(threads_per_block)
    
    # CLAMP: Prevent deadlock
    grid_size = min(needed_blocks, hardware_cap)
    grid_size = max(grid_size, 1)

    # 4. Launch Persistent Kernel
    sptrsv_csc_syncfree_warp_kernel[grid_size, threads_per_block](
        data, indices, indptr, b, d_y,
        d_h_nodes, d_task_ptr,
        d_in_degree, d_task_counter, 
        num_total_tasks, 0
    )
    
    return d_y

def sptrsv_numba_csc_syncfree_persistent_graph(data, indices, indptr, b, M, N, nnz, 
                                               wavefronts, wave_start, in_edge_counts, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width
        )
    return sptrsv_numba_csc_syncfree_persistent_solve(
        data, indices, indptr, b, M, N, nnz, 
        _ANALYSIS_CACHE[key], in_edge_counts
    )
