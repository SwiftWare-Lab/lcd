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