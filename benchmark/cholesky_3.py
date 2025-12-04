import numpy as np
import numba
from numba import cuda, njit
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
import cupy as cp
import math

# ============================================================================
# 0. HELPERS
# ============================================================================

def _to_device_int32(arr):
    return cuda.to_device(np.asarray(arr, dtype=np.int32))

def _to_host_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr.copy_to_host().astype(np.int32)
    return np.asarray(arr, dtype=np.int32)

# ============================================================================
# 1. DEVICE FUNCTIONS (Binary Search & AXPY)
# ============================================================================

MAX_NNZ_PER_WARP = 512

@cuda.jit(device=True)
def binary_search(indices, start, end, val):
    """Returns index of 'val' in indices[start:end], or -1 if not found."""
    low = start
    high = end
    
    while low < high:
        mid = (low + high) // 2
        curr = indices[mid]
        if curr < val:
            low = mid + 1
        elif curr > val:
            high = mid
        else:
            return mid
    return -1

@cuda.jit(device=True)
def binary_search_shmem(s_indices, length, val):
    low = 0
    high = length
    while low < high:
        mid = (low + high) // 2
        curr = s_indices[mid]
        if curr < val: low = mid + 1
        elif curr > val: high = mid
        else: return mid
    return -1

@cuda.jit(device=True)
def warp_sparse_axpy(target_col, target_start, target_end,
                     source_col, source_start, source_end,
                     indices, data, multiplier):
    """
    Performs: Target_Col -= Multiplier * Source_Col
    Parallelized across the Warp.
    """
    lane = cuda.threadIdx.x % 32
    
    # Iterate Source Column in chunks
    for p_src in range(source_start + lane, source_end, 32):
        row_idx = indices[p_src]
        
        val_src = data[p_src]
        update = val_src * multiplier
        
        # Find where to apply this update in Target Column
        target_idx = binary_search(indices, target_start, target_end, row_idx)
        
        if target_idx != -1:
            data[target_idx] -= update

@cuda.jit(device=True)
def warp_sparse_axpy_shmem(s_indices, s_data, target_len,
                           source_col, source_start, source_end,
                           indices, data, multiplier):
    lane = cuda.threadIdx.x % 32
    
    for p_src in range(source_start + lane, source_end, 32):
        row_idx = indices[p_src]
        val_src = data[p_src]
        update = val_src * multiplier
        
        target_idx = binary_search_shmem(s_indices, target_len, row_idx)
        
        if target_idx != -1:
            s_data[target_idx] -= update

# ============================================================================
# 2. WAVEFRONT VARIANT (Left-Looking)
# ============================================================================

@cuda.jit
def cholesky_left_wavefront_kernel(data, indices, indptr, 
                                   lt_indices, lt_indptr, 
                                   wavefronts, wave_start, wave_idx):
    """
    Wavefront Left-Looking.
    1 Block per Column (to use Warp/Block parallelism for AXPY).
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_cols_in_wave = end_wave - start_wave
    
    # Grid Mapping: Block ID -> Column in Wave
    if bid < num_cols_in_wave:
        col = wavefronts[start_wave + bid]
        
        col_start = indptr[col]
        col_end = indptr[col + 1]
        
        # 1. Identify Parents (Columns k < col that update col)
        # These are found in the Row Structure of L (which is L.T)
        dep_start = lt_indptr[col]
        dep_end = lt_indptr[col + 1]
        
        # Iterate Parents Sequentially
        for p in range(dep_start, dep_end):
            k = lt_indices[p]
            
            # Look up multiplier L[col, k]
            # This is found in Column k at Row col.
            src_start = indptr[k]
            src_end = indptr[k + 1]
            
            # Determine multiplier (only Lane 0 needs to search)
            multiplier = 0.0
            found_idx = -1
            
            if tid == 0:
                found_idx = binary_search(indices, src_start, src_end, col)
                if found_idx != -1:
                    multiplier = data[found_idx]
            
            # Broadcast multiplier to block
            found_idx = cuda.shfl_sync(0xFFFFFFFF, found_idx, 0)
            multiplier = cuda.shfl_sync(0xFFFFFFFF, multiplier, 0)
            
            if found_idx != -1:
                # Apply AXPY using Block/Warp
                # Note: warp_sparse_axpy uses lane%32.
                # If we use a whole block, we need to stride by blockDim.
                # Re-implementing block-stride AXPY inline:
                
                for p_src in range(src_start + tid, src_end, 128): # 128 = blockDim
                    row_idx = indices[p_src]
                    # Only update lower triangle part (row > col)
                    # The multiplier L[col, k] itself is at row=col. 
                    # We subtract L[row, k]*L[col, k] from L[row, col].
                    if row_idx <= col: continue
                        
                    val_src = data[p_src]
                    update = val_src * multiplier
                    
                    tgt_idx = binary_search(indices, col_start, col_end, row_idx)
                    if tgt_idx != -1:
                        data[tgt_idx] -= update
                        
            cuda.syncthreads() # Finish this parent before next

        # 2. Finalize Diagonal
        if tid == 0:
            diag_idx = binary_search(indices, col_start, col_end, col)
            if diag_idx != -1:
                val = data[diag_idx]
                if val > 0:
                    val = math.sqrt(val)
                    data[diag_idx] = val
                    # Scale column
                    # This is serial in lane 0 (simple loop)
                    # For massive scaling, we could parallelize, but diag is usually 1
                    for ptr in range(diag_idx + 1, col_end):
                        data[ptr] /= val
                        
def cholesky_left_wavefront_solve(data, indices, indptr, lt_indices, lt_indptr, wave_start, wavefronts):
    num_waves = len(wave_start) - 1
    threads = 128
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        if grid_len > 0:
            # 1 Block per Column
            cholesky_left_wavefront_kernel[grid_len, threads](
                data, indices, indptr, lt_indices, lt_indptr, wavefronts, wave_start, i
            )

# ============================================================================
# 2. SYNC-FREE VARIANT (Left-Looking)
# ============================================================================

@cuda.jit
def cholesky_left_syncfree_kernel(data, indices, indptr, 
                                  lt_indices, lt_indptr, 
                                  incom_edges, num_cols):
    """
    1 Warp per Column.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    global_warp_id = (bid * (block_dim // 32)) + warp_id
    total_warps = (cuda.gridDim.x * (block_dim // 32))
    
    # Shared Memory: 4 Warps x 512 Elements
    s_indices = cuda.shared.array((4, 512), dtype=numba.int32)
    s_data = cuda.shared.array((4, 512), dtype=numba.float64)
    
    for col in range(global_warp_id, num_cols, total_warps):
        
        # 1. Wait for Parents
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, col, 0) != 0:
                pass
        cuda.syncwarp()
        
        col_start = indptr[col]
        col_end = indptr[col + 1]
        col_len = col_end - col_start
        
        use_shmem = (col_len <= 512)
        
        if use_shmem:
            # Load Column 'j' into Shared Memory
            for i in range(lane_id, col_len, 32):
                s_indices[warp_id, i] = indices[col_start + i]
                s_data[warp_id, i] = data[col_start + i]
            cuda.syncwarp()
        
        # 2. Iterate Parents (Columns k < col)
        dep_start = lt_indptr[col]
        dep_end = lt_indptr[col + 1]
        
        for p in range(dep_start, dep_end):
            k = lt_indices[p]
            if k >= col: continue 
            
            src_start = indptr[k]
            src_end = indptr[k + 1]
            
            # Find multiplier L[col, k]
            multiplier = 0.0
            found = 0
            if lane_id == 0:
                idx = binary_search(indices, src_start, src_end, col)
                if idx != -1:
                    multiplier = data[idx]
                    found = 1
            
            found = cuda.shfl_sync(0xFFFFFFFF, found, 0)
            multiplier = cuda.shfl_sync(0xFFFFFFFF, multiplier, 0)
            
            if found:
                # Parallel AXPY
                if use_shmem:
                    warp_sparse_axpy_shmem(s_indices[warp_id], s_data[warp_id], col_len,
                                           k, src_start, src_end,
                                           indices, data, multiplier)
                else:
                    warp_sparse_axpy(col, col_start, col_end,
                                     k, src_start, src_end,
                                     indices, data, multiplier)
            
            cuda.syncwarp() 
            
        # 3. Finalize Diagonal & Scale
        if use_shmem:
             if lane_id == 0:
                 diag_idx = binary_search_shmem(s_indices[warp_id], col_len, col)
                 if diag_idx != -1:
                     val = s_data[warp_id, diag_idx]
                     val = math.sqrt(val) if val > 0 else 1.0
                     s_data[warp_id, diag_idx] = val
                     
                     # Scale rest of column
                     for ptr in range(diag_idx + 1, col_len):
                         s_data[warp_id, ptr] /= val
             cuda.syncwarp()
             
             # Write Back
             for i in range(lane_id, col_len, 32):
                 data[col_start + i] = s_data[warp_id, i]
             cuda.threadfence()
             
        else:
            if lane_id == 0:
                diag_idx = binary_search(indices, col_start, col_end, col)
                if diag_idx != -1:
                    val = data[diag_idx]
                    val = math.sqrt(val) if val > 0 else 1.0
                    data[diag_idx] = val
                    
                    for ptr in range(diag_idx + 1, col_end):
                        data[ptr] /= val
            
            cuda.threadfence() 
            
        cuda.syncwarp()
        
        # 4. Signal Children (Parallelized)
        for ptr in range(col_start + lane_id, col_end, 32):
            child_row = indices[ptr]
            if child_row > col:
                cuda.atomic.add(incom_edges, child_row, -1)

def cholesky_left_syncfree_solve(data, indices, indptr, lt_indices, lt_indptr, incom_edges):
    M = len(indptr) - 1
    # Saturation grid
    cholesky_left_syncfree_kernel[1024, 128](
        data, indices, indptr, lt_indices, lt_indptr, incom_edges, M
    )

# ============================================================================
# 3. H-LEVEL VARIANT (Left-Looking)
# ============================================================================

@njit
def _find_root(parent, i):
    root = i
    while parent[root] != root: root = parent[root]
    curr = i
    while curr != root: nxt = parent[curr]; parent[curr] = root; curr = nxt
    return root

@njit
def _union(parent, i, j):
    root_i = _find_root(parent, i); root_j = _find_root(parent, j)
    if root_i != root_j: parent[root_i] = root_j

@njit
def build_left_h_level(lt_indptr, lt_indices, wavefronts, wave_start, band_width=64):
    """
    Builds chains for Left-Looking.
    Dependency: Column 'j' depends on Column 'k' if L[j, k] != 0.
    In L^T (lt_indices), row 'j' contains 'k'.
    So we iterate lt_indices[j] to find dependencies.
    """
    num_waves = len(wave_start) - 1
    n_cols = len(lt_indptr) - 1
    
    h_nodes = np.zeros(len(wavefronts), dtype=np.int32)
    task_ptr = np.zeros(len(wavefronts) + 1, dtype=np.int32)
    h_level_ptr = np.zeros(num_waves + 1, dtype=np.int32)
    global_to_local = np.full(n_cols, -1, dtype=np.int32)
    
    node_cnt = 0; task_cnt = 0; level_cnt = 0
    task_ptr[0] = 0; h_level_ptr[0] = 0
    curr_wave = 0
    
    while curr_wave < num_waves:
        end_wave = min(curr_wave + band_width, num_waves)
        start_idx = wave_start[curr_wave]
        end_idx = wave_start[end_wave]
        band_size = end_idx - start_idx
        if band_size == 0: curr_wave = end_wave; continue

        parent = np.arange(band_size, dtype=np.int32)
        for i in range(band_size): global_to_local[wavefronts[start_idx + i]] = i
        
        # Analyze Dependencies using L^T (lt_indices)
        for i in range(band_size):
            col = wavefronts[start_idx + i]
            # Look at parents (previous columns k)
            for ptr in range(lt_indptr[col], lt_indptr[col+1]):
                k = lt_indices[ptr]
                if k < col: # Parent
                    local_k = global_to_local[k]
                    # If parent is in same band, merge
                    if local_k != -1 and local_k != i:
                        _union(parent, i, local_k)
                        
        for i in range(band_size): parent[i] = _find_root(parent, i)
        
        processed = np.zeros(band_size, dtype=np.int8)
        for i in range(band_size):
            if processed[i]: continue
            root = parent[i]
            for k in range(i, band_size):
                if parent[k] == root:
                    h_nodes[node_cnt] = wavefronts[start_idx + k]
                    node_cnt += 1; processed[k] = 1
            task_cnt += 1; task_ptr[task_cnt] = node_cnt
            
        for i in range(band_size): global_to_local[wavefronts[start_idx + i]] = -1
        level_cnt += 1; h_level_ptr[level_cnt] = task_cnt
        curr_wave = end_wave
        
    return h_nodes[:node_cnt], task_ptr[:task_cnt+1], h_level_ptr[:level_cnt+1]

def cholesky_left_h_level_analyze(lt_indptr, lt_indices, wavefronts, wave_start, band_width=64):
    lt_ptr_h = _to_host_int32(lt_indptr)
    lt_idx_h = _to_host_int32(lt_indices)
    waves_h = _to_host_int32(wavefronts)
    wave_start_h = _to_host_int32(wave_start)
    return build_left_h_level(lt_ptr_h, lt_idx_h, waves_h, wave_start_h, band_width)

@cuda.jit
def cholesky_left_hlevel_kernel(data, indices, indptr, 
                                lt_indices, lt_indptr,
                                h_task_nodes, task_ptr, 
                                level_start, level_end):
    # 1 Warp = 1 Chain
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    warp_id = tid // 32
    lane_id = tid % 32
    global_group = (bid * (cuda.blockDim.x // 32)) + warp_id
    
    task_idx = level_start + global_group
    if task_idx >= level_end: return

    start_node = task_ptr[task_idx]
    end_node = task_ptr[task_idx + 1]
    
    # Iterate Chain
    for i in range(start_node, end_node):
        col = h_task_nodes[i]
        
        col_start = indptr[col]
        col_end = indptr[col + 1]
        
        # Iterate Parents
        dep_start = lt_indptr[col]
        dep_end = lt_indptr[col + 1]
        
        for p in range(dep_start, dep_end):
            k = lt_indices[p]
            if k >= col: continue
            
            src_start = indptr[k]
            src_end = indptr[k + 1]
            
            multiplier = 0.0
            found = 0
            if lane_id == 0:
                idx = binary_search(indices, src_start, src_end, col)
                if idx != -1:
                    multiplier = data[idx]
                    found = 1
            
            found = cuda.shfl_sync(0xFFFFFFFF, found, 0)
            multiplier = cuda.shfl_sync(0xFFFFFFFF, multiplier, 0)
            
            if found:
                warp_sparse_axpy(col, col_start, col_end,
                                 k, src_start, src_end,
                                 indices, data, multiplier)
            
            cuda.syncwarp()
            
        if lane_id == 0:
            diag_idx = binary_search(indices, col_start, col_end, col)
            if diag_idx != -1:
                val = data[diag_idx]
                val = math.sqrt(val) if val > 0 else 1.0
                data[diag_idx] = val
                for ptr in range(diag_idx + 1, col_end):
                    data[ptr] /= val
                    
        cuda.syncwarp()

def cholesky_left_hlevel_solve(data, indices, indptr, lt_indices, lt_indptr, analysis_data):
    # Unpack analysis tuple (3 elements: nodes, ptr, levels)
    # Note: build_left_h_level returns HOST arrays. We need to move them to device here or before.
    # To keep this function pure, assume they are already Device Arrays or moved here.
    
    h_nodes_host, task_ptr_host, h_level_ptr = analysis_data
    
    d_h_nodes = _to_device_int32(h_nodes_host)
    d_task_ptr = _to_device_int32(task_ptr_host)
    
    num_h_levels = len(h_level_ptr) - 1
    threads = 128
    chains_per_block = 4
    
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    for i in range(num_h_levels):
        start, end = h_level_ptr[i], h_level_ptr[i+1]
        num_tasks = end - start
        if num_tasks > 0:
            grid = (num_tasks + chains_per_block - 1) // chains_per_block
            cholesky_left_hlevel_kernel[grid, 128, nb_stream](
                data, indices, indptr, lt_indices, lt_indptr,
                d_h_nodes, d_task_ptr, start, end
            )
    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
