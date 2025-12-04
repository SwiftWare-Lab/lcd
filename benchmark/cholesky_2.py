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
# 1. WARP-PARALLEL DOT PRODUCT
# ============================================================================

MAX_NNZ_PER_WARP = 512

@cuda.jit(device=True)
def warp_sparse_dot_product(row_target_ptr, row_target_end,
                            row_source_ptr, row_source_end,
                            indices, data, limit_col):
    """
    Computes sum(L[i,k] * L[j,k]) for k < limit_col using the full Warp.
    Global Memory Version.
    """
    lane = cuda.threadIdx.x % 32
    val = 0.0
    
    # 1. Warp iterates over Source Row (Row j)
    for ptr_b in range(row_source_ptr + lane, row_source_end, 32):
        col_b = indices[ptr_b]
        
        if col_b >= limit_col:
            continue
            
        val_b = data[ptr_b]
        
        # 2. Binary Search in Target Row (Global Memory)
        low = row_target_ptr
        high = row_target_end
        found_idx = -1
        while low < high:
            mid = (low + high) // 2
            col_mid = indices[mid]
            if col_mid < col_b:
                low = mid + 1
            elif col_mid > col_b:
                high = mid
            else:
                found_idx = mid
                break
        
        if found_idx != -1:
            val += data[found_idx] * val_b

    # 4. Warp Reduction
    mask = 0xFFFFFFFF
    val += cuda.shfl_down_sync(mask, val, 16)
    val += cuda.shfl_down_sync(mask, val, 8)
    val += cuda.shfl_down_sync(mask, val, 4)
    val += cuda.shfl_down_sync(mask, val, 2)
    val += cuda.shfl_down_sync(mask, val, 1)
    
    return val

@cuda.jit(device=True)
def warp_sparse_dot_product_shmem(s_indices, s_data, target_len,
                                  row_source_ptr, row_source_end,
                                  indices, data, limit_col):
    """
    Shared Memory Version.
    s_indices, s_data: Warp-local shared memory pointers.
    target_len: Number of non-zeros in target row.
    """
    lane = cuda.threadIdx.x % 32
    val = 0.0
    
    for ptr_b in range(row_source_ptr + lane, row_source_end, 32):
        col_b = indices[ptr_b]
        if col_b >= limit_col: continue
        
        val_b = data[ptr_b]
        
        # Binary Search in Shared Memory
        low = 0
        high = target_len
        found_idx = -1
        while low < high:
            mid = (low + high) // 2
            col_mid = s_indices[mid]
            if col_mid < col_b:
                low = mid + 1
            elif col_mid > col_b:
                high = mid
            else:
                found_idx = mid
                break
                
        if found_idx != -1:
            val += s_data[found_idx] * val_b
            
    mask = 0xFFFFFFFF
    val += cuda.shfl_down_sync(mask, val, 16)
    val += cuda.shfl_down_sync(mask, val, 8)
    val += cuda.shfl_down_sync(mask, val, 4)
    val += cuda.shfl_down_sync(mask, val, 2)
    val += cuda.shfl_down_sync(mask, val, 1)
    return val

# ============================================================================
# 2. WAVEFRONT VARIANT
# ============================================================================

@cuda.jit
def cholesky_wavefront_kernel(data, indices, indptr, wavefronts, wave_start, wave_idx):
    # This remains Thread-Per-Row (Serial Math) because mapping 1-Block-Per-Row
    # for Wavefront requires a different architecture.
    # It will remain slow compared to SyncFree/HLevel.
    tid = cuda.grid(1)
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    
    if tid < (end_wave - start_wave):
        row = wavefronts[start_wave + tid]
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        # Serial implementation
        for ptr in range(row_start, row_end - 1):
            col_j = indices[ptr]
            
            # Serial Dot
            dot = 0.0
            ptr_a = row_start; end_a = row_end
            ptr_b = indptr[col_j]; end_b = indptr[col_j + 1]
            
            while ptr_a < end_a and ptr_b < end_b:
                ca, cb = indices[ptr_a], indices[ptr_b]
                if ca >= col_j or cb >= col_j: break
                if ca < cb: ptr_a += 1
                elif cb < ca: ptr_b += 1
                else:
                    dot += data[ptr_a] * data[ptr_b]
                    ptr_a += 1; ptr_b += 1
            
            diag_j = data[indptr[col_j + 1] - 1]
            data[ptr] = (data[ptr] - dot) / diag_j

        diag_ptr = row_end - 1
        # Serial Dot Self
        dot = 0.0
        ptr_a = row_start; end_a = row_end
        while ptr_a < end_a:
            if indices[ptr_a] >= row: break
            dot += data[ptr_a] * data[ptr_a]
            ptr_a += 1
            
        val = data[diag_ptr] - dot
        data[diag_ptr] = math.sqrt(val) if val > 0 else 1.0

def cholesky_numba_wavefront(data, indices, indptr, wave_start, wavefronts):
    num_waves = len(wave_start) - 1
    threads = 128
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        if grid_len > 0:
            blocks = (grid_len + threads - 1) // threads
            cholesky_wavefront_kernel[blocks, threads](data, indices, indptr, wavefronts, wave_start, i)

# ============================================================================
# 3. SYNC-FREE VARIANT (Optimized)
# ============================================================================

@cuda.jit
def cholesky_syncfree_kernel(data, indices, indptr, csc_indices, csc_indptr, incom_edges, num_rows):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    global_warp_id = (bid * (block_dim // 32)) + warp_id
    total_warps = grid_dim * (block_dim // 32)
    
    # Shared Memory: 4 Warps x 512 Elements (Adjust based on blockDim=128)
    s_indices = cuda.shared.array((4, 512), dtype=numba.int32)
    s_data = cuda.shared.array((4, 512), dtype=numba.float64)
    
    # Persistent Thread Loop
    for row in range(global_warp_id, num_rows, total_warps):
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0: pass
        cuda.syncwarp()
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        row_len = row_end - row_start
        
        # Use Shared Memory if Row fits
        use_shmem = (row_len <= 512)
        
        if use_shmem:
            # Load Row 'i' into Shared Memory
            for i in range(lane_id, row_len, 32):
                s_indices[warp_id, i] = indices[row_start + i]
                s_data[warp_id, i] = data[row_start + i]
            cuda.syncwarp()
        
        # --- PARALLEL MATH ---
        for ptr in range(row_start, row_end - 1):
            
            # Broadcast the column index 'j' we are solving for
            if lane_id == 0: 
                if use_shmem:
                    target_col = s_indices[warp_id, ptr - row_start]
                else:
                    target_col = indices[ptr]
            target_col = cuda.shfl_sync(0xFFFFFFFF, target_col, 0)
            
            src_start = indptr[target_col]
            src_end = indptr[target_col + 1]
            
            # Warp Parallel Dot
            if use_shmem:
                dot = warp_sparse_dot_product_shmem(
                    s_indices[warp_id], s_data[warp_id], row_len,
                    src_start, src_end,
                    indices, data, target_col
                )
            else:
                dot = warp_sparse_dot_product(
                    row_start, row_end,      # Target Row Bounds (i)
                    src_start, src_end,      # Source Row Bounds (j)
                    indices, data, target_col # Limit (k < j)
                )
            
            if lane_id == 0:
                diag_j = data[src_end - 1]
                if use_shmem:
                    local_idx = ptr - row_start
                    val = (s_data[warp_id, local_idx] - dot) / diag_j
                    s_data[warp_id, local_idx] = val
                else:
                    data[ptr] = (data[ptr] - dot) / diag_j
            
            cuda.syncwarp()
            
        # Compute Diagonal
        if use_shmem:
            # Dot Self: Sum squares of all elements except the diagonal itself
            dot = 0.0
            # Diag is at index (row_len - 1)
            for k in range(lane_id, row_len - 1, 32):
                val = s_data[warp_id, k]
                dot += val * val
            
            # Reduction
            mask = 0xFFFFFFFF
            dot += cuda.shfl_down_sync(mask, dot, 16)
            dot += cuda.shfl_down_sync(mask, dot, 8)
            dot += cuda.shfl_down_sync(mask, dot, 4)
            dot += cuda.shfl_down_sync(mask, dot, 2)
            dot += cuda.shfl_down_sync(mask, dot, 1)
            
            if lane_id == 0:
                val = s_data[warp_id, row_len - 1] - dot
                val = math.sqrt(val) if val > 0 else 1.0
                s_data[warp_id, row_len - 1] = val
            
            cuda.syncwarp()
            
            # Write Back
            for i in range(lane_id, row_len, 32):
                data[row_start + i] = s_data[warp_id, i]
            cuda.threadfence()
            
        else:
            dot = warp_sparse_dot_product(
                row_start, row_end,
                row_start, row_end,
                indices, data, row
            )
            
            if lane_id == 0:
                diag_ptr = row_end - 1
                val = data[diag_ptr] - dot
                data[diag_ptr] = math.sqrt(val) if val > 0 else 1.0
                cuda.threadfence() # Commit globally
        
        cuda.syncwarp()

        # Parallel Signal
        sig_start = csc_indptr[row]
        sig_end = csc_indptr[row+1]
        for p in range(sig_start + lane_id, sig_end, 32):
            cuda.atomic.add(incom_edges, csc_indices[p], -1)

def cholesky_numba_syncfree(data, indices, indptr, csc_indices, csc_indptr, incom_edges):
    M = len(indptr) - 1
    # Increase blocks to saturate GPU
    cholesky_syncfree_kernel[1024, 128](data, indices, indptr, csc_indices, csc_indptr, incom_edges, M)

# ============================================================================
# 4. H-LEVEL VARIANT (Optimized)
# ============================================================================

# ... (Analysis functions _find_root, _union, build_cholesky_h_level, cholesky_h_level_analyze - NO CHANGE) ...
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
def build_cholesky_h_level(indptr, indices, wavefronts, wave_start, band_width=64):
    num_waves = len(wave_start) - 1
    n_rows = len(indptr) - 1
    h_nodes = np.zeros(len(wavefronts), dtype=np.int32)
    task_ptr = np.zeros(len(wavefronts) + 1, dtype=np.int32)
    h_level_ptr = np.zeros(num_waves + 1, dtype=np.int32)
    global_to_local = np.full(n_rows, -1, dtype=np.int32)
    
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
        for i in range(band_size):
            row = wavefronts[start_idx + i]
            for ptr in range(indptr[row], indptr[row+1]):
                col = indices[ptr]
                local_col = global_to_local[col]
                if local_col != -1 and local_col != i: _union(parent, i, local_col)
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

def cholesky_h_level_analyze(indptr, indices, wavefronts, wave_start, band_width=64):
    indptr_h = _to_host_int32(indptr)
    indices_h = _to_host_int32(indices)
    waves_h = _to_host_int32(wavefronts)
    wave_start_h = _to_host_int32(wave_start)
    h_nodes, task_ptr, level_ptr = build_cholesky_h_level(
        indptr_h, indices_h, waves_h, wave_start_h, band_width)
    return (_to_device_int32(h_nodes), _to_device_int32(task_ptr), level_ptr)

@cuda.jit
def cholesky_hlevel_kernel(data, indices, indptr, h_task_nodes, task_ptr, level_start, level_end, sub_warp_size):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    # HARDCODED: 1 Chain = 1 Warp (32 threads) for simplicity
    # Changing this requires updating the reduction logic inside warp_sparse_dot_product
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = cuda.blockDim.x // 32
    global_group = (bid * warps_per_block) + warp_id
    
    task_idx = level_start + global_group
    if task_idx >= level_end: return

    start_node = task_ptr[task_idx]
    end_node = task_ptr[task_idx + 1]
    
    for i in range(start_node, end_node):
        row = h_task_nodes[i]
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        # Loop Columns
        for ptr in range(row_start, row_end - 1):
            if lane_id == 0: target_col = indices[ptr]
            target_col = cuda.shfl_sync(0xFFFFFFFF, target_col, 0)
            
            src_start = indptr[target_col]
            src_end = indptr[target_col + 1]
            
            # Parallel Dot
            dot = warp_sparse_dot_product(
                row_start, row_end,
                src_start, src_end,
                indices, data, target_col
            )
            
            if lane_id == 0:
                diag_j = data[src_end - 1]
                data[ptr] = (data[ptr] - dot) / diag_j
            
            cuda.syncwarp() # Wait for data[ptr] write
            
        # Diagonal
        dot = warp_sparse_dot_product(
            row_start, row_end,
            row_start, row_end,
            indices, data, row
        )
        if lane_id == 0:
            diag_ptr = row_end - 1
            val = data[diag_ptr] - dot
            data[diag_ptr] = math.sqrt(val) if val > 0 else 1.0
            
        cuda.syncwarp()

def cholesky_numba_hlevel_solve(data, indices, indptr, analysis_data, sub_warp_size=32):
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    # Force 32 for Cholesky
    sub_warp_size = 32
    threads_per_block = 128
    chains_per_block = 4
    num_h_levels = len(h_level_ptr) - 1
    
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    
    cp_stream.begin_capture()
    for i in range(num_h_levels):
        start, end = h_level_ptr[i], h_level_ptr[i+1]
        num_tasks = end - start
        if num_tasks > 0:
            grid = (num_tasks + chains_per_block - 1) // chains_per_block
            cholesky_hlevel_kernel[grid, 128, nb_stream](
                data, indices, indptr, d_h_nodes, d_task_ptr, start, end, sub_warp_size)
    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()