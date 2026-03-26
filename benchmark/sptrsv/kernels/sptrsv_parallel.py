import numpy as np
import numba
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
from numba.core.errors import NumbaPerformanceWarning
import cupy as cp
import warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

THREADS_PER_BLOCK = 128


def _to_device_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr
    return cuda.to_device(np.asarray(arr, dtype=np.int32))


def _to_host_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr.copy_to_host().astype(np.int32)
    return np.asarray(arr, dtype=np.int32)


# ============================================================================
# 1. WAVEFRONT KERNELS  (level-synchronous, 1 thread = 1 row / 1 col)
# ============================================================================

@cuda.jit
def _csr_wf_thread_kernel(data, indices, indptr, b, y,
                          wavefronts, wave_offset, wave_len):
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if gid >= wave_len:
        return
    row = wavefronts[wave_offset + gid]
    row_start = indptr[row]
    row_end = indptr[row + 1]
    acc = 0.0
    for j in range(row_start, row_end - 1):
        acc += data[j] * y[indices[j]]
    y[row] = (b[row] - acc) / data[row_end - 1]


@cuda.jit
def _csc_wf_thread_kernel(data, indices, indptr, b, y, M,
                          wavefronts, wave_offset, wave_len):
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if gid >= wave_len:
        return
    col = wavefronts[wave_offset + gid]
    if col >= M:
        return
    col_start = indptr[col]
    col_end = indptr[col + 1]
    y[col] = (b[col] - y[col]) / data[col_start]
    y_val = y[col]
    for j in range(col_start + 1, col_end):
        cuda.atomic.add(y, indices[j], data[j] * y_val)


# ---- host wrappers (plain loop + cuda.synchronize) ----

def sptrsv_parallel_csr_wf(data, indices, indptr, b, M, N, nnz,
                           wave_start, wavefronts,
                           threads_per_block=THREADS_PER_BLOCK):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wf = _to_device_int32(wavefronts)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        offset = int(wave_start[i])
        wlen = int(wave_start[i + 1] - wave_start[i])
        if wlen <= 0:
            continue
        blocks = (wlen + threads_per_block - 1) // threads_per_block
        _csr_wf_thread_kernel[blocks, threads_per_block](
            data, indices, indptr, b, d_y,
            d_wf, np.int32(offset), np.int32(wlen))
        cuda.synchronize()
    return d_y


def sptrsv_parallel_csc_wf(data, indices, indptr, b, M, N, nnz,
                           wave_start, wavefronts,
                           threads_per_block=THREADS_PER_BLOCK):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wf = _to_device_int32(wavefronts)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        offset = int(wave_start[i])
        wlen = int(wave_start[i + 1] - wave_start[i])
        if wlen <= 0:
            continue
        blocks = (wlen + threads_per_block - 1) // threads_per_block
        _csc_wf_thread_kernel[blocks, threads_per_block](
            data, indices, indptr, b, d_y, M,
            d_wf, np.int32(offset), np.int32(wlen))
        cuda.synchronize()
    return d_y


# ---- host wrappers (CUDA-graph captured) ----

def sptrsv_parallel_csr_wf_graph(data, indices, indptr, b, M, N, nnz,
                                 wave_start, wavefronts,
                                 threads_per_block=THREADS_PER_BLOCK):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wf = _to_device_int32(wavefronts)
    num_waves = len(wave_start) - 1

    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    cp_stream.begin_capture()

    for i in range(num_waves):
        offset = int(wave_start[i])
        wlen = int(wave_start[i + 1] - wave_start[i])
        if wlen <= 0:
            continue
        blocks = (wlen + threads_per_block - 1) // threads_per_block
        _csr_wf_thread_kernel[blocks, threads_per_block, nb_stream](
            data, indices, indptr, b, d_y,
            d_wf, np.int32(offset), np.int32(wlen))

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    return d_y


def sptrsv_parallel_csc_wf_graph(data, indices, indptr, b, M, N, nnz,
                                 wave_start, wavefronts,
                                 threads_per_block=THREADS_PER_BLOCK):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_wf = _to_device_int32(wavefronts)
    num_waves = len(wave_start) - 1

    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    cp_stream.begin_capture()

    for i in range(num_waves):
        offset = int(wave_start[i])
        wlen = int(wave_start[i + 1] - wave_start[i])
        if wlen <= 0:
            continue
        blocks = (wlen + threads_per_block - 1) // threads_per_block
        _csc_wf_thread_kernel[blocks, threads_per_block, nb_stream](
            data, indices, indptr, b, d_y, M,
            d_wf, np.int32(offset), np.int32(wlen))

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    return d_y


# ============================================================================
# 2. SYNC-FREE KERNELS  (single launch, 1 thread = 1 row / 1 col)
# ============================================================================

# ---- CSR (pull): thread accumulates off-diag, signals successors ----

@cuda.jit
def _csr_sf_thread_kernel(data, indices, indptr, b, y, M,
                          dep_counts, succ_offsets, succ_indices):
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row >= M:
        return
    while cuda.atomic.add(dep_counts, row, 0) > 0:
        pass

    row_start = indptr[row]
    row_end = indptr[row + 1]
    acc = 0.0
    for j in range(row_start, row_end - 1):
        acc += data[j] * y[indices[j]]
    y[row] = (b[row] - acc) / data[row_end - 1]

    cuda.threadfence()

    s0 = succ_offsets[row]
    s1 = succ_offsets[row + 1]
    for p in range(s0, s1):
        cuda.atomic.add(dep_counts, succ_indices[p], -1)


# ---- CSC (push): thread solves diagonal, scatters, decrements deps ----

@cuda.jit
def _csc_sf_thread_kernel(data, indices, indptr, b, y, M, dep_counts):
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if col >= M:
        return
    while cuda.atomic.add(dep_counts, col, 0) != 1:
        pass

    cuda.threadfence()

    col_start = indptr[col]
    col_end = indptr[col + 1]
    y[col] = (b[col] - y[col]) / data[col_start]
    y_val = y[col]

    for j in range(col_start + 1, col_end):
        row_idx = indices[j]
        cuda.atomic.add(y, row_idx, data[j] * y_val)

    cuda.threadfence()

    for j in range(col_start + 1, col_end):
        cuda.atomic.add(dep_counts, indices[j], -1)


# ---- host wrappers ----

def sptrsv_parallel_csr_sf(data, indices, indptr, b, M, N, nnz,
                           in_edges, succ_offsets, succ_indices,
                           threads_per_block=THREADS_PER_BLOCK):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep = cuda.to_device(in_edges.copy())
    d_succ_off = cuda.to_device(succ_offsets)
    d_succ_idx = cuda.to_device(succ_indices)
    blocks = (M + threads_per_block - 1) // threads_per_block
    _csr_sf_thread_kernel[blocks, threads_per_block](
        data, indices, indptr, b, d_y, M,
        d_dep, d_succ_off, d_succ_idx)
    return d_y


def sptrsv_parallel_csc_sf(data, indices, indptr, b, M, N, nnz,
                           in_edges,
                           threads_per_block=THREADS_PER_BLOCK):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep = cuda.to_device(in_edges.copy())
    blocks = (M + threads_per_block - 1) // threads_per_block
    _csc_sf_thread_kernel[blocks, threads_per_block](
        data, indices, indptr, b, d_y, M, d_dep)
    return d_y


# ============================================================================
# 3. DAG-COARSENING (H-LEVEL) KERNELS  (1 thread = 1 chain)
# ============================================================================

# ---- CSR: thread walks the chain row-by-row (pull) ----

@cuda.jit
def _csr_hlevel_thread_kernel(data, indices, indptr, b, y,
                              h_task_nodes, task_ptr,
                              level_start_task, level_end_task):
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    task_idx = level_start_task + gid
    if task_idx >= level_end_task:
        return

    node_start = task_ptr[task_idx]
    node_end = task_ptr[task_idx + 1]

    for i in range(node_start, node_end):
        row = h_task_nodes[i]
        row_start = indptr[row]
        row_end = indptr[row + 1]
        acc = 0.0
        for j in range(row_start, row_end - 1):
            acc += data[j] * y[indices[j]]
        y[row] = (b[row] - acc) / data[row_end - 1]


# ---- CSC: thread walks the chain col-by-col (push) ----

@cuda.jit
def _csc_hlevel_thread_kernel(data, indices, indptr, b, y, M,
                              h_task_nodes, task_ptr,
                              level_start_task, level_end_task):
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    task_idx = level_start_task + gid
    if task_idx >= level_end_task:
        return

    node_start = task_ptr[task_idx]
    node_end = task_ptr[task_idx + 1]

    for i in range(node_start, node_end):
        col = h_task_nodes[i]
        if col >= M:
            continue
        col_start = indptr[col]
        col_end = indptr[col + 1]
        y[col] = (b[col] - y[col]) / data[col_start]
        y_val = y[col]
        for j in range(col_start + 1, col_end):
            cuda.atomic.add(y, indices[j], data[j] * y_val)


# ---- host solve wrappers (analysis_data from sptrsv_h_level_analyze) ----

def sptrsv_parallel_csr_hlevel_solve(data, indices, indptr, b, M, N, nnz,
                                     analysis_data,
                                     threads_per_block=THREADS_PER_BLOCK):
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_h_levels = len(h_level_ptr) - 1

    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    cp_stream.begin_capture()

    for i in range(num_h_levels):
        start_task = int(h_level_ptr[i])
        end_task = int(h_level_ptr[i + 1])
        num_tasks = end_task - start_task
        if num_tasks <= 0:
            continue
        blocks = (num_tasks + threads_per_block - 1) // threads_per_block
        _csr_hlevel_thread_kernel[blocks, threads_per_block, nb_stream](
            data, indices, indptr, b, d_y,
            d_h_nodes, d_task_ptr,
            start_task, end_task)

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    return d_y


def sptrsv_parallel_csc_hlevel_solve(data, indices, indptr, b, M, N, nnz,
                                     analysis_data,
                                     threads_per_block=THREADS_PER_BLOCK):
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_h_levels = len(h_level_ptr) - 1

    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    cp_stream.begin_capture()

    for i in range(num_h_levels):
        start_task = int(h_level_ptr[i])
        end_task = int(h_level_ptr[i + 1])
        num_tasks = end_task - start_task
        if num_tasks <= 0:
            continue
        blocks = (num_tasks + threads_per_block - 1) // threads_per_block
        _csc_hlevel_thread_kernel[blocks, threads_per_block, nb_stream](
            data, indices, indptr, b, d_y, M,
            d_h_nodes, d_task_ptr,
            start_task, end_task)

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    return d_y


# ---- compatibility wrappers (auto-cache analysis, same API as h-level graph) ----

from .sptrsv_hlevel import sptrsv_h_level_analyze, _ANALYSIS_CACHE


def sptrsv_parallel_csr_hlevel_graph(data, indices, indptr, b, M, N, nnz,
                                     wavefronts, wave_start, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width, "parallel")
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width)
    return sptrsv_parallel_csr_hlevel_solve(
        data, indices, indptr, b, M, N, nnz, _ANALYSIS_CACHE[key])


def sptrsv_parallel_csc_hlevel_graph(data, indices, indptr, b, M, N, nnz,
                                     wavefronts, wave_start, band_width=64):
    key = (indptr.device_ctypes_pointer.value, band_width, "parallel")
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = sptrsv_h_level_analyze(
            indptr, indices, wavefronts, wave_start, band_width)
    return sptrsv_parallel_csc_hlevel_solve(
        data, indices, indptr, b, M, N, nnz, _ANALYSIS_CACHE[key])
