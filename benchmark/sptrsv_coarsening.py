import numpy as np
import numba
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import triton
import triton.language as tl

from numba import cuda
from numba import config
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
from numba.core.errors import NumbaPerformanceWarning

config.CUDA_ENABLE_PYNVJITLINK = 1
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from bench_utils import compute_wavefront


REAL_TYPE = torch.float64
REAL_TYPE_TL = tl.float64
DEVICE = torch.device("cuda:0")


def _validate_threads(threads_per_block):
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
    max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    if threads_per_block > max_threads:
        raise ValueError(f"threads_per_block must be <= {max_threads}")


def _to_device_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return cuda.to_device(np.asarray(arr, dtype=np.int32))


def _to_wavefront_tensor(wavefronts):
    if not isinstance(wavefronts, torch.Tensor):
        tensor = torch.tensor(wavefronts, dtype=torch.int32, device=DEVICE)
    else:
        tensor = wavefronts.to(device=DEVICE)
        if tensor.dtype != torch.int32:
            tensor = tensor.to(dtype=torch.int32)
    return tensor.contiguous()


def coarsen_wavefronts(wavefront_list, coarsening_factor):
    if coarsening_factor <= 1:
        return wavefront_list
    
    coarsened = []
    for i in range(0, len(wavefront_list), coarsening_factor):
        end_idx = min(i + coarsening_factor, len(wavefront_list))
        merged_wave = np.concatenate(wavefront_list[i:end_idx])
        coarsened.append(merged_wave)
    
    return coarsened


def compute_coarsened_wavefronts(L, coarsening_factor):
    wavefront_list = compute_wavefront(L)
    return coarsen_wavefronts(wavefront_list, coarsening_factor)


@cuda.jit
def sptrsv_numba_csc_coarsened_kernel(data, indices, indptr, b, y, M, N, nnz, wavefronts, wave_offset, wave_len):
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    for pos in range(wave_len):
        col = wavefronts[wave_offset + pos]
        if col >= M:
            continue
        col_start = indptr[col]
        col_end = indptr[col + 1]

        if tid == 0:
            y[col] = (b[col] - y[col]) / data[col_start]
        cuda.syncthreads()

        y_col = y[col]
        for idx in range(col_start + 1 + tid, col_end, block_size):
            row_idx = indices[idx]
            val = data[idx]
            cuda.atomic.add(y, row_idx, val * y_col)
        cuda.syncthreads()


def sptrsv_numba_csc_coarsened(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor=2, threads_per_block=128):
    _validate_threads(threads_per_block)

    wave_start = np.asarray(wave_start, dtype=np.int32)
    num_waves = len(wave_start) - 1
    if num_waves <= 0:
        return cuda.to_device(np.zeros(M, dtype=np.float64))

    d_wavefronts = _to_device_int32(wavefronts)
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))

    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0:
            continue
        sptrsv_numba_csc_coarsened_kernel[1, threads_per_block](
            data, indices, indptr, b, d_y, M, N, nnz, d_wavefronts, np.int32(offset), np.int32(wave_len))
        cuda.synchronize()
    return d_y


@cuda.jit
def sptrsv_numba_csr_coarsened_kernel(data, indices, indptr, b, y, M, N, nnz, wavefronts, wave_offset, wave_len):
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    shared_sum = cuda.shared.array(0, dtype=numba.float64)

    for pos in range(wave_len):
        row = wavefronts[wave_offset + pos]
        if row >= M:
            continue
        row_start = indptr[row]
        row_end = indptr[row + 1]

        partial = 0.0
        for idx in range(row_start + tid, row_end - 1, block_size):
            partial += data[idx] * y[indices[idx]]

        shared_sum[tid] = partial
        cuda.syncthreads()

        s = block_size // 2
        while s > 0:
            if tid < s:
                shared_sum[tid] += shared_sum[tid + s]
            cuda.syncthreads()
            s //= 2

        if tid == 0:
            diag = data[row_end - 1]
            y[row] = (b[row] - shared_sum[0]) / diag
        cuda.syncthreads()


def sptrsv_numba_csr_coarsened(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor=2, threads_per_block=128):
    _validate_threads(threads_per_block)

    wave_start = np.asarray(wave_start, dtype=np.int32)
    num_waves = len(wave_start) - 1
    if num_waves <= 0:
        return cuda.to_device(np.zeros(M, dtype=np.float64))

    d_wavefronts = _to_device_int32(wavefronts)
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))

    shared_mem_bytes = int(threads_per_block * 8)
    for i in range(num_waves):
        offset = int(wave_start[i])
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0:
            continue
        sptrsv_numba_csr_coarsened_kernel[1, threads_per_block, 0, shared_mem_bytes](
            data, indices, indptr, b, d_y, M, N, nnz, d_wavefronts, np.int32(offset), np.int32(wave_len))
        cuda.synchronize()
    return d_y


def sptrsv_numba_csc_coarsened_mapping(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor=2, threads_per_block=128):
    return sptrsv_numba_csc_coarsened(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts,
        coarsening_factor=coarsening_factor, threads_per_block=threads_per_block)


def sptrsv_numba_csr_coarsened_mapping(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor=2, threads_per_block=128):
    return sptrsv_numba_csr_coarsened(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts,
        coarsening_factor=coarsening_factor, threads_per_block=threads_per_block)


def sptrsv_numba_csr_coarsened_mapping(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts, coarsening_factor=2, threads_per_block=128):
    return sptrsv_numba_csr_coarsened(
        data, indices, indptr, b, M, N, nnz, wave_start, wavefronts,
        coarsening_factor=coarsening_factor, threads_per_block=threads_per_block)


@triton.jit
def sptrsv_triton_csc_coarsened_kernel(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, wavefronts, wave_offset, wave_len
):
    row_loc = tl.program_id(axis=0)
    if row_loc >= wave_len:
        return
    col = tl.load(wavefronts + wave_offset + row_loc)
    is_valid_row = col < M

    if is_valid_row:
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
        tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + start))
        for j in range(start+1, end):
            val = tl.load(data_ptr + j)
            x_val = tl.load(y_ptr + col)
            r_i = tl.load(indices_ptr + j)
            tl.atomic_add(y_ptr + r_i, val * x_val)


def sptrsv_triton_csc_coarsened(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts, coarsening_factor=2):
    wave_start = np.asarray(wave_start, dtype=np.int32)
    wavefront_tensor = _to_wavefront_tensor(wavefronts)
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0:
            continue
        grid = (wave_len,)
        offset = int(wave_start[i])
        sptrsv_triton_csc_coarsened_kernel[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            BLOCK_SIZE=128,
            wavefronts=wavefront_tensor,
            wave_offset=np.int32(offset),
            wave_len=np.int32(wave_len))
    return y


@triton.jit
def sptrsv_triton_csr_coarsened_kernel(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, wavefronts, wave_offset, wave_len
):
    row_loc = tl.program_id(axis=0)
    if row_loc >= wave_len:
        return
    row = tl.load(wavefronts + wave_offset + row_loc)
    is_valid_row = row < M

    if is_valid_row:
        start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
        
        rhs_val = tl.load(rhs_vec + row)
        diag = tl.load(data_ptr + end - 1)
        y_val = tl.cast(0.0, REAL_TYPE_TL)
        
        for j in range(start, end - 1):
            val = tl.load(data_ptr + j)
            col_idx = tl.load(indices_ptr + j)
            x_val = tl.load(y_ptr + col_idx)
            y_val += val * x_val
        
        y_val = (rhs_val - y_val) / diag
        tl.store(y_ptr + row, y_val)


def sptrsv_triton_csr_coarsened(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts, coarsening_factor=2):
    wave_start = np.asarray(wave_start, dtype=np.int32)
    wavefront_tensor = _to_wavefront_tensor(wavefronts)
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        wave_len = int(wave_start[i + 1] - wave_start[i])
        if wave_len <= 0:
            continue
        grid = (wave_len,)
        offset = int(wave_start[i])
        sptrsv_triton_csr_coarsened_kernel[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            wavefronts=wavefront_tensor,
            wave_offset=np.int32(offset),
            wave_len=np.int32(wave_len))
    return y


def compact_2d_list(wavefront_list):
    waves = [element for sublist in wavefront_list for element in sublist]
    waves = np.array(waves, dtype=np.int32)
    wave_start_loc = np.zeros(len(wavefront_list) + 1, dtype=np.int32)
    wave_start_loc[0] = 0
    for i in range(1, len(wavefront_list)+1):
        wave_start_loc[i] = wave_start_loc[i - 1] + len(wavefront_list[i - 1])
    return wave_start_loc, waves

