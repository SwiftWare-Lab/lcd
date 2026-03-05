
import numpy as np
import torch
import triton
import triton.language as tl
import os

REAL_TYPE = torch.float64
REAL_TYPE_TL = tl.float64
DEVICE = torch.device("cuda:0")

# os.environ["TRITON_INTERPRET"] = "1"
@triton.jit
def sptrsv_sync_free_csr_baseline_kernel(
        data_ptr, indices_ptr, indptr_ptr, map_ptr, map_vals, rhs_vec, y_ptr,
        M, N, nnz, count_dep
):
    row = tl.program_id(axis=0)
    is_valid_row = row < M
    while tl.atomic_add(count_dep + row, 0, mask=is_valid_row) != 1:
        pass
    start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
    rhs_val = tl.load(rhs_vec + row)
    diag = tl.load(data_ptr + end-1)
    y_val =  tl.cast(0.0, REAL_TYPE_TL)
    for i in range(start, end-1, 32):
        nnz_indx = i + tl.arange(0, 32)
        mask = nnz_indx < end - 1
        indices_vec = tl.load(indices_ptr + nnz_indx, mask=mask, other=0)
        y_vec = tl.load(y_ptr + indices_vec, mask=is_valid_row, other=0)
        val_vec = tl.load(data_ptr + nnz_indx, mask=mask, other=0)
        product = y_vec * val_vec
        y_val += tl.sum(product, axis=0)
    y_val = (rhs_val - y_val) / diag
    tl.store(y_ptr + row, y_val)
    start_col = tl.load(map_ptr + row, mask=is_valid_row, other=0)
    end_col = tl.load(map_ptr + row + 1, mask=is_valid_row, other=0)
    for i in range(start_col, end_col, 32):
        nnz_indx = i + tl.arange(0, 32)
        values = tl.load(map_vals + nnz_indx, mask=nnz_indx < end_col, other=M)
        tl.atomic_add(count_dep + values, -1, mask=values < M)


def sptrsv_sync_free_csr_baseline(data, indices, indptr, x, M, N, nnz, dep_counts, map_ptr, map_vals):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    grid = (M,)
    d_dep_count = dep_counts.clone().detach()
    sptrsv_sync_free_csr_baseline_kernel[grid](
        data,
        indices,
        indptr,
        map_ptr, map_vals,
        x,
        y,
        M,
        N,
        nnz,
        d_dep_count)
    return y

@triton.jit
def sptrsv_sync_free_csc_baseline_kernel(
        data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
        M, N, nnz, count_dep, BLOCK_SIZE: tl.constexpr
):
    col = tl.program_id(axis=0)
    is_valid_row = col < M
    while tl.atomic_add(count_dep + col, 0, mask=is_valid_row) != 1:
        pass
    start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)

    # Compute y[col]
    y_col_val = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + start)
    tl.store(y_ptr + col, y_col_val)

    # Vectorized scatter: process BLOCK_SIZE elements at a time
    for i in range(start + 1, end, BLOCK_SIZE):
        nnz_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = nnz_idx < end
        data_col = tl.load(data_ptr + nnz_idx, mask=mask, other=0)
        row_indices = tl.load(indices_ptr + nnz_idx, mask=mask, other=M)
        are_valid_rows = row_indices < M
        tl.atomic_add(y_ptr + row_indices, data_col * y_col_val, mask=are_valid_rows)
        tl.atomic_add(count_dep + row_indices, -1, mask=are_valid_rows)

def sptrsv_sync_free_csc_baseline(data, indices, indptr, x, M, N, nnz, dep_counts):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)
    sptrsv_sync_free_csc_baseline_kernel[grid](
        data,
        indices,
        indptr,
        x,
        y,
        M,
        N,
        nnz,
        count_dep=d_dep_count,
        BLOCK_SIZE=32
    )
    return y

@triton.jit
def sptrsv_wavefront_csr_baseline_kernel(
        data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
        M, N, nnz, wavefront , BLOCK_SIZE: tl.constexpr
):
    row_loc = tl.program_id(axis=0)
    row = tl.load(wavefront + row_loc)
    is_valid_row = row < M

    start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
    rhs_val = tl.load(rhs_vec + row)
    diag = tl.load(data_ptr + end-1)
    y_val =  tl.cast(0.0, REAL_TYPE_TL)
    for i in range(start, end-1, BLOCK_SIZE):
        nnz_indx = i + tl.arange(0, BLOCK_SIZE)
        mask = nnz_indx < end - 1
        indices_vec = tl.load(indices_ptr + nnz_indx, mask=mask, other=0)
        y_vec = tl.load(y_ptr + indices_vec, mask=is_valid_row, other=0)
        val_vec = tl.load(data_ptr + nnz_indx, mask=mask, other=0)
        product = y_vec * val_vec
        y_val += tl.sum(product, axis=0)
    y_val = (rhs_val - y_val) / diag
    tl.store(y_ptr + row, y_val)


def sptrsv_wavefront_csr_baseline(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start)-1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_wavefront_csr_baseline_kernel[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            wavefront=wavefronts[wave_start[i]],
            BLOCK_SIZE=32
        )
    return y

@triton.jit
def sptrsv_wavefront_csc_baseline_kernel(
        data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
        M, N, nnz, wavefront,  BLOCK_SIZE: tl.constexpr
):
    row_loc = tl.program_id(axis=0)
    col = tl.load(wavefront + row_loc)
    is_valid_col = col < M

    start = tl.load(indptr_ptr + col, mask=is_valid_col, other=nnz)
    end = tl.load(indptr_ptr + col + 1, mask=is_valid_col, other=nnz)
    tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)  )/ tl.load(data_ptr + start))
    index_range = tl.arange(0, BLOCK_SIZE)
    x_val = tl.load(y_ptr + col)
    for j in range(start+1, end, BLOCK_SIZE):
        nnz_idx = j + index_range
        indices_vec = tl.load(indices_ptr + nnz_idx, mask=nnz_idx < end, other=M)
        val_vec = tl.load(data_ptr + nnz_idx, mask=nnz_idx < end, other=0)
        tl.atomic_add(y_ptr + indices_vec, val_vec * x_val)

def sptrsv_wavefront_csc_baseline(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    num_waves = len(wave_start)-1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_wavefront_csc_baseline_kernel[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            wavefront=wavefronts[wave_start[i]],
            BLOCK_SIZE=32
        )
    return y