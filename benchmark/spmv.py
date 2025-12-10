import numpy as np
from scipy.sparse import csc_matrix  # For CPU operations
from scipy.sparse.linalg import spsolve  # CPU sparse solver
import cupy as cp  # For GPU operations
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix  # GPU sparse matrix
# import csr_matrix as cp_csr_matrix  # GPU sparse matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix  # GPU sparse matrix
from cupyx.scipy.sparse.linalg import spsolve_triangular as cp_spsolve  # GPU sparse solver
import cupyx.scipy.sparse.linalg as cpx_linalg
import time
import scipy.io as scio
from scipy.sparse import csc_matrix, tril, triu
import cupyx.scipy.sparse as cusparse
import torch

NUM_RUNS=30

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def spmv_kernel(
    data_ptr, indices_ptr, indptr_ptr, x_ptr, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)  # Get the row id based on the program grid
    is_valid_row = row < M

    # Load start and end pointers safely (use `mask` for invalid threads)
    start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
    # define an array of float64 with size 2
    y_val = 0.0

    for i in range(start, end):
        col = tl.load(indices_ptr + i)
        val = tl.load(data_ptr + i)
        x_val = tl.load(x_ptr + col)
        y_val += val * x_val
    tl.store(y_ptr + row, y_val)


@triton.jit
def spmv_csc_kernel(
    data_ptr, indices_ptr, indptr_ptr, x_ptr, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr
):
    col = tl.program_id(axis=0)  # Get the row id based on the program grid
    is_valid_row = col < M

    start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
    end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
    for j in range(start, end):
        row = tl.load(indices_ptr + j)
        val = tl.load(data_ptr + j)
        x_val = tl.load(x_ptr + col)
        tl.atomic_add(y_ptr + row, val * x_val)


def spmv_triton(data, indices, indptr, x, M, N, nnz):
    # Allocate result vector on GPU
    y = torch.zeros(M, dtype=torch.float32, device=DEVICE)

    # Launch Triton kernel
    grid = (M,)  # One thread per row
    spmv_csc_kernel[grid](
        data,
        indices,
        indptr,
        x,
        y,
        M,
        N,
        nnz,
        BLOCK_SIZE=128)
    return y