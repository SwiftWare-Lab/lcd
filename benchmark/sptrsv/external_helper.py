# external_benchmarks.py
import math
import time
import numpy as np
import cupy as cp
from numba import cuda
from scipy.sparse import tril

import yysptrsv_wrapper
import cusparse_sptrsv


def matrix_warp_numpy(m, csrRowPtr, border, WARP_SIZE=32):
    csrRowPtr = np.asarray(csrRowPtr)
    n_blocks = (m + WARP_SIZE - 1) // WARP_SIZE
    block_starts = np.arange(0, m, WARP_SIZE, dtype=np.int32)
    block_ends = np.minimum(block_starts + WARP_SIZE, m)

    element_n = csrRowPtr[block_ends] - csrRowPtr[block_starts]
    rows_per_block = block_ends - block_starts
    avg_element_n = element_n / rows_per_block
    dense_mask = avg_element_n >= border

    warp_num = np.zeros(m + 1, dtype=np.int32)
    row = 0
    k = 1

    for b in range(n_blocks):
        block_size = rows_per_block[b]
        if dense_mask[b]:
            for _ in range(block_size):
                row += 1
                warp_num[k] = row
                k += 1
        else:
            row += block_size
            warp_num[k] = row
            k += 1

    return warp_num[:k].astype(np.int32)


def bench_yysptrsv(A_csr, rep=100, warmup=25, border=10):
    m = A_csr.shape[0]
    b = np.ones(m, dtype=np.float64)

    d_data_csr    = cp.asarray(A_csr.data)
    d_indices_csr = cp.asarray(A_csr.indices)
    d_indptr_csr  = cp.asarray(A_csr.indptr)
    d_b           = cp.asarray(b)

    warp_num_host = matrix_warp_numpy(
        m=m,
        csrRowPtr=A_csr.indptr,
        border=border,
        WARP_SIZE=32
    )
    len_add = len(warp_num_host)
    d_warp_num = cp.asarray(warp_num_host)

    d_x = cp.empty(m, dtype=A_csr.data.dtype)
    d_get_value = cp.zeros(m, dtype=np.int32)
    d_id_extractor = cp.empty(1, dtype=np.int32)

    WARP_SIZE = 32
    WARP_PER_BLOCK = 32
    num_threads = WARP_PER_BLOCK * WARP_SIZE
    num_blocks = math.ceil(((len_add - 1) * WARP_SIZE) / num_threads)

    def run_once():
        d_get_value[:] = 0
        d_x[:] = 0
        d_id_extractor[:] = 0

        yysptrsv_wrapper.YYSpTRSV(
            d_indptr_csr.data.ptr,
            d_indices_csr.data.ptr,
            d_data_csr.data.ptr,
            d_get_value.data.ptr,
            m,
            A_csr.nnz,
            d_b.data.ptr,
            d_x.data.ptr,
            0,
            d_warp_num.data.ptr,
            len_add,
            d_id_extractor.data.ptr,
            num_threads,
            num_blocks
        )
        cuda.synchronize()

    for _ in range(warmup):
        run_once()

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times = np.array(times)
    return np.median(times), np.min(times), np.max(times)


def bench_cusparse_wrapper(A_csr, rep=100, warmup=25):
    rowptr = A_csr.indptr.astype(np.int32)
    colind = A_csr.indices.astype(np.int32)
    values = A_csr.data.astype(np.float64)

    solver = cusparse_sptrsv.CuSparseSpTRSV(
        rowptr, colind, values, lower=True, unit_diag=False
    )

    solver.analyze()

    for _ in range(warmup):
        solver.solve()

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        solver.solve()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times = np.array(times)
    return np.median(times), np.min(times), np.max(times)