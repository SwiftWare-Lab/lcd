import numpy as np
import numba
from numba import prange
import time
import sys
from benchmark.bench_utils import get_matrix_list
import scipy.io as scio

@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_softmax_unfused(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in prange(m):
        sum = 0
        max_val = np.max(Y[j1][:])
        for j2 in range(n):
            Y[j1][j2] -= max_val
        for jj2 in range(n):
            Y[j1][jj2] = np.exp(Y[j1][jj2])
            sum += Y[j1][jj2]
        for jjj2 in range(n):
            Y[j1][jjj2] /= sum
    return Y

@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_softmax_fused_v1(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
        sum = 0
        max_val = np.max(Y[i1][:])
        for j2 in range(n):
            Y[i1][j2] -= max_val
        for jj2 in range(n):
            Y[i1][jj2] = np.exp(Y[i1][jj2])
            sum += Y[i1][jj2]
        for jjj2 in range(n):
            Y[i1][jjj2] /= sum
    return Y

@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_softmax_fused_v2(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
        sum = 0
        max_val = np.max(Y[i1][:])
        for j2 in range(n):
            Y[i1][j2] -= max_val
            Y[i1][j2] = np.exp(Y[i1][j2])
            sum += Y[i1][j2]
        for jjj2 in range(n):
            Y[i1][jjj2] /= sum
    return Y

@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_softmax_fused_v1_array_contracted(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        y = np.zeros(n)
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i3] += Ax[i2] * B[Ai[i2]][i3]
        sum = 0
        max_val = np.max(y[:])
        for j2 in range(n):
            y[j2] -= max_val
        for jj2 in range(n):
            y[jj2] = np.exp(y[jj2])
            sum += y[jj2]
        for jjj2 in range(n):
            y[jjj2] /= sum
        Y[i1][:] = y
    return Y



@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_softmax_fused_v2_array_contracted(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        y = np.zeros(n)
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i3] += Ax[i2] * B[Ai[i2]][i3]
        sum = 0
        max_val = np.max(y[:])
        for j2 in range(n):
            y[j2] -= max_val
            tmp = np.exp(y[j2])
            sum += tmp
        for jjj2 in range(n):
            y[jjj2] /= sum
        Y[i1][:] = y
    return Y


def benchmark(matrix, func, *args, repeat=5):
    times = []
    func(*args)  # Warm-up
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    time_median = np.median(times)
    print(f"{func.__name__},{time_median:.6f},{repeat},{matrix}")
    return time_median

mat_file = sys.argv[1]
mat_folder = sys.argv[2]
# TODO: This is temporary only for testing the decorator.
matrices = get_matrix_list(mat_file, mat_folder)

print('implementation,median_time,repeats,matrix,nnz')
for matrix_path in matrices:
    matrix = scio.mmread(matrix_path)
    matrix_name = matrix_path.split("/")[-1]
    A = matrix.tocsr()
    m = A.shape[0]
    p = A.shape[1]
    n = 128

    Ap = A.indptr
    Ai = A.indices
    Ax = A.data.astype(np.float32)
    nnz = A.nnz
    B = np.random.rand(m, n).astype(np.float32)
    benchmark(matrix_name, spmm_softmax_unfused, Ap, Ai, Ax, B, m, n, repeat=10)
    benchmark(matrix_name, spmm_softmax_fused_v1, Ap, Ai, Ax, B, m, n, repeat=10)
    benchmark(matrix_name, spmm_softmax_fused_v2, Ap, Ai, Ax, B, m, n, repeat=10)
    benchmark(matrix_name, spmm_softmax_fused_v1_array_contracted, Ap, Ai, Ax, B, m, n, repeat=10)
    benchmark(matrix_name, spmm_softmax_fused_v2_array_contracted, Ap, Ai, Ax, B, m, n, repeat=10)