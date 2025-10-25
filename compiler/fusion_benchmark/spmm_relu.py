import numpy as np
import numba
from numba import prange
import time
import sys
from benchmark.bench_utils import get_matrix_list
import scipy.io as scio

@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_relu_unfused(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in prange(m):
        for j2 in range(n):
            if Y[j1][j2] < 0:
                Y[j1][j2] = 0

@numba.njit(numba.float32[:, :](numba.int32[:], numba.int32[:], numba.float32[:], numba.float32[:, :], numba.int64,
                                numba.int64,), cache=True, parallel=True)
def spmm_relu_fused(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
        for j2 in range(n):
            if Y[i1][j2] < 0:
                Y[i1][j2] = 0

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
    benchmark(matrix_name, spmm_relu_unfused, Ap, Ai, Ax, B, m, n, repeat=10)
    benchmark(matrix_name, spmm_relu_fused, Ap, Ai, Ax, B, m, n, repeat=10)