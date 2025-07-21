import numba
from decorator import fusion
import numpy as np
import scipy.io as scio
import time
from benchmark.bench_utils import get_matrix_list
from numba import prange

function_props = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
}
@numba.njit(numba.float32[:, :](numba.int64, numba.int64, numba.int64, numba.int64, numba.int32[:], numba.int32[:],
                          numba.float32[:, :], numba.float32[:, :], numba.float32[:, :]), cache=True, parallel=True)
def sddmm_spmm_unfused(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in prange(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1][j3] += Xx[j2] * D[Ai[j2]][j3]
    return Y

@numba.njit(numba.float32[:, :](numba.int64, numba.int64, numba.int64, numba.int64, numba.int32[:], numba.int32[:], numba.float32[:, :], numba.float32[:, :], numba.float32[:, :]), cache=True, parallel=True)
@fusion(transformation=[("i1", "j1")], function_props=function_props)
def sddmm_spmm_fused_v1(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in prange(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1][j3] += Xx[j2] * D[Ai[j2]][j3]
    return Y

@numba.njit(numba.float32[:, :](numba.int64, numba.int64, numba.int64, numba.int64, numba.int32[:], numba.int32[:], numba.float32[:, :], numba.float32[:, :], numba.float32[:, :]), cache=True, parallel=True)
@fusion(transformation=[("i1", "j1"), ("i2", "j2")], function_props=function_props)
def sddmm_spmm_fused_v2(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in prange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in prange(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1][j3] += Xx[j2] * D[Ai[j2]][j3]
    return Y





# @fusion(transformation="(i2, j2)")
# def sddmm_spmm_f1(m, n, k, nnz, Ap, Ai, B, C, D):
#     Xx = np.zeros(nnz)
#     Y = np.zeros((m, n))
#     for i1 in range(m):
#         for i2 in range(Ap[i1], Ap[i1 + 1]):
#             for i3 in range(k):
#                 Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
#         for j2 in range(Ap[i1], Ap[i1 + 1]):
#             for j3 in range(n):
#                 Y[i1][j3] += Xx[j2] * D[i1][j3]

def benchmark(matrix, func, *args, repeat=5):
    times = []
    func(*args)  # Warm-up
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    avg = sum(times) / repeat
    print(f"{func.__name__}, {avg:.6f}, {repeat}, {matrix}")
    return avg


# TODO: This is temporary only for testing the decorator.
matrices = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/SPD/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/SPD/")

matrix = scio.mmread("/home/salehm32/projects/fused-gnn/fusion/data/tri-banded/tri-banded-1000.mtx")
A = matrix.tocsr()
m = A.shape[0]
p = A.shape[1]
n = 128
k = 128

Ap = A.indptr
Ai = A.indices
Ax = A.data.astype(np.float32)
nnz = A.nnz
B = np.random.rand(m, k).astype(np.float32)
C = np.random.rand(p, k).astype(np.float32)
D = np.random.rand(p, n).astype(np.float32)

out_unfused = sddmm_spmm_unfused(m, n, k, nnz, Ap, Ai, B, C, D)

out_fused_v1 = sddmm_spmm_fused_v1(m, n, k, nnz, Ap, Ai, B, C, D)

out_fused_v2 = sddmm_spmm_fused_v2(m, n, k, nnz, Ap, Ai, B, C, D)

if np.allclose(out_unfused, out_fused_v1, atol=1e-6):
    print("PASS")
else:
    print("FAIL")

for matrix_path in matrices:
    matrix = scio.mmread(matrix_path)
    A = matrix.tocsr()
    m = A.shape[0]
    p = A.shape[1]
    n = 128
    k = 128

    Ap = A.indptr
    Ai = A.indices
    Ax = A.data.astype(np.float32)
    nnz = A.nnz
    B = np.random.rand(m, k).astype(np.float32)
    C = np.random.rand(p, k).astype(np.float32)
    D = np.random.rand(p, n).astype(np.float32)
    benchmark(matrix_path, sddmm_spmm_unfused, m, n, k, nnz, Ap, Ai, B, C, D, repeat=10)
    benchmark(matrix_path, sddmm_spmm_fused_v1, m, n, k, nnz, Ap, Ai, B, C, D, repeat=10)
    benchmark(matrix_path, sddmm_spmm_fused_v2, m, n, k, nnz, Ap, Ai, B, C, D, repeat=10)


