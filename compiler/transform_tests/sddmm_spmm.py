import numpy as np
from compiler.decorator import fusion
from numba import njit, prange
import numba
import sys
import scipy.io as scio

function_props = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
}


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


def spmm_spmm_v4_unfused(m, n, Ap, Ai, Ax, B):
    Y = np.zeros((m, n), dtype=np.float32)
    Z = np.zeros((m, n), dtype=np.float32)
    for i3 in range(n):
        for i1 in range(m):
            for i2 in range(Ap[i1], Ap[i1 + 1]):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j3 in range(n):
        for j1 in range(m):
            for j2 in range(Ap[j1], Ap[j1 + 1]):
                Z[j1][j3] += Ax[j2] * Y[Ai[j2]][j3]
    return Z


@fusion(transformation=[("i3", "j3")], function_props=function_props)
def spmm_spmm_v4_fused(m, n, Ap, Ai, Ax, B):
    Y = np.zeros((m, n), dtype=np.float32)
    Z = np.zeros((m, n), dtype=np.float32)
    for i3 in range(n):
        for i1 in range(m):
            for i2 in range(Ap[i1], Ap[i1 + 1]):
                Y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j3 in range(n):
        for j1 in range(m):
            for j2 in range(Ap[j1], Ap[j1 + 1]):
                Z[j1][j3] += Ax[j2] * Y[Ai[j2]][j3]
    return Z


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


if __name__ == '__main__':
    mat_path = sys.argv[1]
    matrix = scio.mmread(mat_path)

    # matrix = scio.mmread("/home/salehm32/projects/fused-gnn/fusion/data/tri-banded/tri-banded-1000.mtx")
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

    print("SDDMM_SpMM_FUSED_V1_TEST: ")
    if np.allclose(out_unfused, out_fused_v1, atol=1e-6):
        print("PASS")
    else:
        print("FAIL")

    print("SDDMM_SpMM_FUSED_V2_TEST: ")
    if np.allclose(out_unfused, out_fused_v2, atol=1e-6):
        print("PASS")
    else:
        print("FAIL")

    out_unfused = spmm_spmm_v4_unfused(m, n, Ap, Ai, Ax, D)

    out_fused = spmm_spmm_v4_fused(m, n, Ap, Ai, Ax, D)

    print("SpMM_SpMM_V4_FUSED_TEST: ")
    if np.allclose(out_unfused, out_fused, atol=1e-6):
        print("PASS")
    else:
        print("FAIL")


#TODO: Add other pairs of operations