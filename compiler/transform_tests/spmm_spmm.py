from compiler.decorator import sparse_fusion, fusion
import numpy as np
import scipy.io as scio
import sys

function_props = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
}
@fusion(transformation=[('i1', 'j1')], num_threads=8, function_props=function_props)
def spmm_spmm_fused(m, n, Ap, Ai, Ax, B):
    y = np.zeros((m,n))
    z = np.zeros((m,n))
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
    return z


def spmm_spmm_unfused(m, n, Ap, Ai, Ax, B):
    y = np.zeros((m,n))
    z = np.zeros((m,n))
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
    return z


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

    unfused_res = spmm_spmm_unfused(m,n,Ap,Ai,Ax,B)
    fused_res = spmm_spmm_fused(m,n,Ap,Ai,Ax,B)

    print("SpMM_SpMM_FUSED_TEST: ")
    if np.allclose(unfused_res, fused_res, atol=1e-6):
        print("PASS")
    else:
        print("FAIL")