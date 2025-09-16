import numpy as np
from compiler.decorator import fusion

function_props = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
}

@fusion(transformation=[("i1", "j1")], function_props=function_props)
def spmm_gemm(Ap, Ai, Ax, B, C, m, n, k):
    X = np.zeros(m*k)
    Y = np.zeros(m*n)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1+1]):
            for i3 in range(k):
                X[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(k):
            for j3 in range(n):
                Y[j1][j3] += X[j1][j2] * C[j2][j3]
    return Y

#TODO: implement other data strucutures like BSR
def spmm_gemm_unfused(Ap, Ai, Ax, B, C, m, n, k):
    X = np.zeros(m*k)
    Y = np.zeros(m*n)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1+1]):
            for i3 in range(k):
                X[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(k):
            for j3 in range(n):
                Y[j1][j3] += X[j1][j2] * C[j2][j3]
    return Y


#TODO: What is needed to perform dependency analysis on this?
def spmm_gemm_bsr(Ap, Ai, Ax, B, C, m, n, k, bm, bk):
    n_block_rows = len(Ap) - 1
    Y = np.zeros(m,n)
    for i1 in range(n_block_rows):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                for i4 in range(bm):
                    for i5 in range(bk):
                        Y[i1*bm + i4][i3] += Ax[i2][i4][i5] * B[Ai[i2]*bk + i5][i3]
    for j1 in range(m):
        for j2 in range(n):
            for j3 in range(n):
                Y[j1][j3] += Y[j1][j2] * C[j2][j3]

#TODO: How to perform dependency analysis on blocked implementations?
def spmm_gemm_bsr_v2(Ap, Ai, Ax, B, C, m, n, k, bm, bk):
    n_block_rows = len(Ap) - 1
    Y = np.zeros(m,n)
    for i1 in range(n_block_rows):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
                        Y[i1*bm:(i1+1)*bm][:] += Ax[i2] @ B[Ai[i2]*bk:(Ai[i2] + 1)*bk][:]
    for j1 in range(m):
        for j2 in range(n):
            for j3 in range(n):
                Y[j1][j3] += Y[j1][j2] * C[j2][j3]


