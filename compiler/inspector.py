import inspect
import types

import numpy as np


def fusion(f: types.FunctionType, *args, **kwargs):
    source = inspect.getsource(f)
    source = '\n'.join(source.splitlines()[1:])

    old_code_bj = f.__code__



@fusion(transformation="(i2, j2)")
@fusion(transformation="(i1, j1)")
def sddmm_spmm(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n))
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1][j3] += Xx[j2] * D[j1][j3]


@fusion(transformation="(i2, j2)")
def sddmm_spmm_f1(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n))
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
        for j2 in range(Ap[i1], Ap[i1 + 1]):
            for j3 in range(n):
                Y[i1][j3] += Xx[j2] * D[i1][j3]
