import numba
from numba import njit, frange
from compiler.loop_carried_compiler import MyCompiler
import numpy as np


@njit(numba.float64[:](numba.int32[:], numba.int32[:], numba.float64[:], numba.float64[:], numba.int32),
      pipeline_class=MyCompiler)
def spmv_spmv(Ap, Ai, Ax, B, m):
    y = np.zeros(m)
    z = np.zeros(m)
    for i1 in frange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            y[i1] += Ax[i2] * B[Ai[i2]]
    for j1 in frange(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            z[j1] += Ax[j2] * y[Ai[j2]]
    return y


def spmv_spmv_wrong(Ap, Ai, Ax, B, m):
    y = np.zeros(m)
    z = np.zeros(m)
    for i1 in frange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            y[i1] += Ax[i2] * B[Ai[i2]]
        for j2 in range(Ap[i1], Ap[i1 + 1]):
            z[i1] += Ax[j2] * y[Ai[j2]]
    return y