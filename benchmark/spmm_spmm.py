import numba
from numba import njit, frange
from compiler.loop_carried_compiler import MyCompiler
import numpy as np

@njit(numba.float64[:,:](numba.int32[:], numba.int32[:], numba.float64[:], numba.float64[:,:], numba.int32, numba.int32),
      pipeline_class=MyCompiler)
def spmm_spmm_v1(Ap, Ai, Ax, B, m, n):
    y = np.zeros((m, n))
    z = np.zeros((m, n))
    for i1 in frange(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in frange(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
    return y

# @njit(numba.float64[:,:](numba.int32[:], numba.int32[:], numba.float64[:], numba.float64[:,:], numba.int32, numba.int32),
#       pipeline_class=MyCompiler)
# def spmm_spmm_v2(Ap, Ai, Ax, B, m, n):
#     y = np.zeros((m, n))
#     z = np.zeros((m, n))
#     for i3 in frange(n):
#         for i1 in range(m):
#             for i2 in range(Ap[i1], Ap[i1 + 1]):
#                 y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
#     for j1 in frange(m):
#         for j2 in range(Ap[j1], Ap[j1 + 1]):
#             for j3 in range(n):
#                 z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
#     return y

# @njit(numba.float64[:,:](numba.int32[:], numba.int32[:], numba.float64[:], numba.float64[:,:], numba.int32, numba.int32),
#       pipeline_class=MyCompiler)
# def spmm_spmm_v3(Ap, Ai, Ax, B, m, n):
#     y = np.zeros((m, n))
#     z = np.zeros((m, n))
#     for i1 in frange(m):
#         for i2 in range(Ap[i1], Ap[i1 + 1]):
#             for i3 in range(n):
#                 y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
#     for j3 in range(n):
#         for j1 in frange(m):
#             for j2 in range(Ap[j1], Ap[j1 + 1]):
#                 z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
#     return y
#
# @njit(numba.float64[:,:](numba.int32[:], numba.int32[:], numba.float64[:], numba.float64[:,:], numba.int32, numba.int32),
#       pipeline_class=MyCompiler)
# def spmm_spmm_v3(Ap, Ai, Ax, B, m, n):
#     y = np.zeros((m, n))
#     z = np.zeros((m, n))
#     for i3 in range(n):
#         for i1 in frange(m):
#             for i2 in range(Ap[i1], Ap[i1 + 1]):
#                 y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
#     for j3 in range(n):
#         for j1 in frange(m):
#             for j2 in range(Ap[j1], Ap[j1 + 1]):
#                 z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
#     return y