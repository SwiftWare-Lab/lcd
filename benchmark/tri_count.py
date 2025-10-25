import numpy as np
from numba import njit, prange
import numba
from compiler.loop_carried_compiler import MyCompiler


@njit(numba.float64(numba.int32[:], numba.int32[:], numba.float32[:], numba.int64, numba.float32[:,:], numba.float32[:]),
      pipeline_class=MyCompiler)
def tri_count(row_ptr, col_idx, values, n, X, result_values):
    sum_tri = 0
    for i in range(n):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            for p in range(row_ptr[col_idx[j]], row_ptr[col_idx[j] + 1]):
                X[i][col_idx[p]] += values[j] * values[p]
    for i in range(n):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            result_values[j] = X[i][col_idx[j]]
    for i in range(n):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            sum_tri += result_values[j]
    sum_tri = sum_tri / 6
    return sum_tri