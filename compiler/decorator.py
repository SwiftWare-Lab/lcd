import inspect
import types

import numpy as np
import types
import ast
from legality_check import finalize_legality_of_fusion, transform_ast
import scipy.io as scio
from numba import njit
import numba



def fusion(**kwargs):
    def decorator(function: types.FunctionType):
        transformations = kwargs['transformation']
        function_props = kwargs['function_props']
        source = inspect.getsource(function)
        source = '\n'.join([l for l in source.splitlines() if not l.startswith("@")])

        old_code_obj = function.__code__
        cur_ast = ast.parse(source)
        for transformation in transformations:
            if finalize_legality_of_fusion(transformation, cur_ast):
                transformed_tree = transform_ast(transformation, cur_ast)
                print(ast.unparse(transformed_tree))
                transformed_tree = ast.fix_missing_locations(transformed_tree)
                cur_ast = transformed_tree
            else:
                raise "Transformation {} is not legal.".format(transformation)

        new_code_obj = compile(transformed_tree, old_code_obj.co_filename, 'exec')
        new_f = types.FunctionType(new_code_obj.co_consts[0], function.__globals__)
        return new_f
    return decorator


function_props = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
}
@njit(numba.float32[:, :](numba.int64, numba.int64, numba.int64, numba.int64, numba.int32[:], numba.int32[:],
                          numba.float32[:, :], numba.float32[:, :], numba.float32[:, :]))
def sddmm_spmm_unfused(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1][j3] += Xx[j2] * D[Ai[j2]][j3]
    return Y

@njit(numba.float32[:, :](numba.int64, numba.int64, numba.int64, numba.int64, numba.int32[:], numba.int32[:], numba.float32[:, :], numba.float32[:, :], numba.float32[:, :]))
@fusion(transformation=[("i1", "j1"), ("i2", "j2")], function_props=function_props)
def sddmm_spmm(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros((m, n), dtype=np.float32)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in range(m):
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

# TODO: This is temporary only for testing the decorator.
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

out_fused_v1 = sddmm_spmm(m, n, k, nnz, Ap, Ai, B, C, D)

if np.allclose(out_unfused, out_fused_v1, atol=1e-6):
    print("PASS")
else:
    print("FAIL")

