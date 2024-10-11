# script that generates a 2x3 sparse matrix and a 3x4 dense matrix and outputs to mtx file

import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
import ctypes

A_sp = sp.random(2, 3, density=1.0, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
mmwrite("A.mtx", A_sp)
B_sp = sp.random(3, 4, density=1.0, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
mmwrite("B.mtx", B_sp)
# Convert B to np
B = B_sp.toarray().astype(np.float32)
C = np.zeros((2, 4), dtype=np.float32)

cuda_lib= ctypes.CDLL('./cuda_lib.so')

cuda_lib.test_cusparseSpMMCSR.restype = ctypes.c_void_p
C_res = cuda_lib.test_cusparseSpMMCSR( ctypes.c_int(A_sp.shape[0]),
                                    ctypes.c_int(A_sp.shape[1]),
                                    ctypes.c_int(A_sp.nnz),
                                    ctypes.c_int(B.shape[1]),
                                    ctypes.c_void_p(A_sp.indptr.ctypes.data),
                                    ctypes.c_void_p(A_sp.indices.ctypes.data),
                                    ctypes.c_void_p(A_sp.data.ctypes.data),
                                    ctypes.c_void_p(B.ctypes.data),
                                    ctypes.c_void_p(C.ctypes.data))

# Print the resulting matrix
C_ref = A_sp.toarray().astype(np.float32) @ B
C_arr = np.ctypeslib.as_array(C_res, shape=(2, 4))
print(C_arr)
print(C_ref)
print(np.allclose(C_arr, C_ref, atol=1e-6))

