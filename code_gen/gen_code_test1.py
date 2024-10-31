# This is a script to test the the code generator
import scipy.sparse as sp
import numpy as np
import cupy as cp

# for converting scipy arrays to cfmtx
import sys
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore
from scipy.io import mmwrite
import os

# import test kernel
from test_generated_kernel import benchmark_unroll_coarsen

k = 10

def gen_sp_matrix(n_rows, n_cols, d):
    mat_sp = sp.random(n_rows, n_cols,
                       density=d, format='csr',
                       dtype=np.int32,
                       data_rvs=lambda s: np.random.randint(1, 11, size=s))
    return mat_sp

def csr_to_cfmtx(M_sp, kernel_m, kernel_n):
    mmwrite("M.mtx", M_sp)
    os.system(f"python3 ../utils/compressed_format_generator.py M.mtx {kernel_m} {kernel_n} > /dev/null 2>&1")
    M_cf = read_compressed_matrix("M.cfmtx")
    return M_cf

def cuSPARSE_instance(A, B):
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    A_sparse_csr = cp.sparse.csr_matrix(cp.asarray(A))

    # GPU-based timing (start)
    start_event.record()

    # Perform sparse matrix multiplication: A_sparse_csr.dot(B_dense)
    c_cuSparse = A_sparse_csr.dot(cp.asarray(B))

    end_event.record()
    end_event.synchronize()
    c_cuSparse_ref = c_cuSparse.get()

    elapsed_time_cuSparse = cp.cuda.get_elapsed_time(start_event, end_event)

    return elapsed_time_cuSparse  

def benchmark_cuSPARSE(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    results = []
    for _ in range(k):
        results.append(cuSPARSE_instance(A, B))
    return np.mean(results)

def cuBLAS_instance(A, B):
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()

    c_cuBLAS = cp.matmul(cp.asarray(A), cp.asarray(B))

    # Wait for all operations to complete
    end_event.record()
    end_event.synchronize()

    elapsed_time_cuBLAS = cp.cuda.get_elapsed_time(start_event, end_event)

    return elapsed_time_cuBLAS

def benchmark_cuBLAS(A, B):
    results = []
    for _ in range(k):
        results.append(cuBLAS_instance(A, B))
    return np.mean(results)

A_dim = (768, 768)
A_density = 0.5

B_col = 64
B_density = 1.0

kernel_m = 2
kernel_n = 4

A_sp = gen_sp_matrix(A_dim[0], A_dim[1], A_density)
B_sp = gen_sp_matrix(A_dim[1], B_col, B_density)
A = A_sp.toarray()
B  = B_sp.toarray()

A_cf = csr_to_cfmtx(A_sp, kernel_m, kernel_n)

triton_time = benchmark_unroll_coarsen(A_cf, B)
cuSPARSE_time = benchmark_cuSPARSE(A, B)
cuBLAS_time = benchmark_cuBLAS(A, B)


print(f"Triton time: {triton_time}")
print(f"cuSPARSE time: {cuSPARSE_time}")
print(f"cuBLAS time: {cuBLAS_time}")




