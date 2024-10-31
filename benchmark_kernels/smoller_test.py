import scipy.sparse as sp
import numpy as np
import cupy as cp

# for converting scipy arrays to cfmtx
import sys
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore
from scipy.io import mmwrite
import os

# for generating, running kernel code
sys.path.insert(1, '../code_gen')
from code_gen_utils import generate_kernel_code, run_generated_kernel # type: ignore

A_dim = (768, 768)
A_density = 0.5
pattern_size = 4
kernel_n = 4
unroll_factor = (1,1)
loop_order = ['i', 'j']
k = 20
B_col = 512
B_density = 1.0

def csr_to_cfmtx(M_sp, kernel_m, kernel_n):
    mmwrite("M.mtx", M_sp)
    os.system(f"python3 ../utils/compressed_format_generator.py M.mtx {kernel_m} {kernel_n} > /dev/null 2>&1")
    M_cf = read_compressed_matrix("M.cfmtx")
    return M_cf

def gen_sp_matrix(n_rows, n_cols, d):
    mat_sp = sp.random(n_rows, n_cols,
                       density=d, format='csr',
                       dtype=np.int32,
                       data_rvs=lambda s: np.random.randint(1, 11, size=s))
    return mat_sp

def benchmark_gen_code(code, A_cf, B):
    results = []
    for _ in range(k):
        results.append(run_generated_kernel(code, A_cf, B))
    return np.mean(results)

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

A_sp = gen_sp_matrix(A_dim[0], A_dim[1], A_density)
A_cf = csr_to_cfmtx(A_sp, pattern_size, kernel_n)
B_sp = gen_sp_matrix(A_dim[1], B_col, B_density)
A = A_sp.toarray()
B = B_sp.toarray()
transformed_unroll_factor = {'i': unroll_factor[0], 'j': unroll_factor[1]}
# gen_file_code = generate_kernel_code(A_cf.patterns, transformed_unroll_factor, loop_order)
# averaged_exec_time = benchmark_gen_code(gen_file_code, A_cf, B)

# print(averaged_exec_time)

print(f"cuSPARSE: {benchmark_cuSPARSE(A, B)}")
print(f"cuBLAS: {benchmark_cuBLAS(A, B)}")
