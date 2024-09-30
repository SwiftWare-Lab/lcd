import os
import sys
import ctypes
import numpy as np
from benchmark_thread_level import benchmark_thread_level
from benchmark_warp_level import benchmark_warp_level
from benchmark_3row_warp_unroll import benchmark_3row_warp_unroll
from benchmark_enumerated import benchmark_enumerated
from benchmark_unroll_coarsen import benchmark_unroll_coarsen
from read_compressed_format import read_compressed_matrix
from scipy.io import mmread
import matplotlib.pyplot as plt

sys.path.insert(1, '../utils')

# Script for generating plots for benchmarking the kernels
cols = [32, 64, 128, 256, 512]
row_n = 768
densities = np.arange(0.1, 0.8, 0.1)
cusparse_csr= ctypes.CDLL('./libcuSPARSE_spMM_csr.so')  # Load the cuSPARSE implementation

for col in cols:
    thread_level_results = []
    warp_level_results = []
    warp_unroll_results = []
    enumerated_results = []
    unroll_coarsen_results = []
    cuSPARSE_results = []
    for density in densities:
        A_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.mtx"
        B_path = f"../matrices/B_sparse_512x{col}_1.0.mtx"

        # Load A and B as np arrays
        A_sp = mmread(A_path)
        A_sp = A_sp.tocsr()
        B_sp = mmread(B_path)

        # Convert scipy arrays to numpy arrays
        A = A_sp.toarray().astype(np.int32)
        B = B_sp.toarray().astype(np.float32)
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

        # Convert A to cfmtx format using compressed_format_generator.py
        os.system(f"python3 ../utils/compressed_format_generator.py {A_path} 8 8")

        A_cf_path = f"../matrices/A_sparse_{row_n}x128_{density:.2f}.cfmtx"
        A_cf = read_compressed_matrix(A_cf_path)

        thread_level_res = benchmark_thread_level(A, B)
        warp_level_res = benchmark_warp_level(A, B)
        warp_unroll_res = benchmark_3row_warp_unroll(A, B)      # Comment this out if you want to run with dimensions that are not multiples of 3
        enumerated_res = benchmark_enumerated(A, B)
        unroll_coarsen_res = benchmark_unroll_coarsen(A_cf, B)
        cusparse_csr.benchmark_cusparseSpMMCSR.restype = ctypes.c_float
        cuSPARSE_res = cusparse_csr.benchmark_cusparseSpMMCSR(
            ctypes.c_int(A_sp.shape[0]),
            ctypes.c_int(A_sp.shape[1]),
            ctypes.c_int(A_sp.nnz),
            ctypes.c_int(B.shape[1]),
            ctypes.c_void_p(A_sp.indptr.ctypes.data),
            ctypes.c_void_p(A_sp.indices.ctypes.data),
            ctypes.c_void_p(A_sp.data.ctypes.data),
            ctypes.c_void_p(B.ctypes.data),
            ctypes.c_void_p(C.ctypes.data),
        )

        unroll_coarsen_results.append(unroll_coarsen_res)
        thread_level_results.append(thread_level_res)
        warp_level_results.append(warp_level_res)
        warp_unroll_results.append(warp_unroll_res)
        enumerated_results.append(enumerated_res)
        cuSPARSE_results.append(cuSPARSE_res)
        
    
    plt.figure()
    plt.plot(densities, thread_level_results, label="Thread Level")
    #plt.plot(densities, warp_level_results, label="Warp Level")            Warp level seems to not be working properly at the moment, commented them out
    #plt.plot(densities, warp_unroll_results, label="3 Row Warp Unroll")
    #plt.plot(densities, enumerated_results, label="Enumerated")
    plt.plot(densities, unroll_coarsen_results, label="Unroll Coarsen")
    plt.plot(densities, cuSPARSE_results, label="cuSPARSE")
    plt.xlabel("Density")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"B Columns = {col}")
    plt.legend()
    # save the plot as a png with file name density_vs_time_{dim}.png
    plt.savefig(f"B_columns_{col}.png")
