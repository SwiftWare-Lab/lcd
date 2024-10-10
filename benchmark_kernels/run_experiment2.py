import os
import sys
import ctypes
import numpy as np
from benchmark_thread_level import benchmark_thread_level
from benchmark_warp_level import benchmark_warp_level
from benchmark_3row_warp_unroll import benchmark_3row_warp_unroll
from benchmark_enumerated import benchmark_enumerated
from benchmark_unroll_coarsen3 import benchmark_unroll_coarsen
from read_compressed_format import read_compressed_matrix
from scipy.io import mmread
import matplotlib.pyplot as plt
import time

sys.path.insert(1, '../utils')

# Script for generating plots for benchmarking the kernels
cols = [32, 64, 128, 256, 512]
row_n = 768
densities = np.arange(0.1, 0.8, 0.1)
k = 20  # Number of iterations to average
cuda_lib= ctypes.CDLL('./cuda_lib.so')  # Load the cuSPARSE implementation

print("Running benchmarks...")

for col in cols:
    thread_level_results = []
    warp_level_results = []
    warp_unroll_results = []
    enumerated_results = []
    unroll_coarsen_results = []
    cuSPARSE_results = []
    cuBLAS_results = []
    builtin_results = []
    
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
        os.system(f"python3 ../utils/compressed_format_generator.py {A_path} 3 4 > /dev/null 2>&1")

        A_cf_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.cfmtx"
        A_cf = read_compressed_matrix(A_cf_path)

        # Function to run a benchmark `k` times and return the average
        def average_benchmark(benchmark_fn, *args):
            results = []
            for _ in range(k):
                results.append(benchmark_fn(*args))
            return np.mean(results)

        # Run each benchmark `k` times and compute the average
        thread_level_res = average_benchmark(benchmark_thread_level, A, B)
        warp_level_res = average_benchmark(benchmark_warp_level, A, B)
        warp_unroll_res = average_benchmark(benchmark_3row_warp_unroll, A, B)
        enumerated_res = average_benchmark(benchmark_enumerated, A, B)
        unroll_coarsen_res = average_benchmark(benchmark_unroll_coarsen, A_cf, B)

        # For cuSPARSE, run the benchmark `k` times and average the results
        def benchmark_cusparse(A_sp, B, C):
            cuda_lib.benchmark_cusparseSpMMCSR.restype = ctypes.c_float
            return cuda_lib.benchmark_cusparseSpMMCSR(
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


        def benchmark_cublas(A_np, B, C):
            cuda_lib.benchmark_cublasSgemm.restype = ctypes.c_float
            res = cuda_lib.benchmark_cublasSgemm(
                ctypes.c_int(A_sp.shape[0]),
                ctypes.c_int(A_sp.shape[1]),
                ctypes.c_int(B.shape[1]),
                ctypes.c_void_p(A_np.ctypes.data),
                ctypes.c_void_p(B.ctypes.data),
                ctypes.c_void_p(C.ctypes.data),
            )
            return res

        # benchmark builin matrix multiplication
        def benchmark_builtin(A_np, B, C):
            # time A @ B
            start = time.time()
            C = A @ B
            end = time.time()
            # conver time to ms
            return (end - start) * 1000


        cuSPARSE_res = average_benchmark(benchmark_cusparse, A_sp, B, C)
        cuBLAS_res = average_benchmark(benchmark_cublas, A, B, C)
        builtin_res = average_benchmark(benchmark_builtin, A, B, C)

        # Append results to corresponding lists
        thread_level_results.append(thread_level_res)
        warp_level_results.append(warp_level_res)
        warp_unroll_results.append(warp_unroll_res)
        enumerated_results.append(enumerated_res)
        unroll_coarsen_results.append(unroll_coarsen_res)
        cuSPARSE_results.append(cuSPARSE_res)
        cuBLAS_results.append(cuBLAS_res)
        builtin_results.append(builtin_res)
        
    # Plotting the results
    plt.figure()
    plt.plot(densities, thread_level_results, label="Thread Level")
    #plt.plot(densities, warp_level_results, label="Warp Level")  # Comment this out if it's not working
    #plt.plot(densities, warp_unroll_results, label="3 Row Warp Unroll")
    #plt.plot(densities, enumerated_results, label="Enumerated")
    plt.plot(densities, unroll_coarsen_results, label="Unroll Coarsen")
    plt.plot(densities, cuSPARSE_results, label="cuSPARSE")
    plt.plot(densities, cuBLAS_results, label="cuBLAS")
    #plt.plot(densities, builtin_results, label="Built-in")
    plt.xlabel("Density")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"B Columns = {col}")
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(f"B_columns_{col}.png")

print("Done!")