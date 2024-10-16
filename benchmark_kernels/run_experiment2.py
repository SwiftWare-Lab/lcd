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
import cupy as cp

sys.path.insert(1, '../utils')

# Script for generating plots for benchmarking the kernels
cols = [32, 64, 128, 256, 512]
row_n = 768
densities = np.arange(0.1, 0.8, 0.1)
k = 20  # Number of iterations to average
# cuda_lib= ctypes.CDLL('./cuda_lib.so')  # Load the cuSPARSE implementation

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
        A = A_sp.toarray().astype(np.float32)
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


        def benchmark_cusparse(A, B, C):
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


        def benchmark_cublas(A_np, B, C):
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()

            start_event.record()

            c_cuBLAS = cp.matmul(cp.asarray(A), cp.asarray(B))

            # Wait for all operations to complete
            end_event.record()
            end_event.synchronize()

            elapsed_time_cuBLAS = cp.cuda.get_elapsed_time(start_event, end_event)

            return elapsed_time_cuBLAS

        # benchmark builin matrix multiplication
        def benchmark_builtin(A_np, B, C):
            # time A @ B
            start = time.time()
            C = A @ B
            end = time.time()
            # conver time to ms
            return (end - start) * 1000

        # Run each benchmark `k` times and compute the average
        thread_level_res = average_benchmark(benchmark_thread_level, A, B)
        unroll_coarsen_res = average_benchmark(benchmark_unroll_coarsen, A_cf, B)
        cuSPARSE_res = average_benchmark(benchmark_cusparse, A, B, C)
        cuBLAS_res = average_benchmark(benchmark_cublas, A, B, C)
        builtin_res = average_benchmark(benchmark_builtin, A, B, C)

        # Append results to corresponding lists
        thread_level_results.append(thread_level_res)
        unroll_coarsen_results.append(unroll_coarsen_res)
        cuSPARSE_results.append(cuSPARSE_res)
        cuBLAS_results.append(cuBLAS_res)
        builtin_results.append(builtin_res)

    # Plotting the results
    plt.figure()
    plt.plot(densities, thread_level_results, label="Thread Level")
    plt.plot(densities, unroll_coarsen_results, label="Unroll Coarsen")
    plt.plot(densities, cuSPARSE_results, label="cuSPARSE")
    plt.plot(densities, cuBLAS_results, label="cuBLAS")
    plt.xlabel("Density")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"B Columns = {col}")
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f"B_columns_{col}.png")

print("Done!")
