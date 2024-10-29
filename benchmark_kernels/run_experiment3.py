import os
import sys
import ctypes
import numpy as np
import scipy.sparse as sp
from benchmark_thread_level import benchmark_thread_level
from benchmark_warp_level import benchmark_warp_level
from benchmark_3row_warp_unroll import benchmark_3row_warp_unroll
from benchmark_enumerated import benchmark_enumerated
from benchmark_unroll_coarsen3 import benchmark_unroll_coarsen
from compressed_format_generator import csr_to_compressed_format_converter
from scipy.io import mmread
import matplotlib.pyplot as plt
import time
import cupy as cp
import importlib.util

sys.path.insert(1, '../utils')
sys.path.insert(1, '../code_gen')

from generate_triton_kernel_modular import generate_kernel

def get_patterns(kernel_m):
    patterns = {}
    for i in range(1, 2 ** (kernel_m * 1)):
        x = 2 ** (kernel_m * 1) - i
        arr = np.array([True if x == '1' else False for x in bin(x)[2:].zfill(kernel_m * 1)])
        patterns[tuple(arr)] = i - 1

    return patterns

def convert_patterns(patterns):
    converted_patterns = {}
    for pattern_tuple, num in patterns.items():
        indices = [i for i, val in enumerate(pattern_tuple) if val]
        converted_patterns[num] = indices
    return converted_patterns


# Script for generating plots for benchmarking the kernels

imports = """
import sys
import os

sys.path.insert(1, '../utils')
import numpy as np
import scipy.sparse as sp
import torch
import triton
import triton.language as tl
from cfmtx_utils import preprocess_patterns, create_array_offsets, flatten_array
from scipy.io import mmwrite
from read_compressed_format import read_compressed_matrix
"""

benchmarking_code = """
def benchmark_unroll_coarsen(A_cf, B):
    # Convert the patterns dict to a list for the GPU
    pattern_list = preprocess_patterns(A_cf.patterns)

    # Create the offsets for the Avp and Aip arrays
    Avp_offsets = create_array_offsets(A_cf.padded_values_data_array)
    Aip_offsets = create_array_offsets(A_cf.indices_array)

    # Load everything into the GPU
    device = torch.device('cuda')

    # Flatten and load arrays for A
    A_vp_flattened = flatten_array(A_cf.padded_values_data_array)
    A_vp = torch.tensor(A_vp_flattened, dtype=torch.int32, device=device)
    A_ip_flattened = flatten_array(A_cf.indices_array)
    A_ip = torch.tensor(A_ip_flattened, dtype=torch.int32, device=device)
    A_rp = torch.tensor(A_cf.row_panels_pointer, dtype=torch.int32, device=device)
    patterns_ptr = torch.tensor(pattern_list, dtype=torch.int32, device=device)
    Avp_offsets_ptr = torch.tensor(Avp_offsets, dtype=torch.int32, device=device)
    Aip_offsets_ptr = torch.tensor(Aip_offsets, dtype=torch.int32, device=device)

    # Load B and C arrays
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.zeros((A_cf.num_row, B.shape[1]), dtype=torch.float32, device=device)

    # Launch the kernel
    num_row_panels = len(A_cf.row_panels_pointer[0]) - 1
    grid = (num_row_panels, len(A_cf.patterns))

    # Create CUDA events to record the time
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Launch Triton kernel (or any CUDA operation)
    start.record()

    unroll_coarsen_kernel[grid](A_vp, A_ip, A_rp,
                                B_ptr, C_ptr,
                                patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr,
                                A_cf.kernel_rows, A_cf.kernel_cols,
                                B.shape[1])
    stop.record()

    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms
"""

kernel_n = 4
pattern_sizes = np.arange(2, 7, 1)
unroll_factors_i = [2]
unroll_factors_j = [2]
rows= 1024
cols = 512
densities = np.arange(0.1, 0.8, 0.1)
k = 20  # Number of iterations to average

# Generate B dense matrix with 'rows' number of rows, 'cols' number of cols in scipy
B_sp = sp.random(cols, rows, density=1.0, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
B = B_sp.toarray().astype(np.int32)

print("Running benchmarks...")
for pattern_size in pattern_sizes:

    patterns = get_patterns(pattern_size)

    thread_level_results = []
    warp_level_results = []
    warp_unroll_results = []
    enumerated_results = []
    unroll_coarsen_classic_results = []
    unroll_coarsen_i_results = []
    unroll_coarsen_j_results = []
    cuSPARSE_results = []
    cuBLAS_results = []
    builtin_results = []

    for density in densities:
        A_sp = sp.random(rows, cols, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
        A = A_sp.toarray().astype(np.int32)
        patterns = get_patterns(pattern_size)
        A_cf = csr_to_compressed_format_converter(A_sp, patterns, pattern_size, kernel_n)

        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

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
        
        def run_generated_kernel(gen_file, A_cf, B):
            
            output_file = "generated_kernel.py"
            with open(output_file, "w") as f:
                f.write(gen_file)

            spec = importlib.util.spec_from_file_location("gen_code", output_file)
            gen_code = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_code)

            return gen_code.benchmark_unroll_coarsen(A_cf, B)
        
        def generate_kernel_file(patterns, unroll_factors, loop_order):
            transformed_patterns = convert_patterns(patterns)

            gen_code = generate_kernel(transformed_patterns, unroll_factors, loop_order)

            # Split into lines and remove the first two lines
            code_lines = gen_code.splitlines()
            gen_code = "\n".join(code_lines[2:])

            # Add the necessary imports
            gen_code = imports + gen_code

            # Add benchmarking function
            gen_code += benchmarking_code

            return gen_code
        
        def benchmark_unroll_coarsen_classic(A_cf, B):
            unroll_factors = {'i': 1, 'j': 1}
            loop_order = ['i', 'j']
            gen_code = generate_kernel_file(patterns, unroll_factors, loop_order)

            return run_generated_kernel(gen_code, A_cf, B)

        
        def benchmark_unroll_coarsen_i(A_cf, B):
            unroll_factors = {'i': 2, 'j': 1}
            loop_order = ['i', 'j']
            gen_code = generate_kernel_file(patterns, unroll_factors, loop_order)
            
            return run_generated_kernel(gen_code, A_cf, B)
        
        def benchmark_unroll_coarsen_j(A_cf, B):
            unroll_factors = {'i': 1, 'j': 2}
            loop_order = ['i', 'j']
            gen_code = generate_kernel_file(patterns, unroll_factors, loop_order)
            
            return run_generated_kernel(gen_code, A_cf, B)
        
        # Run each benchmark `k` times and compute the average
        thread_level_res = average_benchmark(benchmark_thread_level, A, B)
        unroll_coarsen_classic_res = average_benchmark(benchmark_unroll_coarsen_classic, A_cf, B)
        unroll_coarsen_i_res = average_benchmark(benchmark_unroll_coarsen_i, A_cf, B)
        unroll_coarsen_j_res = average_benchmark(benchmark_unroll_coarsen_j, A_cf, B)
        cuSPARSE_res = average_benchmark(benchmark_cusparse, A, B, C)
        cuBLAS_res = average_benchmark(benchmark_cublas, A, B, C)
        builtin_res = average_benchmark(benchmark_builtin, A, B, C)

        # Append results to corresponding lists
        thread_level_results.append(thread_level_res)
        unroll_coarsen_classic_results.append(unroll_coarsen_classic_res)
        unroll_coarsen_i_results.append(unroll_coarsen_i_res)
        unroll_coarsen_j_results.append(unroll_coarsen_j_res)
        cuSPARSE_results.append(cuSPARSE_res)
        cuBLAS_results.append(cuBLAS_res)
        builtin_results.append(builtin_res)
        

    # Plotting the results
    plt.figure()
    plt.plot(densities, thread_level_results, label="Thread Level")
    plt.plot(densities, unroll_coarsen_classic_results, label="Unroll Coarsen Classic")
    plt.plot(densities, unroll_coarsen_i_results, label="Unroll Coarsen i-unroll_factor=2")
    plt.plot(densities, unroll_coarsen_j_results, label="Unroll Coarsen j-unroll_factor=2")
    plt.plot(densities, cuSPARSE_results, label="cuSPARSE")
    plt.plot(densities, cuBLAS_results, label="cuBLAS")
    plt.xlabel("Density")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"kernel_m = {pattern_size}")
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f"kernel_m={pattern_size}.png")
