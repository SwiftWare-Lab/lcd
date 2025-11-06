import numpy as np
import torch


from bench_utils import *
from spmv import *

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

from sptrsv import *
# Common values are '75' (for compute capability 7.5), '86' (for 8.6), etc.
# import os
# os.environ['NUMBAPRO_NVVM_NUM_THREADS'] = '1'
# os.environ['CUDA_ENABLE_COMPUTE_CAPABILITY'] = '84'  # Adjust this value based on your GPU
import warnings
from numba.core.errors import NumbaPerformanceWarning
import sys
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

device = cuda.get_current_device()
# print(f"Using GPU: {device.name}")
# print(f"Compute Capability: {device.compute_capability}")


# mat_list_file = sys.argv[2]
# mat_list_folder = sys.argv[1]

# mtx_list = get_matrix_list(mat_list_file, mat_list_folder)
mtx_path = sys.argv[1]
mtx_folder = sys.argv[2]
mtx = mtx_folder + "/" + mtx_path
rep_method_list = [
    "numba-csr-wavefront","numba-csr-wavefront-cudagraph-fine",
    "numba-csr-wavefront-cudagraph-coarse"
]


new_configs = []
new_configs.append(
    triton.testing.Benchmark(
        x_names=["matrix_name"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=rep_method_list,  # Label name for the lines
        line_names=[
            "numba-csr-wavefront","numba-csr-wavefront-cudagraph-fine",
            "numba-csr-wavefront-cudagraph-coarse"
        ],
        styles=[
            ("green", "-"), ("blue", "-"), ("green", "--"),
        ],  # Visual styles for the lines
        ylabel="GFLOP/S",  # Label name for the y-axis
        plot_name="sptrsv-cudagraph-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))

@triton.testing.perf_report(new_configs)
def benchmark_reps(matrix_name, provider):
    inner_reps = 50
    matrix = scio.mmread(matrix_name)
    n = matrix.shape[0]
    # convert matrix to csc
    A = matrix.tocsc()
    # makke it sparse lower triangular
    A = tril(A, format="csc")
    if A.data.dtype != np.float64:
        # print(f"Matrix {matrices} is not double precision")
        A.data = A.data.astype(np.float64)

    # add diagonals with 100
    A.setdiag(A.diagonal())
    b = np.ones(n)  # Dense vector b
    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 10, 25
    if provider == 'numba-csr-wavefront':
        # Determine which variant

        threads_per_block = 128
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        func = sptrsv_numba_csr_wavefront_mapping
        def run_with_reps():
            for _ in range(inner_reps):
                func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, threads_per_block)
        # Select function

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_with_reps(), quantiles=quantiles, warmup=warmpup, rep=rep)

    if provider == 'numba-csr-wavefront-cudagraph-coarse':
        threads_per_block = 128
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        func = sptrsv_numba_csr_wavefront_mapping_cudagraph_coarsenodes
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, inner_reps, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)

    if provider == 'numba-csr-wavefront-cudagraph-fine':
        threads_per_block = 128
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        func = sptrsv_numba_csr_wavefront_mapping_cudagraph_finenodes
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, inner_reps, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)

    perf = lambda ms: (A.shape[0] + 2 * A.nnz) / (ms * 1e-3) / 1e9
    #print(f"seconds: {ms}")
    #print(f"max: {perf(max_ms)}, min: {perf(min_ms)}, median: {perf(ms)}")
    # print(f"{provider} -- {matrices} -- seconds: {ms}")
    print(f"{provider},{matrix_name},{ms},{perf(ms)}")
    return perf(ms), perf(min_ms), perf(max_ms)

benchmark_reps.run(print_data=True, save_path=".", show_plots=False)