import numpy as np
import torch
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

from bench_utils import *
from spmv import *

from sksparse.cholmod import cholesky, analyze
from sympy import factor

#from sksparse.cholmod import cholesky, analyze, Factor

from utils.symbolic_cholesky_csc import etree, post_order, row_col_counts, \
    detect_supernodes, convert_csc_to_supernodal, get_supernodal_dag
from sptrsv_supernodal import sptrv_csc_supernodal, sptrsv_csc_supernodal_wavefront


NUM_RUNS=30

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


# Set the compute capability to match your GPU
# Common values are '75' (for compute capability 7.5), '86' (for 8.6), etc.
# import os
# os.environ['NUMBAPRO_NVVM_NUM_THREADS'] = '1'
# os.environ['CUDA_ENABLE_COMPUTE_CAPABILITY'] = '84'  # Adjust this value based on your GPU

# import cuda

from numba import cuda
device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")



mtx_list = [
    "/home/kazem/Downloads/LFAT5/LFAT5.mtx", "/home/kazem/Downloads/ex5/ex5.mtx", "/home/kazem/Downloads/mesh1e1/mesh1e1.mtx","/home/kazem/Downloads/nos4/nos4.mtx",
    # "/home/kazem/Downloads/bcsstk03/bcsstk03.mtx",

    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx",
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx"
            ]
method_list = ["triton", "triton-sync-free", "cusparse",  "numba-sptrsv", "numba-sync-free", "triton-spmv"]
#mtx_list = get_matrix_list("/home/kazem/UFDB/spd_list.txt", "/home/kazem/UFDB/")
#mtx_list = mtx_list[60:]

def correctness_test():
    matrix_signtures = []
    method_correctness = {}
    for mtx_matrix_path in mtx_list:
        print(f"------------ matrix: {mtx_matrix_path}")
        # read matrix from file
        Ao = spio.mmread(mtx_matrix_path)
        Ao_csc = Ao.tocsc()
        # lower part
        Au = sp.triu(Ao).tocsc()
        A = sp.tril(Ao).tocsc()

        factor = cholesky(Ao)
        L = factor.LD()
        P = factor.P()
        b = L.dot(np.ones(L.shape[0]))  # np.random.rand(Ao.shape[0])
        # Solve the linear system Ax = b
        # x = factor(b)
        x = factor.solve_A(b)
        print("nnz L: ", L.nnz)
        # L = factor.L()
        # factor = analyze(A)
        # factor.cholesky(A)
        # print("nz A: ", A.nnz, " nz L: ", L.nnz)

        # x = factor.solve_A(b)
        print("norm: ", np.linalg.norm(b - Ao.dot(x)))
        # permute the matrix Ao
        Aperm = Ao_csc[P, :][:, P]

        parent = np.array(A.shape[0] * [0], dtype=np.int32)
        common = {}

        Ls = convert_csc_to_supernodal(L)

        x_sup_csc = sptrv_csc_supernodal(Ls['supernodes'],
                                         Ls['ip'],
                                         Ls['i'],
                                         Ls['p'],
                                         Ls['x'], b)

        # x_csc = L.solve(b)

        print("norm: ", np.linalg.norm(x_sup_csc - np.ones(L.shape[0])))

        # compute wavefronts of sparse matrix L (CSC)
        dag_L_super = get_supernodal_dag(Ls)
        wf_sup = compute_wavefront(dag_L_super)
        wave_start_loc, waves = compact_2d_list(wf_sup)

        x_sup_wf = sptrsv_csc_supernodal_wavefront(Ls['supernodes'],
                                         Ls['ip'],
                                         Ls['i'],
                                         Ls['p'],
                                         Ls['x'], b, wf_sup)
        print("norm: ", np.linalg.norm(x_sup_wf - np.ones(L.shape[0])))




        # print (x_cpu[:5])
        # print (x_gpu[:5])
        # print (y[:5])
        # print ("Triton sync free", y_sync_free[:50])
        # print (y_numba_cpu[:5])
        # print ("numba synch-free", y_numba_cpu_sync_free[:5])
    return matrix_signtures, method_correctness


mat_signatures, method_correctness = correctness_test()
# convert to df and save as csv
import pandas as pd
df = pd.DataFrame(mat_signatures)
df.to_csv("matrix_signatures.csv")

df = pd.DataFrame(method_correctness)
df.to_csv("method_correctness.csv")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=method_list,  # Label name for the lines
        line_names=["Triton Wavefront", "Triton Sync-free", "Numba SpTRSV CSC", "Numba SpTRSV CSC Sync-Free", "cuSparse", "Triton SPMV-CSC Ref"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("purple", "-"), ("yellow", "-"), ("orange", "--")],  # Visual styles for the lines
        ylabel="GFLOP/S",  # Label name for the y-axis
        plot_name="sptrsv-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))


# use triton benchmark
@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
#    for mtx_matrix_path in mtx_list:
    # read matrix from file
    matrix = scio.mmread(matrices)
    n = matrix.shape[0]
    # convert matrix to csc
    A = matrix.tocsc()
    # makke it sparse lower triangular
    A = tril(A, format="csc")
    if A.data.dtype != np.float64:
        print(f"Matrix {matrices} is not double precision")
        A.data = A.data.astype(np.float64)

    # add diagonals with 100
    A.setdiag(A.diagonal())
    b = np.ones(n)  # Dense vector b
    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 100, 25
    if provider == 'triton':
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_triton(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'triton-sync-free':
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        dep_count = incoming_edge(A)
        dep_count_gpu = torch.tensor(dep_count, dtype=torch.int32, device=DEVICE)
        max_col_size = max(A.getnnz(axis=0))
        if max_col_size < 63:
            BLOCKSIZE = 64
        elif max_col_size < 255:
            BLOCKSIZE = 256
        elif max_col_size < 511:
            BLOCKSIZE = 512
        else:
            BLOCKSIZE = 1024
            assert max_col_size < 1023
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_sync_free_triton(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A.nnz, dep_count_gpu, BLOCKSIZE), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'triton-spmv':
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(np.zeros(n), dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: spmv_triton(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A.nnz), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-sptrsv':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsc_numba_csc(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-sync-free':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        #dep_count_gpu = cuda.to_device(dep_count)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsc_numba_csc_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'cusparse':
        matrix_gpu_csc = cp_csc_matrix(A.tocsc())
        b_gpu = cp.array(b)  # Move dense vector to GPU
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cusparse_solver(matrix_gpu_csc, b_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)

    perf = lambda ms: (A.shape[0] + 2 * A.nnz) / (ms * 1e-3) / 1e9
    #print(f"seconds: {ms}")
    #print(f"max: {perf(max_ms)}, min: {perf(min_ms)}, median: {perf(ms)}")
    print(f"{provider} -- {matrices} -- seconds: {ms}")
    return perf(ms), perf(min_ms), perf(max_ms)

# Call the benchmark function
#benchmark.run(print_data=True, save_path=".", show_plots=True)

