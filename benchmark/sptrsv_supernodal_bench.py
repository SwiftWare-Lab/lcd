import numpy as np
import torch
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

from bench_utils import *
from spmv import *

from sksparse.cholmod import cholesky, analyze
from sympy import factor
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

#from sksparse.cholmod import cholesky, analyze, Factor

from utils.symbolic_cholesky_csc import etree, post_order, row_col_counts, \
    detect_supernodes, convert_csc_to_supernodal, get_supernodal_dag
from sptrsv_supernodal import *
# from sptrsv_supernodal_cublas import (
#     SupernodalSpTrsvCuBLAS,
#     sptrsv_csc_supernodal_cublas
# )
# from sptrsv_supernodal_cudagraphs import (
#     SupernodalSpTrsvCudaGraph,
#     sptrsv_csc_supernodal_cudagraph
# )
from sptrsv_supernodal_cudagraph_syncfree import (
    SupernodalSpTrsvCudaGraphSyncFree,
    sptrsv_supernodal_cudagraph_syncfree
)
# from sptrsv_supernodal_cudagraph_syncfree_true import (
#     TrueSyncFreeSpTrsvCudaGraph,
#     sptrsv_supernodal_cudagraph_syncfree_true
# )
# from sptrsv_supernodal_syncfree import (
#     sptrsv_supernodal_persistent,
#     sptrsv_supernodal_block_level,
#     sptrsv_supernodal_hybrid
# )
from utils.inspectors import sptrsv_csc_supernodal_analysis, \
    sptrsv_csc_supernodal_analysis_1

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
import numba
device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")

import os
os.environ['NUMBA_CUDA_PTX_VERSION'] = '84'

Relaxtion = 8

mtx_list = [
    "/home/mad29/projects/testdata/matrices/1138_bus.mtx",
            ]
method_list = ["python-sptrsv-csc-super", "numba-sptrsv-csc-super", "numba-sptrsv-csc-super-wavefront","cuda-numba-sptrsv-csc-super-syncfree", "cudagraphs-syncfree"]
# NOTE: Other methods commented out to focus on sptrsv_supernodal.py and sptrsv_supernodal_cudagraph_syncfree.py implementations
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
        L = factor.L()
        P = factor.P()
        mat_name = mtx_matrix_path.split("/")[-1]
        matrix_signtures.append(get_matrix_signature(A, mat_name))

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
        print("norm cholesky: ", np.linalg.norm(b - Ao.dot(x), ord=np.inf))

        x_chol = factor.solve_L(b, use_LDLt_decomposition=False)
        print("norm cholmod: ", np.linalg.norm(x_chol - np.ones(L.shape[0]), ord=np.inf))
        # permute the matrix Ao
        # Aperm = Ao_csc[P, :][:, P]
        #
        # parent = np.array(A.shape[0] * [0], dtype=np.int32)
        # common = {}

        Ls = convert_csc_to_supernodal(L, Relaxtion)
        signature_l = get_L_signature(L, Ls, mat_name)
        for k, v in signature_l.items():
            matrix_signtures[-1][k] = v
        x_sup_csc = b.copy()
        sptrv_csc_supernodal(Ls['supernodes'],
                                         Ls['ip'],
                                         Ls['i'],
                                         Ls['p'],
                                         Ls['x'], b, x_sup_csc)

        # x_csc = L.solve(b)

        print("norm python: ", np.linalg.norm(x_sup_csc - np.ones(L.shape[0]), ord=np.inf))

        # compute wavefronts of sparse matrix L (CSC)
        dag_L_super = get_supernodal_dag(Ls)
        wf_sup = compute_wavefront(dag_L_super)
        wave_start_loc, waves = compact_2d_list(wf_sup)
        x_sup_wf = b.copy()
        sptrsv_csc_supernodal_numba(Ls['supernodes'],
                                         Ls['ip'],
                                         Ls['i'],
                                         Ls['p'],
                                         Ls['x'], b, x_sup_wf)
        print("norm numba seq: ", np.linalg.norm(x_sup_wf - np.ones(L.shape[0])))
        x_sup_wf_numba = b.copy()
        sptrsv_csc_supernodal_wavefront_numba(Ls['supernodes'],
                                            Ls['ip'],
                                            Ls['i'],
                                            Ls['p'],
                                            Ls['x'], b, x_sup_wf_numba, wave_start_loc, waves)
        #sptrsv_csc_supernodal_wavefront_numba.parallel_diagnostics(level=4)
        print("norm numba wf: ", np.linalg.norm(x_sup_wf_numba - np.ones(L.shape[0]), ord=np.inf))


        # spy plot of dag_l_super
        # plt.spy(dag_L_super, markersize=2)
        # plt.show()
        thread_flags = sptrsv_csc_supernodal_analysis_1(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['col_to_super'])
        incom_edges = incoming_edge(dag_L_super)
        x_sup_cuda = np.zeros_like(b)
        # Transfer data to the GPU
        d_lx = cuda.to_device(Ls['x'])
        d_super = cuda.to_device(Ls['supernodes'])
        col_to_sup = cuda.to_device(Ls['col_to_super'])
        d_ip = cuda.to_device(Ls['ip'])
        d_i = cuda.to_device(Ls['i'])
        d_p = cuda.to_device(Ls['p'])
        d_b = cuda.to_device(b)
        d_in_edge = cuda.to_device(incom_edges)
        d_thread_flags = cuda.to_device(thread_flags)
        d_x = cuda.to_device(b)
        max_width = Ls['max_width']
        print("max width: ", max_width)
        sptrsv_csc_supernodal_numba_cuda(d_super, col_to_sup, d_ip, d_i, d_p, d_lx, d_x, d_thread_flags, max_width)
        x_sup_cuda = d_x.copy_to_host()
        print("norm gpu: ", np.linalg.norm(x_sup_cuda - np.ones(L.shape[0]), ord=np.inf))

        # x_persistent = sptrsv_supernodal_persistent(Ls, b, thread_flags.copy())
        # print("norm sync-free persistent: ", np.linalg.norm(x_persistent - np.ones(L.shape[0]), ord=np.inf))

        # x_block = sptrsv_supernodal_block_level(Ls, b, thread_flags.copy())
        # print("norm sync-free block-level: ", np.linalg.norm(x_block - np.ones(L.shape[0]), ord=np.inf))

        # x_hybrid = sptrsv_supernodal_hybrid(Ls, b, thread_flags.copy(), size_threshold=32)
        # print("norm sync-free hybrid: ", np.linalg.norm(x_hybrid - np.ones(L.shape[0]), ord=np.inf))

        # cublas_solver = SupernodalSpTrsvCuBLAS(Ls, wave_start_loc, waves)
        # x_cublas = cublas_solver.solve(b)
        # print("norm cuBLAS: ", np.linalg.norm(x_cublas - np.ones(L.shape[0]), ord=np.inf))

        # cudagraphs_solver = SupernodalSpTrsvCudaGraph(Ls, wave_start_loc, waves)
        # x_cudagraphs = cudagraphs_solver.solve(b)
        # print("norm CUDA Graphs: ", np.linalg.norm(x_cudagraphs - np.ones(L.shape[0]), ord=np.inf))

        cudagraphs_syncfree_solver = SupernodalSpTrsvCudaGraphSyncFree(Ls, dag_L_super)
        x_cudagraphs_syncfree = cudagraphs_syncfree_solver.solve(b)
        print("norm CUDA Graphs Sync-Free: ", np.linalg.norm(x_cudagraphs_syncfree - np.ones(L.shape[0]), ord=np.inf))

        # try:
        #     cudagraphs_syncfree_true_solver = TrueSyncFreeSpTrsvCudaGraph(Ls, dag_L_super)
        #     x_cudagraphs_syncfree_true = cudagraphs_syncfree_true_solver.solve(b)
        #     print("norm CUDA Graphs Sync-Free TRUE: ", np.linalg.norm(x_cudagraphs_syncfree_true - np.ones(L.shape[0]), ord=np.inf))
        # except RuntimeError as e:
        #     print(f"CUDA Graphs Sync-Free TRUE: SKIPPED - {str(e).split(chr(10))[0]}")


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
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],
        line_arg="provider",
        line_vals=method_list,
        line_names=["Python SpTRSV CSC", "Numba SpTRSV CSC Supernodal", "Numba SpTRSV CSC Supernodal Wavefront", "CudaNumba SpTRSV CSC Supernodal SyncFree", "CUDA Graphs Sync-Free"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("purple", "-"), ("olive", "-")],
        ylabel="GFLOP/S",
        plot_name="sptrsv-supernodal-performance",
        args={}
    ))

#method_list = ["python-sptrsv-csc-super", "numba-sptrsv-csc-super", "cholmod"]

# use triton benchmark
@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    matrix = spio.mmread(matrices)
    factor = cholesky(matrix)
    L = factor.L()
    b = L.dot(np.ones(L.shape[0]))  # np.random.rand(Ao.shape[0])
    Ls = convert_csc_to_supernodal(L, Relaxtion)
    # compute wavefronts of sparse matrix L (CSC)
    dag_L_super = get_supernodal_dag(Ls)
    wf_sup = compute_wavefront(dag_L_super)
    wave_start_loc, waves = compact_2d_list(wf_sup)
    thread_flags = sptrsv_csc_supernodal_analysis_1(Ls['supernodes'], Ls['ip'],
                                                            Ls['i'], Ls['col_to_super'])

    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 10, 2
    if provider == 'python-sptrsv-csc-super':
        x = b.copy()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrv_csc_supernodal(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['p'], Ls['x'], b, x), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-sptrsv-csc-super':
        x = b.copy()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_csc_supernodal_numba(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['p'], Ls['x'], b, x), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-sptrsv-csc-super-wavefront':
        x= b.copy()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_csc_supernodal_wavefront_numba(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['p'], Ls['x'], b, x, wave_start_loc, waves), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'cuda-numba-sptrsv-csc-super-syncfree':
        d_lx = cuda.to_device(Ls['x'])
        d_super = cuda.to_device(Ls['supernodes'])
        col_to_sup = cuda.to_device(Ls['col_to_super'])
        d_ip = cuda.to_device(Ls['ip'])
        d_i = cuda.to_device(Ls['i'])
        d_p = cuda.to_device(Ls['p'])
        d_thread_flags = cuda.to_device(thread_flags)
        d_x = cuda.to_device(b)
        max_width = Ls['max_width']
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sptrsv_csc_supernodal_numba_cuda(d_super, col_to_sup, d_ip, d_i, d_p, d_lx, d_x,
                                         d_thread_flags, max_width), quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'syncfree-persistent':
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: sptrsv_supernodal_persistent(Ls, b, thread_flags.copy()),
    #         quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'syncfree-block':
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: sptrsv_supernodal_block_level(Ls, b, thread_flags.copy()),
    #         quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'syncfree-hybrid':
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: sptrsv_supernodal_hybrid(Ls, b, thread_flags.copy(), 32),
    #         quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'cublas':
    #     cublas_solver = SupernodalSpTrsvCuBLAS(Ls, wave_start_loc, waves)
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: sptrsv_csc_supernodal_cublas(cublas_solver, b),
    #         quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'cudagraphs':
    #     cudagraphs_solver = SupernodalSpTrsvCudaGraph(Ls, wave_start_loc, waves)
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: cudagraphs_solver.solve(b),
    #         quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'cudagraphs-syncfree':
        cudagraphs_syncfree_solver = SupernodalSpTrsvCudaGraphSyncFree(Ls, dag_L_super)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cudagraphs_syncfree_solver.solve(b),
            quantiles=quantiles, warmup=warmpup, rep=rep)
    # if provider == 'cudagraphs-syncfree-true':
    #     try:
    #         cudagraphs_syncfree_true_solver = TrueSyncFreeSpTrsvCudaGraph(Ls, dag_L_super)
    #         ms, min_ms, max_ms = triton.testing.do_bench(
    #             lambda: cudagraphs_syncfree_true_solver.solve(b),
    #             quantiles=quantiles, warmup=warmpup, rep=rep)
    #     except RuntimeError:
    #         # CUDA Graphs not supported, return NaN
    #         ms, min_ms, max_ms = float('nan'), float('nan'), float('nan')
    # if provider == 'cholmod':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: factor.solve_L(b, use_LDLt_decomposition=False), quantiles=quantiles, warmup=warmpup, rep=rep)

    perf = lambda ms: (L.shape[0] + 2 * L.nnz) / (ms * 1e-3) / 1e9
    #print(f"seconds: {ms}")
    #print(f"max: {perf(max_ms)}, min: {perf(min_ms)}, median: {perf(ms)}")
    print(f"{provider} -- {matrices} -- seconds: {ms}")
    return perf(ms), perf(min_ms), perf(max_ms)

# Call the benchmark function
benchmark.run(print_data=True, save_path=".", show_plots=True)

