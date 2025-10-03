import numpy as np
import torch


from bench_utils import *
from spmv import *


NUM_RUNS=30

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


from sptrsv import *
# Set the compute capability to match your GPU
# Common values are '75' (for compute capability 7.5), '86' (for 8.6), etc.
# import os
# os.environ['NUMBAPRO_NVVM_NUM_THREADS'] = '1'
# os.environ['CUDA_ENABLE_COMPUTE_CAPABILITY'] = '84'  # Adjust this value based on your GPU

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")



mtx_list = [
    # "/home/kazem/Downloads/LFAT5/LFAT5.mtx", "/home/kazem/Downloads/ex5/ex5.mtx", "/home/kazem/Downloads/mesh1e1/mesh1e1.mtx","/home/kazem/Downloads/nos4/nos4.mtx",
    # "/home/kazem/Downloads/bcsstk03/bcsstk03.mtx",
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx"
            ]
method_list = ["triton", "triton-sync-free", "triton-sync-free-csr", "cusparse",  "numba-sptrsv", "numba-sync-free", "triton-spmv"]
mtx_list = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/SPD/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/SPD/")
#mtx_list = mtx_list[60:]

def correctness_test():
    matrix_signtures = []
    method_correctness = {}
    for mtx_matrix_path in mtx_list:
        print(f"------------ matrix: {mtx_matrix_path}")
        # read matrix from file
        matrix = scio.mmread(mtx_matrix_path)
        n = matrix.shape[0]
        # convert matrix to csc
        A = matrix.tocsc()
        if A.data.dtype != np.float64:
            print(f"Matrix {mtx_matrix_path} is not double precision")
            A.data = A.data.astype(np.float64)
        # makke it sparse lower triangular
        A = tril(A, format="csc")
        # add diagonals with 100
        A.setdiag(A.diagonal())
        mat_name = mtx_matrix_path.split("/")[-1]
        matrix_signtures.append(get_matrix_signature(A, mat_name))
        b = A.dot(np.ones(n))  # Dense vector b
        # simple solver
        #b_simp = simple_sptrsv_csc(A, b)
        #print("b:", b_simp)
        # Run CPU solver
        x_cpu, cpu_median = cpu_solver(A, b)
        # Run GPU solver
        x_gpu, gpu_median = gpu_solver(A, b)

        # Run Triton solver
        max_col_size = max(A.getnnz(axis=0))
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        y = sptrsv_triton(val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz, wave_start_loc, waves_gpu)
        #print("y triton csc:", y[:])

        in_edge = incoming_edge(A)
        #b_sfree_cpu = simple_sptrsv_csc_syncfree(A, b)
        # print("b:", b_sfree_cpu[:10])
        #A_dens = np.array(A.todense())
        #print( f"matrix: {mtx_matrix_path} has: {in_edge}")
        dep_count = torch.tensor(in_edge, dtype=torch.int32, device=DEVICE)
        # conver max col to constant
        if max_col_size < 31:
            BLOCKSIZE = 32
        elif max_col_size < 63:
            BLOCKSIZE = 64
        elif max_col_size < 255:
            BLOCKSIZE = 256
        elif max_col_size < 511:
            BLOCKSIZE = 512
        else:
            BLOCKSIZE = 1024
        y_sync_free = sptrsv_sync_free_triton(val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz, dep_count, BLOCKSIZE)

        map_ptr, map_vals, outgoing_edges = incoming_edge_csr(A.tocsr())
        y_sync_free_csr = sptrsv_sync_free_triton_csr(val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz,
                                                      torch.tensor(outgoing_edges, dtype=torch.int32, device=DEVICE),
                                                       torch.tensor(map_ptr, dtype=torch.int32, device=DEVICE),
                                                       torch.tensor(map_vals, dtype=torch.int32, device=DEVICE),
                                                       BLOCKSIZE)


        print(" triton solvers are done!")
        ########### numba sptrsv
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_b = cuda.to_device(b)
        #wavefronts = compute_wavefront(A)
        #wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        y_numba_gpu = sptrsc_numba_csc(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu)
        y_numba_cpu = y_numba_gpu.copy_to_host()

        #d_in_edges = cuda.to_device(in_edge)
        y_numba_gpu_sync_free = sptrsc_numba_csc_sync_free(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge)
        y_numba_cpu_sync_free = y_numba_gpu_sync_free.copy_to_host()

        # Validate results
        method_correctness[mat_name] = {}
        for method in method_list:
            method_correctness[mat_name][method] = 0
        if np.allclose(x_cpu, x_gpu, atol=1e-5):
            print("\ncusparse Solutions match within tolerance!")
            method_correctness[mat_name]['cusparse'] = 1
        else:
            print("\nWarning: cusparse Solutions do not match!")

        if np.allclose(x_cpu, y.cpu().numpy(), atol=1e-5):
            print("\ntriton wavefront Solutions match within tolerance!")
            method_correctness[mat_name]['triton'] = 1
        else:
            print("\nWarning: triton wavefront Solutions do not match!")

        if np.allclose(x_cpu, y_sync_free.cpu().numpy(), atol=1e-5):
            print("\ntriton-sync-free Solutions match within tolerance!")
            method_correctness[mat_name]['triton-sync-free'] = 1
        else:
            print("\nWarning: triton-sync-free Solutions do not match!")

        if np.allclose(x_cpu, y_sync_free_csr.cpu().numpy(), atol=1e-5):
            # print(x_cpu)
            print("\ntriton-sync-free-csr Solutions match within tolerance!")
            method_correctness[mat_name]['triton-sync-free_csr'] = 1
        else:
            print(y_sync_free_csr.cpu().numpy())
            print(x_cpu)
            print("\nWarning: triton-sync-free-csr Solutions do not match!")

        if np.allclose(x_cpu, y_numba_cpu, atol=1e-5):
            print("\nnumba-sptrsv Solutions match within tolerance!")
            method_correctness[mat_name]['numba-sptrsv'] = 1
        else:
            print("\nWarning: numba-sptrsv Solutions do not match!")

        if np.allclose(x_cpu, y_numba_cpu_sync_free, atol=1e-5):
            print("\nnumba-sptrsv-sync-free Solutions match within tolerance!")
            method_correctness[mat_name]['numba-sync-free'] = 1
        else:
            print("\nWarning: numba-sptrsv-sync-free Solutions do not match!")

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
# benchmark.run(print_data=True, save_path=".", show_plots=True)

