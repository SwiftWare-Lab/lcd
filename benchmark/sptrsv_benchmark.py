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
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")



mtx_list = [
"/home/mad29/projects/testdata/matrices/1138_bus.mtx",
"/home/mad29/projects/testdata/matrices/2cubes_sphere.mtx",
"/home/mad29/projects/testdata/matrices/Chem97ZtZ.mtx",
"/home/mad29/projects/testdata/matrices/Dubcova1.mtx",
"/home/mad29/projects/testdata/matrices/Dubcova2.mtx",
"/home/mad29/projects/testdata/matrices/Dubcova3.mtx",
"/home/mad29/projects/testdata/matrices/Emilia_923.mtx",
"/home/mad29/projects/testdata/matrices/Fault_639.mtx",
"/home/mad29/projects/testdata/matrices/Flan_1565.mtx",
"/home/mad29/projects/testdata/matrices/G2_circuit.mtx",
"/home/mad29/projects/testdata/matrices/G3_circuit.mtx",
"/home/mad29/projects/testdata/matrices/Geo_1438.mtx",
"/home/mad29/projects/testdata/matrices/Hook_1498.mtx",
"/home/mad29/projects/testdata/matrices/Kuu.mtx",
"/home/mad29/projects/testdata/matrices/LF10000.mtx",
"/home/mad29/projects/testdata/matrices/Muu.mtx",
"/home/mad29/projects/testdata/matrices/PFlow_742.mtx",
"/home/mad29/projects/testdata/matrices/Pres_Poisson.mtx",
"/home/mad29/projects/testdata/matrices/Serena.mtx",
"/home/mad29/projects/testdata/matrices/StocF-1465.mtx",
"/home/mad29/projects/testdata/matrices/Trefethen_2000.mtx",
"/home/mad29/projects/testdata/matrices/Trefethen_20000.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk08.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk11.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk12.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk13.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk14.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk15.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk16.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk17.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk18.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk21.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk23.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk24.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk25.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk26.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk27.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk28.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk36.mtx",
"/home/mad29/projects/testdata/matrices/bcsstk38.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm08.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm09.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm11.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm12.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm21.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm23.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm24.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm25.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm26.mtx",
"/home/mad29/projects/testdata/matrices/bcsstm39.mtx",
"/home/mad29/projects/testdata/matrices/bibd_81_2.mtx",
"/home/mad29/projects/testdata/matrices/bmw7st_1.mtx",
"/home/mad29/projects/testdata/matrices/bmwcra_1.mtx",
"/home/mad29/projects/testdata/matrices/bodyy4.mtx",
"/home/mad29/projects/testdata/matrices/bodyy5.mtx",
"/home/mad29/projects/testdata/matrices/bodyy6.mtx",
"/home/mad29/projects/testdata/matrices/bone010.mtx",
"/home/mad29/projects/testdata/matrices/boneS01.mtx",
"/home/mad29/projects/testdata/matrices/cbuckle.mtx",
"/home/mad29/projects/testdata/matrices/cfd1.mtx",
"/home/mad29/projects/testdata/matrices/cfd2.mtx",
"/home/mad29/projects/testdata/matrices/crystm01.mtx",
"/home/mad29/projects/testdata/matrices/crystm02.mtx",
"/home/mad29/projects/testdata/matrices/crystm03.mtx",
"/home/mad29/projects/testdata/matrices/ct20stif.mtx",
"/home/mad29/projects/testdata/matrices/cvxbqp1.mtx",
"/home/mad29/projects/testdata/matrices/denormal.mtx",
"/home/mad29/projects/testdata/matrices/ecology2.mtx",
"/home/mad29/projects/testdata/matrices/ex10.mtx",
"/home/mad29/projects/testdata/matrices/ex10hs.mtx",
"/home/mad29/projects/testdata/matrices/ex13.mtx",
"/home/mad29/projects/testdata/matrices/ex15.mtx",
"/home/mad29/projects/testdata/matrices/ex3.mtx",
"/home/mad29/projects/testdata/matrices/ex33.mtx",
"/home/mad29/projects/testdata/matrices/ex9.mtx",
"/home/mad29/projects/testdata/matrices/finan512.mtx",
"/home/mad29/projects/testdata/matrices/fv1.mtx",
"/home/mad29/projects/testdata/matrices/fv2.mtx",
"/home/mad29/projects/testdata/matrices/fv3.mtx",
"/home/mad29/projects/testdata/matrices/gyro.mtx",
"/home/mad29/projects/testdata/matrices/inline_1.mtx",
"/home/mad29/projects/testdata/matrices/jnlbrng1.mtx",
"/home/mad29/projects/testdata/matrices/ldoor.mtx",
"/home/mad29/projects/testdata/matrices/msdoor.mtx",
"/home/mad29/projects/testdata/matrices/nasa1824.mtx",
"/home/mad29/projects/testdata/matrices/nasa2146.mtx",
"/home/mad29/projects/testdata/matrices/nasa2910.mtx",
"/home/mad29/projects/testdata/matrices/nasa4704.mtx",
"/home/mad29/projects/testdata/matrices/nasasrb.mtx",
"/home/mad29/projects/testdata/matrices/nd12k.mtx",
"/home/mad29/projects/testdata/matrices/nd24k.mtx",
"/home/mad29/projects/testdata/matrices/nd3k.mtx",
"/home/mad29/projects/testdata/matrices/nd6k.mtx",
"/home/mad29/projects/testdata/matrices/obstclae.mtx",
"/home/mad29/projects/testdata/matrices/offshore.mtx",
"/home/mad29/projects/testdata/matrices/oilpan.mtx",
"/home/mad29/projects/testdata/matrices/olafu.mtx",
"/home/mad29/projects/testdata/matrices/parabolic_fem.mtx",
"/home/mad29/projects/testdata/matrices/pdb1HYS.mtx",
"/home/mad29/projects/testdata/matrices/plat1919.mtx",
"/home/mad29/projects/testdata/matrices/plbuckle.mtx",
"/home/mad29/projects/testdata/matrices/pwtk.mtx",
"/home/mad29/projects/testdata/matrices/qa8fm.mtx",
"/home/mad29/projects/testdata/matrices/raefsky4.mtx",
"/home/mad29/projects/testdata/matrices/s1rmq4m1.mtx",
"/home/mad29/projects/testdata/matrices/s1rmt3m1.mtx",
"/home/mad29/projects/testdata/matrices/s2rmq4m1.mtx",
"/home/mad29/projects/testdata/matrices/s2rmt3m1.mtx",
"/home/mad29/projects/testdata/matrices/s3dkq4m2.mtx",
"/home/mad29/projects/testdata/matrices/s3dkt3m2.mtx",
"/home/mad29/projects/testdata/matrices/s3rmq4m1.mtx",
"/home/mad29/projects/testdata/matrices/s3rmt3m1.mtx",
"/home/mad29/projects/testdata/matrices/s3rmt3m3.mtx",
"/home/mad29/projects/testdata/matrices/shallow_water1.mtx",
"/home/mad29/projects/testdata/matrices/shallow_water2.mtx",
"/home/mad29/projects/testdata/matrices/ship_001.mtx",
"/home/mad29/projects/testdata/matrices/ship_003.mtx",
"/home/mad29/projects/testdata/matrices/shipsec1.mtx",
"/home/mad29/projects/testdata/matrices/shipsec5.mtx",
"/home/mad29/projects/testdata/matrices/shipsec8.mtx",
"/home/mad29/projects/testdata/matrices/smt.mtx",
"/home/mad29/projects/testdata/matrices/sts4098.mtx",
"/home/mad29/projects/testdata/matrices/t2dah_e.mtx",
"/home/mad29/projects/testdata/matrices/t2dal_e.mtx",
"/home/mad29/projects/testdata/matrices/t3dl_e.mtx",
"/home/mad29/projects/testdata/matrices/ted_B.mtx",
"/home/mad29/projects/testdata/matrices/thermal1.mtx",
"/home/mad29/projects/testdata/matrices/thermal2.mtx",
"/home/mad29/projects/testdata/matrices/thermomech_TC.mtx",
"/home/mad29/projects/testdata/matrices/thermomech_TK.mtx",
"/home/mad29/projects/testdata/matrices/thermomech_dM.mtx",
"/home/mad29/projects/testdata/matrices/tmt_sym.mtx",
"/home/mad29/projects/testdata/matrices/torsion1.mtx",
"/home/mad29/projects/testdata/matrices/vanbody.mtx",
"/home/mad29/projects/testdata/matrices/wathen100.mtx",
"/home/mad29/projects/testdata/matrices/wathen120.mtx",
"/home/mad29/projects/testdata/matrices/x104.mtx"
]
method_list = ["triton", "triton-sync-free", "triton-csr-wavefront", "triton-csr-sync-free", "cusparse", "numba-sptrsv", "numba-sync-free", "numba-csr-wavefront", "numba-csr-sync-free", "triton-spmv"]
# mtx_list = get_matrix_list("/home/mad29/projects/gpu-backend/testlist.txt", "")
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

        # map_ptr, map_vals, outgoing_edges = incoming_edge_csr(A.tocsr())
        # y_sync_free_csr = sptrsv_sync_free_triton_csr(val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz,
        #                                               torch.tensor(outgoing_edges, dtype=torch.int32, device=DEVICE),
        #                                                torch.tensor(map_ptr, dtype=torch.int32, device=DEVICE),
        #                                                torch.tensor(map_vals, dtype=torch.int32, device=DEVICE))

        A_csr = A.tocsr()
        val_f32_csr     = torch.tensor(A_csr.data,    dtype=REAL_TYPE, device=DEVICE)
        indices_i32_csr = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32_csr  = torch.tensor(A_csr.indptr,  dtype=torch.int32, device=DEVICE)

        map_ptr, map_vals, outgoing_edges = incoming_edge_csr(A_csr)

        y_sync_free_csr = sptrsv_sync_free_triton_csr(
            val_f32_csr, indices_i32_csr, indptr_i32_csr, b_f32, n, n, A.nnz,
            torch.tensor(outgoing_edges, dtype=torch.int32, device=DEVICE),
            torch.tensor(map_ptr, dtype=torch.int32, device=DEVICE),
            torch.tensor(map_vals, dtype=torch.int32, device=DEVICE))

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

        # dep_count_csc1 = (A.indptr[1:] - A.indptr[:-1]).astype(np.int32)
        # dep_count_csc1[dep_count_csc1 == 0] = 1
        # y_numba_gpu_sync_free = sptrsc_numba_csc_sync_free(
        #     d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, dep_count_csc1)
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

        # Test triton-csr-wavefront
        A_csr = A.tocsr()
        val_f32_csr = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        indices_i32_csr = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32_csr = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts_csr = compute_wavefront(A)
        wave_start_loc_csr, waves_csr = compact_2d_list(wavefronts_csr)
        waves_gpu_csr = torch.tensor(waves_csr, dtype=torch.int32, device=DEVICE)
        y_csr_wave = sptrsv_triton_csr(val_f32_csr, indices_i32_csr, indptr_i32_csr, b_f32, n, n, A_csr.nnz, wave_start_loc_csr, waves_gpu_csr)
        
        if np.allclose(x_cpu, y_csr_wave.cpu().numpy(), atol=1e-5):
            print("\ntriton-csr-wavefront Solutions match within tolerance!")
            method_correctness[mat_name]['triton-csr-wavefront'] = 1
        else:
            print("\nWarning: triton-csr-wavefront Solutions do not match!")

        if np.allclose(x_cpu, y_sync_free.cpu().numpy(), atol=1e-5):
            print("\ntriton-sync-free Solutions match within tolerance!")
            method_correctness[mat_name]['triton-sync-free'] = 1
        else:
            print("\nWarning: triton-sync-free Solutions do not match!")

        if np.allclose(x_cpu, y_sync_free_csr.cpu().numpy(), atol=1e-5):
            # print(x_cpu)
            print("\ntriton-csr-sync-free Solutions match within tolerance!")
            method_correctness[mat_name]['triton-csr-sync-free'] = 1
        else:
            # print(y_sync_free_csr.cpu().numpy())
            # print(x_cpu)
            print("\nWarning: triton-csr-sync-free Solutions do not match!")

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

            # abs_err = np.max(np.abs(x_cpu - y_numba_cpu_sync_free))
            # rel_err = np.linalg.norm(x_cpu - y_numba_cpu_sync_free) / np.linalg.norm(x_cpu)
            # print(f"  Max abs error: {abs_err:.3e}")
            # print(f"  Relative L2 error: {rel_err:.3e}")

        # Test numba-csr-wavefront
        A_csr_numba = A.tocsr()
        d_data_csr = cuda.to_device(A_csr_numba.data)
        d_indices_csr = cuda.to_device(A_csr_numba.indices)
        d_indptr_csr = cuda.to_device(A_csr_numba.indptr)
        d_b_csr = cuda.to_device(b)
        wavefronts_csr_numba = compute_wavefront(A)
        wave_start_loc_csr_numba, waves_csr_numba = compact_2d_list(wavefronts_csr_numba)
        waves_gpu_csr_numba = cuda.to_device(waves_csr_numba)
        y_numba_csr_wave = sptrsc_numba_csr_wavefront(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, wave_start_loc_csr_numba, waves_gpu_csr_numba)
        y_numba_csr_wave_cpu = y_numba_csr_wave.copy_to_host()
        
        if np.allclose(x_cpu, y_numba_csr_wave_cpu, atol=1e-5):
            print("\nnumba-csr-wavefront Solutions match within tolerance!")
            method_correctness[mat_name]['numba-csr-wavefront'] = 1
        else:
            print("\nWarning: numba-csr-wavefront Solutions do not match!")

        # Test numba-csr-sync-free
        in_edges_csr = incoming_edge_csr_rows(A_csr_numba)
        succ_offsets_csr, succ_indices_csr = build_csr_successors(A_csr_numba)
        y_numba_csr_sync_free = sptrsc_numba_csr_sync_free(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, in_edges_csr, succ_offsets_csr, succ_indices_csr)
        y_numba_csr_sync_free_cpu = y_numba_csr_sync_free.copy_to_host()
        
        if np.allclose(x_cpu, y_numba_csr_sync_free_cpu, atol=1e-5):
            print("\nnumba-csr-sync-free Solutions match within tolerance!")
            method_correctness[mat_name]['numba-csr-sync-free'] = 1
        else:
            print("\nWarning: numba-csr-sync-free Solutions do not match!")

        # print (x_cpu[:5])
        # print (x_gpu[:5])
        # print (y[:5])
        # print ("Triton sync free", y_sync_free[:50])
        # print (y_numba_cpu[:5])
        # print ("numba synch-free", y_numba_cpu_sync_free[:5])
    return matrix_signtures, method_correctness


# mat_signatures, method_correctness = correctness_test()
# # convert to df and save as csv
# import pandas as pd
# df = pd.DataFrame(mat_signatures)
# df.to_csv("matrix_signatures.csv")

# df = pd.DataFrame(method_correctness)
# df.to_csv("method_correctness.csv")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],  # Argument names to use as an x-axis for the plot
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=method_list,  # Label name for the lines
        line_names=["Triton-Wavefront-CSC", "Triton-Sync-free-CSC", "Triton-Wavefront-CSR", "Triton-Sync-free-CSR", "[REF] cuSparse", "Numba-Wavefront-CSC", "Numba-Sync-free-CSC", "Numba-Wavefront-CSR", "Numba-Sync-free-CSR", "[REF] Triton-SpMV"],
        styles=[("green", "-"), ("blue", "-"), ("cyan", "-"), ("magenta", "-"), ("yellow", "-"), ("red", "-"), ("purple", "-"), ("brown", "-"), ("pink", "-"), ("orange", "--")],  # Visual styles for the lines
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
    if provider == 'triton-csr-wavefront':
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_triton_csr(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'triton-csr-sync-free':
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        map_ptr, map_vals, outgoing_edges = incoming_edge_csr(A_csr)
        dep_count_csr = torch.tensor(outgoing_edges, dtype=torch.int32, device=DEVICE)
        map_ptr_gpu = torch.tensor(map_ptr, dtype=torch.int32, device=DEVICE)
        map_vals_gpu = torch.tensor(map_vals, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_sync_free_triton_csr(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A_csr.nnz, dep_count_csr, map_ptr_gpu, map_vals_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
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
    if provider == 'numba-csr-wavefront':
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsc_numba_csr_wavefront(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-csr-sync-free':
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsc_numba_csr_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices), quantiles=quantiles, warmup=warmpup, rep=rep)
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
benchmark.run(print_data=True, save_path=".", show_plots=True)

