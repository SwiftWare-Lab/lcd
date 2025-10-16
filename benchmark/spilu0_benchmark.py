import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
import numpy as np
import torch
import scipy.io as scio
import triton
from numba import cuda
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spilu
from numba.core.errors import NumbaPerformanceWarning

from bench_utils import *
from spilu0 import *

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

NUM_RUNS = 30
DEVICE = torch.device("cuda:0")

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")

mtx_list = [
# "/home/mad29/projects/testdata/matrices/1138_bus.mtx",
# "/home/mad29/projects/testdata/matrices/2cubes_sphere.mtx",
# "/home/mad29/projects/testdata/matrices/Chem97ZtZ.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova1.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova2.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova3.mtx",
# "/home/mad29/projects/testdata/matrices/Emilia_923.mtx",
# "/home/mad29/projects/testdata/matrices/Fault_639.mtx",
# "/home/mad29/projects/testdata/matrices/Flan_1565.mtx",
"/home/mad29/projects/testdata/matrices/G2_circuit.mtx",
# "/home/mad29/projects/testdata/matrices/G3_circuit.mtx",
# "/home/mad29/projects/testdata/matrices/Geo_1438.mtx",
# "/home/mad29/projects/testdata/matrices/Hook_1498.mtx",
# "/home/mad29/projects/testdata/matrices/Kuu.mtx",
# "/home/mad29/projects/testdata/matrices/LF10000.mtx",
# "/home/mad29/projects/testdata/matrices/Muu.mtx",
# "/home/mad29/projects/testdata/matrices/PFlow_742.mtx",
# "/home/mad29/projects/testdata/matrices/Pres_Poisson.mtx",
# "/home/mad29/projects/testdata/matrices/Serena.mtx",
# "/home/mad29/projects/testdata/matrices/StocF-1465.mtx",
# "/home/mad29/projects/testdata/matrices/Trefethen_2000.mtx",
# "/home/mad29/projects/testdata/matrices/Trefethen_20000.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk08.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk11.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk12.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk13.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk14.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk15.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk16.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk17.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk18.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk21.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk23.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk24.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk25.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk26.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk27.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk28.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk36.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk38.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm08.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm09.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm11.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm12.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm21.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm23.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm24.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm25.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm26.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstm39.mtx",
# "/home/mad29/projects/testdata/matrices/bibd_81_2.mtx",
# "/home/mad29/projects/testdata/matrices/bmw7st_1.mtx",
# "/home/mad29/projects/testdata/matrices/bmwcra_1.mtx",
# "/home/mad29/projects/testdata/matrices/bodyy4.mtx",
# "/home/mad29/projects/testdata/matrices/bodyy5.mtx",
# "/home/mad29/projects/testdata/matrices/bodyy6.mtx",
# "/home/mad29/projects/testdata/matrices/bone010.mtx",
# "/home/mad29/projects/testdata/matrices/boneS01.mtx",
# "/home/mad29/projects/testdata/matrices/cbuckle.mtx",
# "/home/mad29/projects/testdata/matrices/cfd1.mtx",
# "/home/mad29/projects/testdata/matrices/cfd2.mtx",
# "/home/mad29/projects/testdata/matrices/crystm01.mtx",
# "/home/mad29/projects/testdata/matrices/crystm02.mtx",
# "/home/mad29/projects/testdata/matrices/crystm03.mtx",
# "/home/mad29/projects/testdata/matrices/ct20stif.mtx",
# "/home/mad29/projects/testdata/matrices/cvxbqp1.mtx",
# "/home/mad29/projects/testdata/matrices/denormal.mtx",
# "/home/mad29/projects/testdata/matrices/ecology2.mtx",
# "/home/mad29/projects/testdata/matrices/ex10.mtx",
# "/home/mad29/projects/testdata/matrices/ex10hs.mtx",
# "/home/mad29/projects/testdata/matrices/ex13.mtx",
# "/home/mad29/projects/testdata/matrices/ex15.mtx",
# "/home/mad29/projects/testdata/matrices/ex3.mtx",
# "/home/mad29/projects/testdata/matrices/ex33.mtx",
# "/home/mad29/projects/testdata/matrices/ex9.mtx",
# "/home/mad29/projects/testdata/matrices/finan512.mtx",
# "/home/mad29/projects/testdata/matrices/fv1.mtx",
# "/home/mad29/projects/testdata/matrices/fv2.mtx",
# "/home/mad29/projects/testdata/matrices/fv3.mtx",
# "/home/mad29/projects/testdata/matrices/gyro.mtx",
# "/home/mad29/projects/testdata/matrices/inline_1.mtx",
# "/home/mad29/projects/testdata/matrices/jnlbrng1.mtx",
# "/home/mad29/projects/testdata/matrices/ldoor.mtx",
# "/home/mad29/projects/testdata/matrices/msdoor.mtx",
# "/home/mad29/projects/testdata/matrices/nasa1824.mtx",
# "/home/mad29/projects/testdata/matrices/nasa2146.mtx",
# "/home/mad29/projects/testdata/matrices/nasa2910.mtx",
# "/home/mad29/projects/testdata/matrices/nasa4704.mtx",
# "/home/mad29/projects/testdata/matrices/nasasrb.mtx",
# "/home/mad29/projects/testdata/matrices/nd12k.mtx",
# "/home/mad29/projects/testdata/matrices/nd24k.mtx",
# "/home/mad29/projects/testdata/matrices/nd3k.mtx",
# "/home/mad29/projects/testdata/matrices/nd6k.mtx",
# "/home/mad29/projects/testdata/matrices/obstclae.mtx",
# "/home/mad29/projects/testdata/matrices/offshore.mtx",
# "/home/mad29/projects/testdata/matrices/oilpan.mtx",
# "/home/mad29/projects/testdata/matrices/olafu.mtx",
# "/home/mad29/projects/testdata/matrices/parabolic_fem.mtx",
# "/home/mad29/projects/testdata/matrices/pdb1HYS.mtx",
# "/home/mad29/projects/testdata/matrices/plat1919.mtx",
# "/home/mad29/projects/testdata/matrices/plbuckle.mtx",
# "/home/mad29/projects/testdata/matrices/pwtk.mtx",
# "/home/mad29/projects/testdata/matrices/qa8fm.mtx",
# "/home/mad29/projects/testdata/matrices/raefsky4.mtx",
# "/home/mad29/projects/testdata/matrices/s1rmq4m1.mtx",
# "/home/mad29/projects/testdata/matrices/s1rmt3m1.mtx",
# "/home/mad29/projects/testdata/matrices/s2rmq4m1.mtx",
# "/home/mad29/projects/testdata/matrices/s2rmt3m1.mtx",
# "/home/mad29/projects/testdata/matrices/s3dkq4m2.mtx",
# "/home/mad29/projects/testdata/matrices/s3dkt3m2.mtx",
# "/home/mad29/projects/testdata/matrices/s3rmq4m1.mtx",
# "/home/mad29/projects/testdata/matrices/s3rmt3m1.mtx",
# "/home/mad29/projects/testdata/matrices/s3rmt3m3.mtx",
# "/home/mad29/projects/testdata/matrices/shallow_water1.mtx",
# "/home/mad29/projects/testdata/matrices/shallow_water2.mtx",
# "/home/mad29/projects/testdata/matrices/ship_001.mtx",
# "/home/mad29/projects/testdata/matrices/ship_003.mtx",
# "/home/mad29/projects/testdata/matrices/shipsec1.mtx",
# "/home/mad29/projects/testdata/matrices/shipsec5.mtx",
# "/home/mad29/projects/testdata/matrices/shipsec8.mtx",
# "/home/mad29/projects/testdata/matrices/smt.mtx",
# "/home/mad29/projects/testdata/matrices/sts4098.mtx",
# "/home/mad29/projects/testdata/matrices/t2dah_e.mtx",
# "/home/mad29/projects/testdata/matrices/t2dal_e.mtx",
# "/home/mad29/projects/testdata/matrices/t3dl_e.mtx",
# "/home/mad29/projects/testdata/matrices/ted_B.mtx",
# "/home/mad29/projects/testdata/matrices/thermal1.mtx",
# "/home/mad29/projects/testdata/matrices/thermal2.mtx",
# "/home/mad29/projects/testdata/matrices/thermomech_TC.mtx",
# "/home/mad29/projects/testdata/matrices/thermomech_TK.mtx",
# "/home/mad29/projects/testdata/matrices/thermomech_dM.mtx",
# "/home/mad29/projects/testdata/matrices/tmt_sym.mtx",
# "/home/mad29/projects/testdata/matrices/torsion1.mtx",
# "/home/mad29/projects/testdata/matrices/vanbody.mtx",
# "/home/mad29/projects/testdata/matrices/wathen100.mtx",
# "/home/mad29/projects/testdata/matrices/wathen120.mtx",
# "/home/mad29/projects/testdata/matrices/x104.mtx"
]

method_list = [
    "numba-csr-wavefront", "numba-csr-sync-free",
    # "numba-csc-wavefront", "numba-csc-sync-free",
]

def validate_ilu0_factorization(A_result, A_original, tol=1e-3):
    # Validate ILU0 by checking if L@U matches A at original nonzero positions
    # Extract L and U from the factorization
    # For ILU0 in-place: lower triangle (with unit diagonal) is L, upper triangle is U
    n = A_result.shape[0]
    
    L_data = []
    L_row = []
    L_col = []
    
    U_data = []
    U_row = []
    U_col = []
    
    A_csr = A_result.tocsr()
    for i in range(n):
        L_data.append(1.0)
        L_row.append(i)
        L_col.append(i)
        
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        for ptr in range(row_start, row_end):
            j = A_csr.indices[ptr]
            val = A_csr.data[ptr]
            if j < i:
                # Lower
                L_data.append(val)
                L_row.append(i)
                L_col.append(j)
            else:
                # Upper (including diagonal)
                U_data.append(val)
                U_row.append(i)
                U_col.append(j)
    
    from scipy.sparse import coo_matrix
    L = coo_matrix((L_data, (L_row, L_col)), shape=(n, n)).tocsr()
    U = coo_matrix((U_data, (U_row, U_col)), shape=(n, n)).tocsr()
    
    LU = L @ U
    
    A_orig_csr = A_original.tocsr()
    errors = []
    
    for i in range(n):
        row_start = A_orig_csr.indptr[i]
        row_end = A_orig_csr.indptr[i + 1]
        for ptr in range(row_start, row_end):
            j = A_orig_csr.indices[ptr]
            a_val = A_orig_csr.data[ptr]
            lu_val = LU[i, j]
            errors.append(abs(lu_val - a_val))
    
    if len(errors) == 0:
        return True
    
    rel_error = np.linalg.norm(errors) / (np.linalg.norm(A_orig_csr.data) + 1e-12)
    
    return rel_error < tol

def correctness_test():
    matrix_signatures = []
    method_correctness = {}
    
    for mtx_matrix_path in mtx_list:
        print(f"------------ matrix: {mtx_matrix_path}")
        matrix = scio.mmread(mtx_matrix_path)
        n = matrix.shape[0]
        A = matrix.tocsr()
        
        if A.data.dtype != np.float64:
            A.data = A.data.astype(np.float64)
        
        A.setdiag(A.diagonal() + 1e-12)
        mat_name = mtx_matrix_path.split("/")[-1]
        matrix_signatures.append(get_matrix_signature(A.tocsc(), mat_name))
        
        A_test = prepare_for_ilu0(A)
        
        method_correctness[mat_name] = {}
        for method in method_list:
            method_correctness[mat_name][method] = 0

        try:
            A_copy = A_test.copy()
            wavefronts = compute_ilu_wavefront_csr(A_copy)
            wave_start_loc, waves = compact_2d_list(wavefronts)
            waves_gpu = cuda.to_device(waves)
            
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            
            ilu0_csr_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu)
            
            result_data = d_data.copy_to_host()
            A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
            
            if validate_ilu0_factorization(A_result, A_test):
                print("numba-csr-wavefront: correct")
                method_correctness[mat_name]['numba-csr-wavefront'] = 1
            else:
                print("Warning: numba-csr-wavefront incorrect")
        except Exception as e:
            print(f"numba-csr-wavefront failed: {e}")

        try:
            A_copy = A_test.copy()
            dep_counts = compute_ilu_dependency_counts_csr(A_copy)
            succ_offsets, succ_indices = build_successors_from_csr(A_copy)
            
            d_succ_offsets = cuda.to_device(succ_offsets)
            d_succ_indices = cuda.to_device(succ_indices)
            
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            d_dep_counts = cuda.to_device(dep_counts.copy())
            
            ilu0_csr_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices)
            
            result_data = d_data.copy_to_host()
            A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
            
            if validate_ilu0_factorization(A_result, A_test):
                print("numba-csr-sync-free: correct")
                method_correctness[mat_name]['numba-csr-sync-free'] = 1
            else:
                print("Warning: numba-csr-sync-free incorrect")
        except Exception as e:
            print(f"numba-csr-sync-free failed: {e}")
        
        # # Test numba-csc-wavefront
        # try:
        #     A_csc = A_test.tocsc()
        #     A_copy = A_csc.copy()
        #     
        #     wavefronts = compute_ilu_wavefront_csc(A_copy)
        #     wave_start_loc, waves = compact_2d_list(wavefronts)
        #     waves_gpu = cuda.to_device(waves)
        #     
        #     d_data = cuda.to_device(A_copy.data)
        #     d_indices = cuda.to_device(A_copy.indices)
        #     d_indptr = cuda.to_device(A_copy.indptr)
        #     
        #     ilu0_csc_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu)
        #     
        #     result_data = d_data.copy_to_host()
        #     A_result = csc_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
        #     
        #     if validate_ilu0_factorization(A_result, A_csc):
        #         print("numba-csc-wavefront: correct")
        #         method_correctness[mat_name]['numba-csc-wavefront'] = 1
        #     else:
        #         print("Warning: numba-csc-wavefront incorrect")
        # except Exception as e:
        #     print(f"numba-csc-wavefront failed: {e}")
        
        # # Test numba-csc-sync-free
        # try:
        #     A_csc = A_test.tocsc()
        #     A_copy = A_csc.copy()
        #     
        #     dep_counts = compute_ilu_dependency_counts_csc(A_copy)
        #     succ_offsets, succ_indices = build_successors_from_csc(A_copy)
        #     
        #     d_succ_offsets = cuda.to_device(succ_offsets)
        #     d_succ_indices = cuda.to_device(succ_indices)
        #     
        #     d_data = cuda.to_device(A_copy.data)
        #     d_indices = cuda.to_device(A_copy.indices)
        #     d_indptr = cuda.to_device(A_copy.indptr)
        #     d_dep_counts = cuda.to_device(dep_counts.copy())
        #     
        #     ilu0_csc_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices)
        #     
        #     result_data = d_data.copy_to_host()
        #     A_result = csc_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
        #     
        #     if validate_ilu0_factorization(A_result, A_csc):
        #         print("numba-csc-sync-free: correct")
        #         method_correctness[mat_name]['numba-csc-sync-free'] = 1
        #     else:
        #         print("Warning: numba-csc-sync-free incorrect")
        # except Exception as e:
        #     print(f"numba-csc-sync-free failed: {e}")
    
    return matrix_signatures, method_correctness

mat_signatures, method_correctness = correctness_test()

import pandas as pd
df = pd.DataFrame(mat_signatures)
df.to_csv("ilu0_matrix_signatures.csv")

df = pd.DataFrame(method_correctness)
df.to_csv("ilu0_method_correctness.csv")

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],
        line_arg="provider",
        line_vals=method_list,
        line_names=[
            "Numba CSR Wavefront", "Numba CSR Sync-free",
            # "Numba CSC Wavefront", "Numba CSC Sync-free"
        ],
        styles=[
            ("red", "-"), ("purple", "-"),
            # ("red", "--"), ("purple", "--")
        ],
        ylabel="GFLOP/S",
        plot_name="ilu0-performance",
        args={}
    )
)

@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    matrix = scio.mmread(matrices)
    n = matrix.shape[0]
    A = matrix.tocsr()
    
    if A.data.dtype != np.float64:
        A.data = A.data.astype(np.float64)
    
    A.setdiag(A.diagonal() + 1e-12)
    A = prepare_for_ilu0(A)
    
    quantiles = [0.5, 0.2, 0.8]
    rep, warmup = 100, 25
    threads_per_block = 1

    if provider == 'numba-csr-wavefront':
        wavefronts = compute_ilu_wavefront_csr(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        
        def run_numba_csr_wave():
            A_copy = A.copy()
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            ilu0_csr_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu, threads_per_block)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_wave, quantiles=quantiles, warmup=warmup, rep=rep)

    # elif provider == 'numba-csc-wavefront':
    #     A_csc = A.tocsc()
    #     wavefronts = compute_ilu_wavefront_csc(A_csc)
    #     wave_start_loc, waves = compact_2d_list(wavefronts)
    #     waves_gpu = cuda.to_device(waves)
    #     
    #     def run_numba_csc_wave():
    #         A_copy = A_csc.copy()
    #         d_data = cuda.to_device(A_copy.data)
    #         d_indices = cuda.to_device(A_copy.indices)
    #         d_indptr = cuda.to_device(A_copy.indptr)
    #         ilu0_csc_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu, threads_per_block)
    #     
    #     ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csc_wave, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'numba-csr-sync-free':
        dep_counts = compute_ilu_dependency_counts_csr(A)
        succ_offsets, succ_indices = build_successors_from_csr(A)
        d_succ_offsets = cuda.to_device(succ_offsets)
        d_succ_indices = cuda.to_device(succ_indices)
        
        def run_numba_csr_sync():
            A_copy = A.copy()
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            d_dep_counts = cuda.to_device(dep_counts.copy())
            ilu0_csr_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices, threads_per_block)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_sync, quantiles=quantiles, warmup=warmup, rep=rep)

    # elif provider == 'numba-csc-sync-free':
    #     A_csc = A.tocsc()
    #     dep_counts = compute_ilu_dependency_counts_csc(A_csc)
    #     succ_offsets, succ_indices = build_successors_from_csc(A_csc)
    #     d_succ_offsets = cuda.to_device(succ_offsets)
    #     d_succ_indices = cuda.to_device(succ_indices)
    #     
    #     def run_numba_csc_sync():
    #         A_copy = A_csc.copy()
    #         d_data = cuda.to_device(A_copy.data)
    #         d_indices = cuda.to_device(A_copy.indices)
    #         d_indptr = cuda.to_device(A_copy.indptr)
    #         d_dep_counts = cuda.to_device(dep_counts.copy())
    #         ilu0_csc_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices, threads_per_block)
    #     
    #     ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csc_sync, quantiles=quantiles, warmup=warmup, rep=rep)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    perf = lambda t_ms: (2 * A.nnz) / (t_ms * 1e-3) / 1e9
    print(f"{provider} -- {matrices} -- time: {ms:.3f} ms")
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".", show_plots=False)
