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
import pandas as pd
from bench_utils import *
from spilu0 import *

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

NUM_RUNS = 30
DEVICE = torch.device("cuda:0")

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
# "/home/mad29/projects/testdata/matrices/Emilia_923.mtx",
# "/home/mad29/projects/testdata/matrices/Fault_639.mtx",
# "/home/mad29/projects/testdata/matrices/Flan_1565.mtx",
# "/home/mad29/projects/testdata/matrices/G2_circuit.mtx",
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
# "/home/mad29/projects/testdata/matrices/obstclae.mtx",
# "/home/mad29/projects/testdata/matrices/offshore.mtx",
# "/home/mad29/projects/testdata/matrices/oilpan.mtx",
# "/home/mad29/projects/testdata/matrices/s1rmq4m1.mtx",
# "/home/mad29/projects/testdata/matrices/s1rmt3m1.mtx",
# "/home/mad29/projects/testdata/matrices/s2rmq4m1.mtx",
# "/home/mad29/projects/testdata/matrices/s2rmt3m1.mtx",
# "/home/mad29/projects/testdata/matrices/s3dkq4m2.mtx",
# "/home/mad29/projects/testdata/matrices/s3dkt3m2.mtx",
# "/home/mad29/projects/testdata/matrices/s3rmq4m1.mtx",
# "/home/mad29/projects/testdata/matrices/s3rmt3m1.mtx",
# "/home/mad29/projects/testdata/matrices/s3rmt3m3.mtx",
# "/home/mad29/projects/testdata/matrices/thermal1.mtx",
# "/home/mad29/projects/testdata/matrices/thermal2.mtx",
# "/home/mad29/projects/testdata/matrices/thermomech_TC.mtx",
# "/home/mad29/projects/testdata/matrices/thermomech_TK.mtx",
# "/home/mad29/projects/testdata/matrices/thermomech_dM.mtx",
# "/home/mad29/projects/testdata/matrices/torsion1.mtx",
# "/home/mad29/projects/testdata/matrices/vanbody.mtx",
# "/home/mad29/projects/testdata/matrices/wathen100.mtx",
# "/home/mad29/projects/testdata/matrices/wathen120.mtx",
# "/home/mad29/projects/testdata/matrices/x104.mtx"
]

method_list = [
    "triton-csr-wavefront", "triton-csr-sync-free",
    "numba-csr-wavefront", "numba-csr-sync-free",
    "numba-csr-wavefront-mapping", "numba-csr-wavefront-unrolling", "numba-csr-wavefront-mapping-unrolling",
    "numba-csr-sync-free-mapping", "numba-csr-sync-free-unrolling", "numba-csr-sync-free-mapping-unrolling",
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
        
        if n > 50000:
            print(f"Skipping matrix with dimension {n} > 50000")
            continue
        
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

        # Test Triton CSR wavefront
        try:
            import torch
            A_copy = A_test.copy()
            wavefronts = compute_ilu_wavefront_csr(A_copy)
            wave_start_loc, waves = compact_2d_list(wavefronts)
            waves_gpu = torch.tensor(waves, dtype=torch.int32, device='cuda')
            diag_pos = find_diagonal_positions(A_copy)
            diag_pos_gpu = torch.tensor(diag_pos, dtype=torch.int32, device='cuda')
            
            data_gpu = torch.tensor(A_copy.data, dtype=torch.float64, device='cuda')
            indices_gpu = torch.tensor(A_copy.indices, dtype=torch.int32, device='cuda')
            indptr_gpu = torch.tensor(A_copy.indptr, dtype=torch.int32, device='cuda')
            
            ilu0_csr_triton_wavefront(data_gpu, indices_gpu, indptr_gpu, n, wave_start_loc, waves_gpu, diag_pos_gpu)
            
            result_data = data_gpu.cpu().numpy()
            A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
            
            if validate_ilu0_factorization(A_result, A_test):
                print("triton-csr-wavefront: correct")
                method_correctness[mat_name]['triton-csr-wavefront'] = 1
            else:
                print("Warning: triton-csr-wavefront incorrect")
        except Exception as e:
            print(f"triton-csr-wavefront failed: {e}")

        # Test Triton CSR sync-free
        try:
            import torch
            A_copy = A_test.copy()
            dep_counts = compute_ilu_dependency_counts_csr(A_copy)
            succ_offsets, succ_indices = build_successors_from_csr(A_copy)
            diag_pos = find_diagonal_positions(A_copy)
            
            data_gpu = torch.tensor(A_copy.data, dtype=torch.float64, device='cuda')
            indices_gpu = torch.tensor(A_copy.indices, dtype=torch.int32, device='cuda')
            indptr_gpu = torch.tensor(A_copy.indptr, dtype=torch.int32, device='cuda')
            succ_offsets_gpu = torch.tensor(succ_offsets, dtype=torch.int32, device='cuda')
            succ_indices_gpu = torch.tensor(succ_indices, dtype=torch.int32, device='cuda')
            diag_pos_gpu = torch.tensor(diag_pos, dtype=torch.int32, device='cuda')
            dep_counts_gpu = torch.tensor(dep_counts.copy(), dtype=torch.int32, device='cuda')
            
            ilu0_csr_triton_sync_free(data_gpu, indices_gpu, indptr_gpu, n, dep_counts_gpu, succ_offsets_gpu, succ_indices_gpu, diag_pos_gpu)
            
            result_data = data_gpu.cpu().numpy()
            A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
            
            if validate_ilu0_factorization(A_result, A_test):
                print("triton-csr-sync-free: correct")
                method_correctness[mat_name]['triton-csr-sync-free'] = 1
            else:
                print("Warning: triton-csr-sync-free incorrect")
        except Exception as e:
            print(f"triton-csr-sync-free failed: {e}")

        # Test Numba CSR wavefront
        try:
            A_copy = A_test.copy()
            wavefronts = compute_ilu_wavefront_csr(A_copy)
            wave_start_loc, waves = compact_2d_list(wavefronts)
            waves_gpu = cuda.to_device(waves)
            diag_pos = find_diagonal_positions(A_copy)
            d_diag_pos = cuda.to_device(diag_pos)
            
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            
            ilu0_csr_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu, d_diag_pos)
            
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
            diag_pos = find_diagonal_positions(A_copy)
            
            d_succ_offsets = cuda.to_device(succ_offsets)
            d_succ_indices = cuda.to_device(succ_indices)
            d_diag_pos = cuda.to_device(diag_pos)
            
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            d_dep_counts = cuda.to_device(dep_counts.copy())
            
            ilu0_csr_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices, d_diag_pos)
            
            result_data = d_data.copy_to_host()
            A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
            
            if validate_ilu0_factorization(A_result, A_test):
                print("numba-csr-sync-free: correct")
                method_correctness[mat_name]['numba-csr-sync-free'] = 1
            else:
                print("Warning: numba-csr-sync-free incorrect")
        except Exception as e:
            print(f"numba-csr-sync-free failed: {e}")

        # Test wavefront optimization variants
        for opt_variant in ['mapping', 'unrolling', 'mapping-unrolling']:
            method_name = f'numba-csr-wavefront-{opt_variant}'
            try:
                A_copy = A_test.copy()
                wavefronts = compute_ilu_wavefront_csr(A_copy)
                diag_pos = find_diagonal_positions(A_copy)
                d_diag_pos = cuda.to_device(diag_pos)
                
                d_data = cuda.to_device(A_copy.data)
                d_indices = cuda.to_device(A_copy.indices)
                d_indptr = cuda.to_device(A_copy.indptr)
                
                # Select the appropriate function
                if opt_variant == 'mapping':
                    func = ilu0_csr_numba_wavefront_mapping
                elif opt_variant == 'unrolling':
                    func = ilu0_csr_numba_wavefront_unrolling
                else:  # 'mapping-unrolling'
                    func = ilu0_csr_numba_wavefront_mapping_unrolling
                
                # Process each wavefront
                for wave_rows in wavefronts:
                    d_wave_rows = cuda.to_device(wave_rows)
                    func(d_data, d_indices, d_indptr, d_wave_rows, d_diag_pos, threads_per_block=128)
                
                result_data = d_data.copy_to_host()
                A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
                
                if validate_ilu0_factorization(A_result, A_test):
                    print(f"{method_name}: correct")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"Warning: {method_name} incorrect")
            except Exception as e:
                print(f"{method_name} failed: {e}")

        # Test sync-free optimization variants
        for opt_variant in ['mapping', 'unrolling', 'mapping-unrolling']:
            method_name = f'numba-csr-sync-free-{opt_variant}'
            try:
                A_copy = A_test.copy()
                dep_counts = compute_ilu_dependency_counts_csr(A_copy)
                succ_offsets, succ_indices = build_successors_from_csr(A_copy)
                diag_pos = find_diagonal_positions(A_copy)
                
                d_succ_offsets = cuda.to_device(succ_offsets)
                d_succ_indices = cuda.to_device(succ_indices)
                d_diag_pos = cuda.to_device(diag_pos)
                
                d_data = cuda.to_device(A_copy.data)
                d_indices = cuda.to_device(A_copy.indices)
                d_indptr = cuda.to_device(A_copy.indptr)
                d_dep_counts = cuda.to_device(dep_counts.copy())
                
                # Select the appropriate function
                if opt_variant == 'mapping':
                    func = ilu0_csr_numba_sync_free_mapping
                elif opt_variant == 'unrolling':
                    func = ilu0_csr_numba_sync_free_unrolling
                else:  # 'mapping-unrolling'
                    func = ilu0_csr_numba_sync_free_mapping_unrolling
                
                func(d_data, d_indices, d_indptr, d_dep_counts, d_succ_offsets, d_succ_indices, d_diag_pos, threads_per_block=128)
                
                result_data = d_data.copy_to_host()
                A_result = csr_matrix((result_data, A_copy.indices, A_copy.indptr), shape=A.shape)
                
                if validate_ilu0_factorization(A_result, A_test):
                    print(f"{method_name}: correct")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"Warning: {method_name} incorrect")
            except Exception as e:
                print(f"{method_name} failed: {e}")
    
    return matrix_signatures, method_correctness

mat_signatures, method_correctness = correctness_test()

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
            "Triton Wavefront", "Triton Sync-free",
            "Numba Wavefront", "Numba Sync-free",
            "Numba Wavefront+Mapping", "Numba Wavefront+Unrolling", "Numba Wavefront+Mapping+Unrolling",
            "Numba Sync-free+Mapping", "Numba Sync-free+Unrolling", "Numba Sync-free+Mapping+Unrolling",
        ],
        styles=[
            ("orange", "-"), ("orange", "--"),
            ("red", "-"), ("purple", "-"),
            ("green", "-"), ("green", "--"), ("green", "-."),
            ("blue", "-"), ("blue", "--"), ("blue", "-."),
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
    
    if n > 50000:
        print(f"Skipping matrix with dimension {n} > 50000")
        # Return NaN in the same format as successful benchmarks (GFLOP/s)
        return float('nan'), float('nan'), float('nan')
    
    quantiles = [0.5, 0.2, 0.8]
    # rep, warmup = 100, 25
    rep, warmup = 10, 5
    threads_per_block = 1

    if provider == 'triton-csr-wavefront':
        import torch
        wavefronts = compute_ilu_wavefront_csr(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device='cuda')
        diag_pos = find_diagonal_positions(A)
        diag_pos_gpu = torch.tensor(diag_pos, dtype=torch.int32, device='cuda')
        
        # Allocate GPU memory
        data_gpu = torch.tensor(A.data.copy(), dtype=torch.float64, device='cuda')
        indices_gpu = torch.tensor(A.indices, dtype=torch.int32, device='cuda')
        indptr_gpu = torch.tensor(A.indptr, dtype=torch.int32, device='cuda')
        original_data = data_gpu.clone()
        
        def run_triton_csr_wave():
            data_gpu.copy_(original_data)
            ilu0_csr_triton_wavefront(data_gpu, indices_gpu, indptr_gpu, n, wave_start_loc, waves_gpu, diag_pos_gpu)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_triton_csr_wave, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'triton-csr-sync-free':
        import torch
        dep_counts = compute_ilu_dependency_counts_csr(A)
        succ_offsets, succ_indices = build_successors_from_csr(A)
        diag_pos = find_diagonal_positions(A)
        
        # Allocate GPU memory
        data_gpu = torch.tensor(A.data.copy(), dtype=torch.float64, device='cuda')
        indices_gpu = torch.tensor(A.indices, dtype=torch.int32, device='cuda')
        indptr_gpu = torch.tensor(A.indptr, dtype=torch.int32, device='cuda')
        succ_offsets_gpu = torch.tensor(succ_offsets, dtype=torch.int32, device='cuda')
        succ_indices_gpu = torch.tensor(succ_indices, dtype=torch.int32, device='cuda')
        diag_pos_gpu = torch.tensor(diag_pos, dtype=torch.int32, device='cuda')
        dep_counts_gpu = torch.tensor(dep_counts, dtype=torch.int32, device='cuda')
        original_data = data_gpu.clone()
        
        def run_triton_csr_sync_free():
            data_gpu.copy_(original_data)
            ilu0_csr_triton_sync_free(data_gpu, indices_gpu, indptr_gpu, n, dep_counts_gpu, succ_offsets_gpu, succ_indices_gpu, diag_pos_gpu)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_triton_csr_sync_free, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'numba-csr-wavefront':
        wavefronts = compute_ilu_wavefront_csr(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        diag_pos = find_diagonal_positions(A)
        d_diag_pos = cuda.to_device(diag_pos)
        
        # Allocate GPU memory once
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_data = cuda.device_array_like(A.data)
        d_original_data = cuda.to_device(A.data.copy())
        
        def run_numba_csr_wave():
            # Reset data on GPU before each run (device-to-device copy, not timed as overhead)
            d_data.copy_to_device(d_original_data)
            ilu0_csr_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu, d_diag_pos, threads_per_block)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_wave, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'numba-csr-sync-free':
        dep_counts = compute_ilu_dependency_counts_csr(A)
        succ_offsets, succ_indices = build_successors_from_csr(A)
        diag_pos = find_diagonal_positions(A)
        
        # Allocate GPU memory once
        d_succ_offsets = cuda.to_device(succ_offsets)
        d_succ_indices = cuda.to_device(succ_indices)
        d_diag_pos = cuda.to_device(diag_pos)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_data = cuda.device_array_like(A.data)
        d_dep_counts = cuda.device_array_like(dep_counts)
        d_original_data = cuda.to_device(A.data.copy())
        d_original_dep_counts = cuda.to_device(dep_counts.copy())
        
        def run_numba_csr_sync():
            # Reset data and dependency counts on GPU before each run (device-to-device copy)
            d_data.copy_to_device(d_original_data)
            d_dep_counts.copy_to_device(d_original_dep_counts)
            ilu0_csr_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices, d_diag_pos, threads_per_block)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_sync, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider in ['numba-csr-wavefront-mapping', 'numba-csr-wavefront-unrolling', 'numba-csr-wavefront-mapping-unrolling']:
        # Determine which variant
        if 'unrolling' in provider and 'mapping' in provider:
            opt_variant = 'mapping-unrolling'
        elif 'mapping' in provider:
            opt_variant = 'mapping'
        else:
            opt_variant = 'unrolling'
        
        threads_per_block = 128
        
        wavefronts = compute_ilu_wavefront_csr(A)
        diag_pos = find_diagonal_positions(A)
        d_diag_pos = cuda.to_device(diag_pos)
        
        # Allocate GPU memory once
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_data = cuda.device_array_like(A.data)
        d_original_data = cuda.to_device(A.data.copy())
        
        # Pre-upload wavefront arrays
        d_wavefronts = [cuda.to_device(wave_rows) for wave_rows in wavefronts]
        
        # Select function
        if opt_variant == 'mapping':
            func = ilu0_csr_numba_wavefront_mapping
        elif opt_variant == 'unrolling':
            func = ilu0_csr_numba_wavefront_unrolling
        else:  # 'mapping-unrolling'
            func = ilu0_csr_numba_wavefront_mapping_unrolling
        
        def run_numba_csr_wave_opt():
            # Reset data on GPU before each run (device-to-device copy)
            d_data.copy_to_device(d_original_data)
            for d_wave_rows in d_wavefronts:
                func(d_data, d_indices, d_indptr, d_wave_rows, d_diag_pos, threads_per_block)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_wave_opt, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider in ['numba-csr-sync-free-mapping', 'numba-csr-sync-free-unrolling', 'numba-csr-sync-free-mapping-unrolling']:
        # Determine which variant
        if 'unrolling' in provider and 'mapping' in provider:
            opt_variant = 'mapping-unrolling'
        elif 'mapping' in provider:
            opt_variant = 'mapping'
        else:
            opt_variant = 'unrolling'
        
        threads_per_block = 128
        
        dep_counts = compute_ilu_dependency_counts_csr(A)
        succ_offsets, succ_indices = build_successors_from_csr(A)
        diag_pos = find_diagonal_positions(A)
        
        # Allocate GPU memory once
        d_succ_offsets = cuda.to_device(succ_offsets)
        d_succ_indices = cuda.to_device(succ_indices)
        d_diag_pos = cuda.to_device(diag_pos)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_data = cuda.device_array_like(A.data)
        d_dep_counts = cuda.device_array_like(dep_counts)
        d_original_data = cuda.to_device(A.data.copy())
        d_original_dep_counts = cuda.to_device(dep_counts.copy())
        
        # Select function
        if opt_variant == 'mapping':
            func = ilu0_csr_numba_sync_free_mapping
        elif opt_variant == 'unrolling':
            func = ilu0_csr_numba_sync_free_unrolling
        else:  # 'mapping-unrolling'
            func = ilu0_csr_numba_sync_free_mapping_unrolling
        
        def run_numba_csr_sync_opt():
            # Reset data and dependency counts on GPU before each run (device-to-device copy)
            d_data.copy_to_device(d_original_data)
            d_dep_counts.copy_to_device(d_original_dep_counts)
            func(d_data, d_indices, d_indptr, d_dep_counts, d_succ_offsets, d_succ_indices, d_diag_pos, threads_per_block)
        
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_sync_opt, quantiles=quantiles, warmup=warmup, rep=rep)

    else:
        print(f"Unknown provider: {provider}, skipping...")
        return float('nan'), float('nan'), float('nan')

    perf = lambda t_ms: (2 * A.nnz) / (t_ms * 1e-3) / 1e9 if not np.isnan(t_ms) else float('nan')
    print(f"{provider} -- {matrices} -- time: {ms:.3f} ms")
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".", show_plots=False)
