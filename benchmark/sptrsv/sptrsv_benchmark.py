import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bench_utils import *
from spmv import *

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

from sptrsv_kernels.sptrsv_baseline import *
from sptrsv_kernels.sptrsv_unrolled import (
    sptrsv_numba_csc_wavefront_unrolling_4, sptrsv_numba_csc_wavefront_unrolling_8,
    sptrsv_numba_csc_wavefront_unrolling_16, sptrsv_numba_csc_wavefront_unrolling_32,
    sptrsv_numba_csc_wavefront_unrolling_64,
    sptrsv_numba_csc_sync_free_unrolling_4, sptrsv_numba_csc_sync_free_unrolling_8,
    sptrsv_numba_csc_sync_free_unrolling_16, sptrsv_numba_csc_sync_free_unrolling_32,
    sptrsv_numba_csc_sync_free_unrolling_64,
    sptrsv_numba_csr_wavefront_unrolling_4, sptrsv_numba_csr_wavefront_unrolling_8,
    sptrsv_numba_csr_wavefront_unrolling_16, sptrsv_numba_csr_wavefront_unrolling_32,
    sptrsv_numba_csr_wavefront_unrolling_64,
    sptrsv_numba_csr_sync_free_unrolling_4, sptrsv_numba_csr_sync_free_unrolling_8,
    sptrsv_numba_csr_sync_free_unrolling_16, sptrsv_numba_csr_sync_free_unrolling_32,
    sptrsv_numba_csr_sync_free_unrolling_64,
    sptrsv_numba_csc_wavefront_mapping, sptrsv_numba_csc_sync_free_mapping,
    sptrsv_numba_csr_wavefront_mapping, sptrsv_numba_csr_sync_free_mapping,
)
from sptrsv_kernels.sptrsv_coarsened import (
    compute_coarsened_wavefronts, compact_2d_list, compact_coarsened_wavefronts,
    sptrsv_numba_csc_coarsened_mapping, sptrsv_numba_csr_coarsened_mapping,
    sptrsv_numba_csc_wavefront_block_coarsening, sptrsv_numba_csr_wavefront_block_coarsening,
    sptrsv_numba_csc_wavefront_thread_coarsening, sptrsv_numba_csr_wavefront_thread_coarsening,
    sptrsv_numba_csc_sync_free_block_coarsening, sptrsv_numba_csr_sync_free_block_coarsening,
    sptrsv_numba_csc_sync_free_thread_coarsening, sptrsv_numba_csr_sync_free_thread_coarsening,
)
from sptrsv_kernels.sptrsv_hlevel import (
    sptrsv_h_level_analyze,
    sptrsv_numba_csc_h_level_solve, sptrsv_numba_csr_h_level_solve,
    sptrsv_numba_csc_h_level_graph, sptrsv_numba_csr_h_level_graph,
)

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")



mtx_list = [
"/home/mad29/projects/testdata/matrices/1138_bus.mtx",
"/home/mad29/projects/testdata/matrices/2cubes_sphere.mtx",
# "/home/mad29/projects/testdata/matrices/Chem97ZtZ.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova1.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova2.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova3.mtx",
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
# ============================================================================
# Benchmark Configuration
# ============================================================================
# Parameter ranges for kernel variants
unroll_factors = [4, 8, 16, 32, 64]
thread_counts = [32, 64, 128, 256]
block_coarsening_factors = [2, 4, 8, 16]
sync_free_block_factors = [1, 2, 4]
thread_coarsening_factors = [1, 2, 4]
wavefront_coarsening_factors = [1, 2, 4, 8]
band_widths = [2, 4, 8, 16]

# ============================================================================
# Method List Construction
# Comment out any section to disable that category of tests
# ============================================================================
method_list = []

# ----------------------------------------------------------------------------
# BASELINES
# ----------------------------------------------------------------------------
method_list.extend([
    "cusparse",
    "csc-wf-base",
    "csc-sf-base",
    "csr-wf-base",
    "csr-sf-base",
])

# ----------------------------------------------------------------------------
# TEST CANDIDATES - WAVEFRONT - UNROLLED
# ----------------------------------------------------------------------------
# Unroll factor 1 (mapping variants)
for threads in thread_counts:
    method_list.extend([
        f"csc-wf-u1-t{threads}",
        f"csr-wf-u1-t{threads}",
    ])

# Explicit unrolling with factors 4-64
for factor in unroll_factors:
    for threads in thread_counts:
        method_list.extend([
            f"csc-wf-u{factor}-t{threads}",
            f"csr-wf-u{factor}-t{threads}",
        ])

# ----------------------------------------------------------------------------
# TEST CANDIDATES - WAVEFRONT - COARSENED
# ----------------------------------------------------------------------------
# Block coarsening (merge dependency levels)
for cf in block_coarsening_factors:
    method_list.extend([
        f"csc-wf-coarse-block-wc{cf}",
        f"csr-wf-coarse-block-wc{cf}",
    ])

# Thread coarsening
for threads in thread_counts:
    method_list.extend([
        f"csc-wf-coarse-thread-tc{threads}",
        f"csr-wf-coarse-thread-tc{threads}",
    ])

# Combined 3-level coarsening (wavefront + warp + thread)
for cf in wavefront_coarsening_factors:
    for threads in thread_counts:
        warps = threads // 32
        for tcf in thread_coarsening_factors:
            method_list.extend([
                f"csc-wf-coarse-comb-wc{cf}-w{warps}-tc{tcf}",
                f"csr-wf-coarse-comb-wc{cf}-w{warps}-tc{tcf}",
            ])

# ----------------------------------------------------------------------------
# TEST CANDIDATES - SYNC-FREE - UNROLLED
# ----------------------------------------------------------------------------
# Unroll factor 1 (mapping variants)
for threads in thread_counts:
    method_list.extend([
        f"csc-sf-u1-t{threads}",
        f"csr-sf-u1-t{threads}",
    ])

# Explicit unrolling with factors 4-64
for factor in unroll_factors:
    for threads in thread_counts:
        method_list.extend([
            f"csc-sf-u{factor}-t{threads}",
            f"csr-sf-u{factor}-t{threads}",
        ])

# ----------------------------------------------------------------------------
# TEST CANDIDATES - SYNC-FREE - COARSENED
# ----------------------------------------------------------------------------
# Block coarsening
for bcf in sync_free_block_factors:
    method_list.extend([
        f"csc-sf-coarse-block-bc{bcf}",
        f"csr-sf-coarse-block-bc{bcf}",
    ])

# Thread coarsening
for threads in thread_counts:
    method_list.extend([
        f"csc-sf-coarse-thread-tc{threads}",
        f"csr-sf-coarse-thread-tc{threads}",
    ])

# ----------------------------------------------------------------------------
# TEST CANDIDATES - H-LEVEL
# ----------------------------------------------------------------------------
for bw in band_widths:
    method_list.extend([
        f"csc-h-bw{bw}",
        f"csr-h-bw{bw}",
        ])

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
        y_numba_gpu = sptrsv_numba_csc(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu)
        y_numba_cpu = y_numba_gpu.copy_to_host()

        #d_in_edges = cuda.to_device(in_edge)
        y_numba_gpu_sync_free = sptrsv_numba_csc_sync_free(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge)

        # dep_count_csc1 = (A.indptr[1:] - A.indptr[:-1]).astype(np.int32)
        # dep_count_csc1[dep_count_csc1 == 0] = 1
        # y_numba_gpu_sync_free = sptrsv_numba_csc_sync_free(
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

        # Test triton-csr-mapping
        try:
            wave_start_loc_csr_int = [int(x) for x in wave_start_loc_csr]
            y_csr_mapping = sptrsv_triton_csr_mapping(val_f32_csr, indices_i32_csr, indptr_i32_csr, b_f32, n, n, A_csr.nnz, wave_start_loc_csr_int, waves_gpu_csr)
            if np.allclose(x_cpu, y_csr_mapping.cpu().numpy(), atol=1e-5):
                print("\ntriton-csr-mapping Solutions match within tolerance!")
                method_correctness[mat_name]['triton-csr-mapping'] = 1
            else:
                print("\nWarning: triton-csr-mapping Solutions do not match!")
        except Exception as e:
            print(f"\ntriton-csr-mapping failed: {e}")

        # Test triton-csr-sync-free-mapping
        try:
            map_ptr_csr, map_vals_csr, outgoing_edges_csr = incoming_edge_csr(A_csr)
            dep_count_csr_map = torch.tensor(outgoing_edges_csr, dtype=torch.int32, device=DEVICE)
            map_ptr_gpu_csr = torch.tensor(map_ptr_csr, dtype=torch.int32, device=DEVICE)
            map_vals_gpu_csr = torch.tensor(map_vals_csr, dtype=torch.int32, device=DEVICE)
            y_csr_sync_free_map = sptrsv_triton_csr_sync_free_mapping(val_f32_csr, indices_i32_csr, indptr_i32_csr, b_f32, n, n, A_csr.nnz, dep_count_csr_map, map_ptr_gpu_csr, map_vals_gpu_csr)
            if np.allclose(x_cpu, y_csr_sync_free_map.cpu().numpy(), atol=1e-5):
                print("\ntriton-csr-sync-free-mapping Solutions match within tolerance!")
                method_correctness[mat_name]['triton-csr-sync-free-mapping'] = 1
            else:
                print("\nWarning: triton-csr-sync-free-mapping Solutions do not match!")
        except Exception as e:
            print(f"\ntriton-csr-sync-free-mapping failed: {e}")

        # Test triton-csc-mapping
        try:
            wave_start_loc_int = [int(x) for x in wave_start_loc]
            y_csc_mapping = sptrsv_triton_csc_mapping(val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz, wave_start_loc_int, waves_gpu)
            if np.allclose(x_cpu, y_csc_mapping.cpu().numpy(), atol=1e-5):
                print("\ntriton-csc-mapping Solutions match within tolerance!")
                method_correctness[mat_name]['triton-csc-mapping'] = 1
            else:
                print("\nWarning: triton-csc-mapping Solutions do not match!")
        except Exception as e:
            print(f"\ntriton-csc-mapping failed: {e}")

        if np.allclose(x_cpu, y_numba_cpu, atol=1e-5):
            print("\ncsc-wf-base Solutions match within tolerance!")
            method_correctness[mat_name]['csc-wf-base'] = 1
        else:
            print("\nWarning: csc-wf-base Solutions do not match!")

        if np.allclose(x_cpu, y_numba_cpu_sync_free, atol=1e-5):
            print("\ncsc-sf-base Solutions match within tolerance!")
            method_correctness[mat_name]['csc-sf-base'] = 1
        else:
            print("\nWarning: csc-sf-base Solutions do not match!")

            # abs_err = np.max(np.abs(x_cpu - y_numba_cpu_sync_free))
            # rel_err = np.linalg.norm(x_cpu - y_numba_cpu_sync_free) / np.linalg.norm(x_cpu)
            # print(f"  Max abs error: {abs_err:.3e}")
            # print(f"  Relative L2 error: {rel_err:.3e}")

        # Test CSC wavefront mapping variant
        succ_offsets_csc, succ_indices_csc = build_csc_successors(A)
        d_succ_offsets_csc = cuda.to_device(succ_offsets_csc)
        d_succ_indices_csc = cuda.to_device(succ_indices_csc)
        
        try:
            y_opt = sptrsv_numba_csc_wavefront_mapping(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block=128)
            y_opt_cpu = y_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                print("\ncsc-wf-u1-t128 Solutions match within tolerance!")
                method_correctness[mat_name]['csc-wf-u1-t128'] = 1
            else:
                print("\nWarning: csc-wf-u1-t128 Solutions do not match!")
        except Exception as e:
            print(f"\ncsc-wf-u1-t128 failed: {e}")

        # Test CSC sync-free mapping variant
        try:
            y_opt = sptrsv_numba_csc_sync_free_mapping(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block=128)
            y_opt_cpu = y_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                print("\ncsc-sf-u1-t128 Solutions match within tolerance!")
                method_correctness[mat_name]['csc-sf-u1-t128'] = 1
            else:
                print("\nWarning: csc-sf-u1-t128 Solutions do not match!")
        except Exception as e:
            print(f"\ncsc-sf-u1-t128 failed: {e}")

        # Test CSC wavefront unrolling variants with different factors
        func_map_csc_wf = {
            4: sptrsv_numba_csc_wavefront_unrolling_4,
            8: sptrsv_numba_csc_wavefront_unrolling_8,
            16: sptrsv_numba_csc_wavefront_unrolling_16,
            32: sptrsv_numba_csc_wavefront_unrolling_32,
            64: sptrsv_numba_csc_wavefront_unrolling_64,
        }
        for factor in unroll_factors:
            method_name = f'csc-wf-u{factor}-t128'
            try:
                unroll_func = func_map_csc_wf[factor]
                y_opt = unroll_func(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block=128)
                y_opt_cpu = y_opt.copy_to_host()
                
                if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

        # Test CSC sync-free unrolling variants with different factors
        func_map_csc_sf = {
            4: sptrsv_numba_csc_sync_free_unrolling_4,
            8: sptrsv_numba_csc_sync_free_unrolling_8,
            16: sptrsv_numba_csc_sync_free_unrolling_16,
            32: sptrsv_numba_csc_sync_free_unrolling_32,
            64: sptrsv_numba_csc_sync_free_unrolling_64,
        }
        for factor in unroll_factors:
            method_name = f'csc-sf-u{factor}-t128'
            try:
                unroll_func = func_map_csc_sf[factor]
                y_opt = unroll_func(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block=128)
                y_opt_cpu = y_opt.copy_to_host()
                
                if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

        # Test numba-csr-wavefront
        A_csr_numba = A.tocsr()
        d_data_csr = cuda.to_device(A_csr_numba.data)
        d_indices_csr = cuda.to_device(A_csr_numba.indices)
        d_indptr_csr = cuda.to_device(A_csr_numba.indptr)
        d_b_csr = cuda.to_device(b)
        wavefronts_csr_numba = compute_wavefront(A)
        wave_start_loc_csr_numba, waves_csr_numba = compact_2d_list(wavefronts_csr_numba)
        waves_gpu_csr_numba = cuda.to_device(waves_csr_numba)
        y_numba_csr_wave = sptrsv_numba_csr_wavefront(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, wave_start_loc_csr_numba, waves_gpu_csr_numba)
        y_numba_csr_wave_cpu = y_numba_csr_wave.copy_to_host()
        
        if np.allclose(x_cpu, y_numba_csr_wave_cpu, atol=1e-5):
            print("\ncsr-wf-base Solutions match within tolerance!")
            method_correctness[mat_name]['csr-wf-base'] = 1
        else:
            print("\nWarning: csr-wf-base Solutions do not match!")

        # Test numba-csr-sync-free
        in_edges_csr = incoming_edge_csr_rows(A_csr_numba)
        succ_offsets_csr, succ_indices_csr = build_csr_successors(A_csr_numba)
        y_numba_csr_sync_free = sptrsv_numba_csr_sync_free(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, in_edges_csr, succ_offsets_csr, succ_indices_csr)
        y_numba_csr_sync_free_cpu = y_numba_csr_sync_free.copy_to_host()
        
        if np.allclose(x_cpu, y_numba_csr_sync_free_cpu, atol=1e-5):
            print("\ncsr-sf-base Solutions match within tolerance!")
            method_correctness[mat_name]['csr-sf-base'] = 1
        else:
            print("\nWarning: csr-sf-base Solutions do not match!")

        # Test CSR wavefront mapping variant
        method_name = 'csr-wf-u1-t128'
        try:
            y_csr_opt = sptrsv_numba_csr_wavefront_mapping(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, wave_start_loc_csr_numba, waves_gpu_csr_numba, threads_per_block=128)
            y_csr_opt_cpu = y_csr_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_csr_opt_cpu, atol=1e-5):
                print(f"\n{method_name} Solutions match within tolerance!")
                method_correctness[mat_name][method_name] = 1
            else:
                print(f"\nWarning: {method_name} Solutions do not match!")
        except Exception as e:
            print(f"\n{method_name} failed: {e}")

        # Test CSR wavefront unrolling variants with different factors
        func_map_csr_wf = {
            4: sptrsv_numba_csr_wavefront_unrolling_4,
            8: sptrsv_numba_csr_wavefront_unrolling_8,
            16: sptrsv_numba_csr_wavefront_unrolling_16,
            32: sptrsv_numba_csr_wavefront_unrolling_32,
            64: sptrsv_numba_csr_wavefront_unrolling_64,
        }
        for factor in unroll_factors:
            method_name = f'csr-wf-u{factor}-t128'
            try:
                unroll_func = func_map_csr_wf[factor]
                y_csr_opt = unroll_func(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, wave_start_loc_csr_numba, waves_gpu_csr_numba, threads_per_block=128)
                y_csr_opt_cpu = y_csr_opt.copy_to_host()
                
                if np.allclose(x_cpu, y_csr_opt_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

        # Test CSR sync-free mapping variant
        method_name = 'csr-sf-u1-t128'
        try:
            y_csr_opt = sptrsv_numba_csr_sync_free_mapping(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, in_edges_csr, succ_offsets_csr, succ_indices_csr, threads_per_block=128)
            y_csr_opt_cpu = y_csr_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_csr_opt_cpu, atol=1e-5):
                print(f"\n{method_name} Solutions match within tolerance!")
                method_correctness[mat_name][method_name] = 1
            else:
                print(f"\nWarning: {method_name} Solutions do not match!")
        except Exception as e:
            print(f"\n{method_name} failed: {e}")

        # Test CSR sync-free unrolling variants with different factors
        func_map_csr_sf = {
            4: sptrsv_numba_csr_sync_free_unrolling_4,
            8: sptrsv_numba_csr_sync_free_unrolling_8,
            16: sptrsv_numba_csr_sync_free_unrolling_16,
            32: sptrsv_numba_csr_sync_free_unrolling_32,
            64: sptrsv_numba_csr_sync_free_unrolling_64,
        }
        for factor in unroll_factors:
            method_name = f'csr-sf-u{factor}-t128'
            try:
                unroll_func = func_map_csr_sf[factor]
                y_csr_opt = unroll_func(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, in_edges_csr, succ_offsets_csr, succ_indices_csr, threads_per_block=128)
                y_csr_opt_cpu = y_csr_opt.copy_to_host()
                
                if np.allclose(x_cpu, y_csr_opt_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

        # Test CSC wavefront-based coarsening with block/thread coarsening
        print("\nTesting CSC wavefront-based coarsening methods...")
        original_wavefronts = compute_wavefront(A)
        
        for cf in block_coarsening_factors:
            # Test wavefront block coarsening
            method_name = f'csc-wf-coarse-block-wc{cf}'
            try:
                wave_start, waves, level_boundaries = compact_coarsened_wavefronts(original_wavefronts, cf)
                waves_gpu = cuda.to_device(waves)
                y_coarse = sptrsv_numba_csc_wavefront_block_coarsening(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, 
                    wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1)
                y_coarse_cpu = y_coarse.copy_to_host()
                
                if np.allclose(x_cpu, y_coarse_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSC wavefront thread coarsening (only for cf=1)
        for threads in thread_counts:
            method_name = f'csc-wf-coarse-thread-tc{threads}'
            try:
                wave_start, waves, level_boundaries = compact_coarsened_wavefronts(original_wavefronts, 1)
                waves_gpu = cuda.to_device(waves)
                y_coarse = sptrsv_numba_csc_wavefront_thread_coarsening(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, 
                    wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads)
                y_coarse_cpu = y_coarse.copy_to_host()
                
                if np.allclose(x_cpu, y_coarse_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSC sync-free block coarsening
        print("\nTesting CSC sync-free coarsening methods...")
        for bcf in sync_free_block_factors:
            method_name = f'csc-sf-coarse-block-bc{bcf}'
            try:
                y_sf = sptrsv_numba_csc_sync_free_block_coarsening(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, 
                    in_edge, block_coarsening_factor=bcf)
                y_sf_cpu = y_sf.copy_to_host()
                
                if np.allclose(x_cpu, y_sf_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSC sync-free thread coarsening
        for threads in thread_counts:
            method_name = f'csc-sf-coarse-thread-tc{threads}'
            try:
                y_sf = sptrsv_numba_csc_sync_free_thread_coarsening(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, 
                    in_edge, thread_coarsening_factor=1, threads_per_block=threads)
                y_sf_cpu = y_sf.copy_to_host()
                
                if np.allclose(x_cpu, y_sf_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSR wavefront-based coarsening with block/thread coarsening
        print("\nTesting CSR wavefront-based coarsening methods...")
        
        for cf in block_coarsening_factors:
            # Test wavefront block coarsening
            method_name = f'csr-wf-coarse-block-wc{cf}'
            try:
                wave_start, waves, level_boundaries = compact_coarsened_wavefronts(original_wavefronts, cf)
                waves_gpu = cuda.to_device(waves)
                y_coarse = sptrsv_numba_csr_wavefront_block_coarsening(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1)
                y_coarse_cpu = y_coarse.copy_to_host()
                
                if np.allclose(x_cpu, y_coarse_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSR wavefront thread coarsening (only for cf=1)
        for threads in thread_counts:
            method_name = f'csr-wf-coarse-thread-tc{threads}'
            try:
                wave_start, waves, level_boundaries = compact_coarsened_wavefronts(original_wavefronts, 1)
                waves_gpu = cuda.to_device(waves)
                y_coarse = sptrsv_numba_csr_wavefront_thread_coarsening(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads)
                y_coarse_cpu = y_coarse.copy_to_host()
                
                if np.allclose(x_cpu, y_coarse_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSR sync-free block coarsening
        print("\nTesting CSR sync-free coarsening methods...")
        for bcf in sync_free_block_factors:
            method_name = f'csr-sf-coarse-block-bc{bcf}'
            try:
                y_sf = sptrsv_numba_csr_sync_free_block_coarsening(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    in_edges_csr, succ_offsets_csr, succ_indices_csr, block_coarsening_factor=bcf)
                y_sf_cpu = y_sf.copy_to_host()
                
                if np.allclose(x_cpu, y_sf_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test CSR sync-free thread coarsening
        for threads in thread_counts:
            method_name = f'csr-sf-coarse-thread-tc{threads}'
            try:
                y_sf = sptrsv_numba_csr_sync_free_thread_coarsening(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    in_edges_csr, succ_offsets_csr, succ_indices_csr, 
                    thread_coarsening_factor=1, threads_per_block=threads)
                y_sf_cpu = y_sf.copy_to_host()
                
                if np.allclose(x_cpu, y_sf_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test H-Level methods
        print("\nTesting H-Level methods...")
        for bw in band_widths:
            # Test CSC H-Level (now using CUDA graphs)
            method_name = f'csc-h-bw{bw}'
            try:
                y_h_level = sptrsv_numba_csc_h_level_graph(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz,
                    waves, wave_start_loc, band_width=bw)
                y_h_level_cpu = y_h_level.copy_to_host()
                
                if np.allclose(x_cpu, y_h_level_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
            
            # Test CSR H-Level (now using CUDA graphs)
            method_name = f'csr-h-bw{bw}'
            try:
                y_h_level_csr = sptrsv_numba_csr_h_level_graph(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    waves_csr_numba, wave_start_loc_csr_numba, band_width=bw)
                y_h_level_csr_cpu = y_h_level_csr.copy_to_host()
                
                if np.allclose(x_cpu, y_h_level_csr_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")
        
        # Test 3-level coarsening variants
        print("\nTesting 3-level coarsening (wavefront + warp + thread)...")
        for cf in [1, 2, 4, 8]:
            for threads in [128]:  # Test with one thread count for correctness
                warps = threads // 32
                for tcf in [1, 2, 4]:
                    method_name = f'csc-wf-coarse-comb-wc{cf}-w{warps}-tc{tcf}'
                    try:
                        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(original_wavefronts, cf)
                        waves_gpu = cuda.to_device(waves)
                        y_coarse = sptrsv_numba_csc_coarsened_mapping(
                            d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, 
                            wave_start, waves_gpu, cf, threads, thread_coarsening_factor=tcf, 
                            level_boundaries=level_boundaries)
                        y_coarse_cpu = y_coarse.copy_to_host()
                        
                        if np.allclose(x_cpu, y_coarse_cpu, atol=1e-5):
                            print(f"\n{method_name} Solutions match within tolerance!")
                            method_correctness[mat_name][method_name] = 1
                        else:
                            print(f"\nWarning: {method_name} Solutions do not match!")
                    except Exception as e:
                        print(f"\n{method_name} failed: {e}")
        
        for cf in [1, 2, 4, 8]:
            for threads in [128]:  # Test with one thread count for correctness
                warps = threads // 32
                for tcf in [1, 2, 4]:
                    method_name = f'csr-wf-coarse-comb-wc{cf}-w{warps}-tc{tcf}'
                    try:
                        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(original_wavefronts, cf)
                        waves_gpu = cuda.to_device(waves)
                        y_coarse = sptrsv_numba_csr_coarsened_mapping(
                            d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                            wave_start, waves_gpu, cf, threads, thread_coarsening_factor=tcf,
                            level_boundaries=level_boundaries)
                        y_coarse_cpu = y_coarse.copy_to_host()
                        
                        if np.allclose(x_cpu, y_coarse_cpu, atol=1e-5):
                            print(f"\n{method_name} Solutions match within tolerance!")
                            method_correctness[mat_name][method_name] = 1
                        else:
                            print(f"\nWarning: {method_name} Solutions do not match!")
                    except Exception as e:
                        print(f"\n{method_name} failed: {e}")

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
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],
        line_arg="provider",
        line_vals=method_list,
        line_names=method_list,
        ylabel="GFLOP/S",
        plot_name="sptrsv-performance",
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
    
    # ========================================================================
    # BASELINES
    # ========================================================================
    if provider == 'cusparse':
        matrix_gpu_csc = cp_csc_matrix(A.tocsc())
        b_gpu = cp.array(b)  # Move dense vector to GPU
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cusparse_solver(matrix_gpu_csc, b_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csc-wf-base':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csc-sf-base':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csr-wf-base':
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csr-sf-base':
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # ========================================================================
    # WAVEFRONT - UNROLLED (mapping u1 + explicit unrolling)
    # ========================================================================
    if provider.startswith('csc-wf-u') and '-t' in provider:
        parts = provider.replace('csc-wf-u', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        
        if unroll_factor == 1:
            # Mapping variant
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
        else:
            # Unrolling variant
            func_map = {
                4: sptrsv_numba_csc_wavefront_unrolling_4,
                8: sptrsv_numba_csc_wavefront_unrolling_8,
                16: sptrsv_numba_csc_wavefront_unrolling_16,
                32: sptrsv_numba_csc_wavefront_unrolling_32,
                64: sptrsv_numba_csc_wavefront_unrolling_64,
            }
            unroll_func = func_map[unroll_factor]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-wf-u') and '-t' in provider:
        parts = provider.replace('csr-wf-u', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        
        if unroll_factor == 1:
            # Mapping variant
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
        else:
            # Unrolling variant
            func_map = {
                4: sptrsv_numba_csr_wavefront_unrolling_4,
                8: sptrsv_numba_csr_wavefront_unrolling_8,
                16: sptrsv_numba_csr_wavefront_unrolling_16,
                32: sptrsv_numba_csr_wavefront_unrolling_32,
                64: sptrsv_numba_csr_wavefront_unrolling_64,
            }
            unroll_func = func_map[unroll_factor]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # ========================================================================
    # WAVEFRONT - COARSENED
    # ========================================================================
    if provider.startswith('csc-wf-coarse-block-wc'):
        coarsening_factor = int(provider.replace('csc-wf-coarse-block-wc', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-wf-coarse-block-wc'):
        coarsening_factor = int(provider.replace('csr-wf-coarse-block-wc', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csc-wf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csc-wf-coarse-thread-tc', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, 1)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-wf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csr-wf-coarse-thread-tc', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, 1)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csc-wf-coarse-comb-wc') and '-w' in provider and '-tc' in provider:
        parts = provider.replace('csc-wf-coarse-comb-wc', '').split('-w')
        coarsening_factor = int(parts[0])
        parts2 = parts[1].split('-tc')
        warps = int(parts2[0])
        thread_coarsening_factor = int(parts2[1])
        threads_per_block = warps * 32
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_coarsened_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, coarsening_factor, threads_per_block, thread_coarsening_factor=thread_coarsening_factor, level_boundaries=level_boundaries), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-wf-coarse-comb-wc') and '-w' in provider and '-tc' in provider:
        parts = provider.replace('csr-wf-coarse-comb-wc', '').split('-w')
        coarsening_factor = int(parts[0])
        parts2 = parts[1].split('-tc')
        warps = int(parts2[0])
        thread_coarsening_factor = int(parts2[1])
        threads_per_block = warps * 32
        
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_coarsened_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, coarsening_factor, threads_per_block, thread_coarsening_factor=thread_coarsening_factor, level_boundaries=level_boundaries), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # ========================================================================
    # SYNC-FREE - UNROLLED
    # ========================================================================
    if provider.startswith('csc-sf-u') and '-t' in provider:
        parts = provider.replace('csc-sf-u', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        succ_offsets_csc, succ_indices_csc = build_csc_successors(A)
        d_succ_offsets_csc = cuda.to_device(succ_offsets_csc)
        d_succ_indices_csc = cuda.to_device(succ_indices_csc)
        
        if unroll_factor == 1:
            # Mapping variant
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
        else:
            # Unrolling variant
            func_map = {
                4: sptrsv_numba_csc_sync_free_unrolling_4,
                8: sptrsv_numba_csc_sync_free_unrolling_8,
                16: sptrsv_numba_csc_sync_free_unrolling_16,
                32: sptrsv_numba_csc_sync_free_unrolling_32,
                64: sptrsv_numba_csc_sync_free_unrolling_64,
            }
            unroll_func = func_map[unroll_factor]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-sf-u') and '-t' in provider:
        parts = provider.replace('csr-sf-u', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        
        if unroll_factor == 1:
            # Mapping variant
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
        else:
            # Unrolling variant
            func_map = {
                4: sptrsv_numba_csr_sync_free_unrolling_4,
                8: sptrsv_numba_csr_sync_free_unrolling_8,
                16: sptrsv_numba_csr_sync_free_unrolling_16,
                32: sptrsv_numba_csr_sync_free_unrolling_32,
                64: sptrsv_numba_csr_sync_free_unrolling_64,
            }
            unroll_func = func_map[unroll_factor]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # ========================================================================
    # SYNC-FREE - COARSENED
    # ========================================================================
    if provider.startswith('csc-sf-coarse-block-bc'):
        block_coarsening_factor = int(provider.replace('csc-sf-coarse-block-bc', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, block_coarsening_factor=block_coarsening_factor), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-sf-coarse-block-bc'):
        block_coarsening_factor = int(provider.replace('csr-sf-coarse-block-bc', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, block_coarsening_factor=block_coarsening_factor), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csc-sf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csc-sf-coarse-thread-tc', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-sf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csr-sf-coarse-thread-tc', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # ========================================================================
    # H-LEVEL
    # ========================================================================
    if provider.startswith('csc-h-bw'):
        band_width = int(provider.replace('csc-h-bw', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-h-bw'):
        band_width = int(provider.replace('csr-h-bw', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # Keep old providers for backwards compatibility (if needed for testing)
    if provider == 'triton':
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        wave_start_loc = [int(x) for x in wave_start_loc]
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
    if provider == 'triton-csr-mapping':
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        wave_start_loc = [int(x) for x in wave_start_loc]
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_triton_csr_mapping(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'triton-csr-sync-free-mapping':
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        map_ptr, map_vals, outgoing_edges = incoming_edge_csr(A_csr)
        dep_count_csr = torch.tensor(outgoing_edges, dtype=torch.int32, device=DEVICE)
        map_ptr_gpu = torch.tensor(map_ptr, dtype=torch.int32, device=DEVICE)
        map_vals_gpu = torch.tensor(map_vals, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_triton_csr_sync_free_mapping(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A_csr.nnz, dep_count_csr, map_ptr_gpu, map_vals_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'triton-csc-mapping':
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        # Convert to Python list with int values to avoid numpy.int32 issues
        wave_start_loc = [int(x) for x in wave_start_loc]
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_triton_csc_mapping(val_f32, indices_i32, indptr_i32, b_f32, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
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
    if provider == 'numba-csc-wavefront':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-csc-sync-free':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        #dep_count_gpu = cuda.to_device(dep_count)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count), quantiles=quantiles, warmup=warmpup, rep=rep)
    # CSC Wavefront Mapping with thread count sweep
    if provider.startswith('numba-csc-wavefront-mapping-t'):
        threads_per_block = int(provider.split('t')[-1])
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC Sync-Free Mapping with thread count sweep
    if provider.startswith('numba-csc-sync-free-mapping-t'):
        threads_per_block = int(provider.split('t')[-1])
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        succ_offsets_csc, succ_indices_csc = build_csc_successors(A)
        d_succ_offsets_csc = cuda.to_device(succ_offsets_csc)
        d_succ_indices_csc = cuda.to_device(succ_indices_csc)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC Wavefront Unrolling with factor and thread count sweep
    if provider.startswith('numba-csc-wavefront-unrolling-') and '-t' in provider:
        parts = provider.replace('numba-csc-wavefront-unrolling-', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        # Select the appropriate specialized function
        func_map = {
            4: sptrsv_numba_csc_wavefront_unrolling_4,
            8: sptrsv_numba_csc_wavefront_unrolling_8,
            16: sptrsv_numba_csc_wavefront_unrolling_16,
            32: sptrsv_numba_csc_wavefront_unrolling_32,
            64: sptrsv_numba_csc_wavefront_unrolling_64,
        }
        unroll_func = func_map[unroll_factor]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC Sync-Free Unrolling with factor and thread count sweep
    if provider.startswith('numba-csc-sync-free-unrolling-') and '-t' in provider:
        parts = provider.replace('numba-csc-sync-free-unrolling-', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        succ_offsets_csc, succ_indices_csc = build_csc_successors(A)
        d_succ_offsets_csc = cuda.to_device(succ_offsets_csc)
        d_succ_indices_csc = cuda.to_device(succ_indices_csc)
        # Select the appropriate specialized function
        func_map = {
            4: sptrsv_numba_csc_sync_free_unrolling_4,
            8: sptrsv_numba_csc_sync_free_unrolling_8,
            16: sptrsv_numba_csc_sync_free_unrolling_16,
            32: sptrsv_numba_csc_sync_free_unrolling_32,
            64: sptrsv_numba_csc_sync_free_unrolling_64,
        }
        unroll_func = func_map[unroll_factor]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-csr-wavefront':
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    if provider == 'numba-csr-sync-free':
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices), quantiles=quantiles, warmup=warmpup, rep=rep)
    # CSR Wavefront Mapping with thread count sweep
    if provider.startswith('numba-csr-wavefront-mapping-t'):
        threads_per_block = int(provider.split('t')[-1])
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Wavefront Unrolling with factor and thread count sweep
    if provider.startswith('numba-csr-wavefront-unrolling-') and '-t' in provider:
        parts = provider.replace('numba-csr-wavefront-unrolling-', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        # Select the appropriate specialized function
        func_map = {
            4: sptrsv_numba_csr_wavefront_unrolling_4,
            8: sptrsv_numba_csr_wavefront_unrolling_8,
            16: sptrsv_numba_csr_wavefront_unrolling_16,
            32: sptrsv_numba_csr_wavefront_unrolling_32,
            64: sptrsv_numba_csr_wavefront_unrolling_64,
        }
        unroll_func = func_map[unroll_factor]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Sync-Free Mapping with thread count sweep
    if provider.startswith('numba-csr-sync-free-mapping-t'):
        threads_per_block = int(provider.split('t')[-1])
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Sync-Free Unrolling with factor and thread count sweep
    if provider.startswith('numba-csr-sync-free-unrolling-') and '-t' in provider:
        parts = provider.replace('numba-csr-sync-free-unrolling-', '').split('-t')
        unroll_factor = int(parts[0])
        threads_per_block = int(parts[1])
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        # Select the appropriate specialized function
        func_map = {
            4: sptrsv_numba_csr_sync_free_unrolling_4,
            8: sptrsv_numba_csr_sync_free_unrolling_8,
            16: sptrsv_numba_csr_sync_free_unrolling_16,
            32: sptrsv_numba_csr_sync_free_unrolling_32,
            64: sptrsv_numba_csr_sync_free_unrolling_64,
        }
        unroll_func = func_map[unroll_factor]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: unroll_func(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    # CSC Wavefront Block Coarsening
    if provider.startswith('numba-csc-wf-block-c'):
        coarsening_factor = int(provider.replace('numba-csc-wf-block-c', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC Wavefront Thread Coarsening
    if provider.startswith('numba-csc-wf-thread-t'):
        threads_per_block = int(provider.replace('numba-csc-wf-thread-t', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, 1)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC Sync-Free Block Coarsening
    if provider.startswith('numba-csc-sf-block-b'):
        block_coarsening_factor = int(provider.replace('numba-csc-sf-block-b', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, block_coarsening_factor=block_coarsening_factor), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC Sync-Free Thread Coarsening
    if provider.startswith('numba-csc-sf-thread-t'):
        threads_per_block = int(provider.replace('numba-csc-sf-thread-t', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Wavefront Block Coarsening
    if provider.startswith('numba-csr-wf-block-c'):
        coarsening_factor = int(provider.replace('numba-csr-wf-block-c', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Wavefront Thread Coarsening
    if provider.startswith('numba-csr-wf-thread-t'):
        threads_per_block = int(provider.replace('numba-csr-wf-thread-t', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, 1)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Sync-Free Block Coarsening
    if provider.startswith('numba-csr-sf-block-b'):
        block_coarsening_factor = int(provider.replace('numba-csr-sf-block-b', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, block_coarsening_factor=block_coarsening_factor), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR Sync-Free Thread Coarsening
    if provider.startswith('numba-csr-sf-thread-t'):
        threads_per_block = int(provider.replace('numba-csr-sf-thread-t', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge_csr_rows(A_csr)
        succ_offsets, succ_indices = build_csr_successors(A_csr)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_sync_free_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, dep_count, succ_offsets, succ_indices, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # H-Level CSC (with CUDA Graphs and Warp-Level Optimization)
    if provider.startswith('numba-csc-h-level-bw'):
        band_width = int(provider.replace('numba-csc-h-level-bw', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        # Pre-compute analysis OUTSIDE the benchmark loop
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        # Benchmark ONLY the solver
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # H-Level CSR (with CUDA Graphs and Warp-Level Optimization)
    if provider.startswith('numba-csr-h-level-bw'):
        band_width = int(provider.replace('numba-csr-h-level-bw', ''))
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)  # Note: CSR uses same wavefronts as CSC
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        # Pre-compute analysis OUTSIDE the benchmark loop
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        # Benchmark ONLY the solver
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSC 3-Level Coarsening (wavefront + warp + thread)
    if provider.startswith('numba-csc-coarsened-c') and '-w' in provider and '-tc' in provider:
        parts = provider.replace('numba-csc-coarsened-c', '').split('-w')
        coarsening_factor = int(parts[0])
        parts2 = parts[1].split('-tc')
        warps = int(parts2[0])
        thread_coarsening_factor = int(parts2[1])
        threads_per_block = warps * 32
        
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_coarsened_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, coarsening_factor, threads_per_block, thread_coarsening_factor=thread_coarsening_factor, level_boundaries=level_boundaries), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    # CSR 3-Level Coarsening (wavefront + warp + thread)
    if provider.startswith('numba-csr-coarsened-c') and '-w' in provider and '-tc' in provider:
        parts = provider.replace('numba-csr-coarsened-c', '').split('-w')
        coarsening_factor = int(parts[0])
        parts2 = parts[1].split('-tc')
        warps = int(parts2[0])
        thread_coarsening_factor = int(parts2[1])
        threads_per_block = warps * 32
        
        A_csr = A.tocsr()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_coarsened_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, coarsening_factor, threads_per_block, thread_coarsening_factor=thread_coarsening_factor, level_boundaries=level_boundaries), quantiles=quantiles, warmup=warmpup, rep=rep)

    perf = lambda ms: (A.shape[0] + 2 * A.nnz) / (ms * 1e-3) / 1e9
    #print(f"seconds: {ms}")
    #print(f"max: {perf(max_ms)}, min: {perf(min_ms)}, median: {perf(ms)}")
    print(f"{provider} -- {matrices} -- seconds: {ms}")
    return perf(ms), perf(min_ms), perf(max_ms)

# Call the benchmark function (plots disabled, CSV logging enabled)
benchmark.run(print_data=True, save_path=".", show_plots=False)

