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

from kernels.sptrsv_baseline import *
from kernels.sptrsv_unrolled import (
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
from kernels.sptrsv_coarsened import (
    compute_coarsened_wavefronts, compact_2d_list, compact_coarsened_wavefronts,
    sptrsv_numba_csc_coarsened_mapping, sptrsv_numba_csr_coarsened_mapping,
    sptrsv_numba_csc_wavefront_block_coarsening, sptrsv_numba_csr_wavefront_block_coarsening,
    sptrsv_numba_csc_wavefront_thread_coarsening, sptrsv_numba_csr_wavefront_thread_coarsening,
    sptrsv_numba_csc_sync_free_block_coarsening, sptrsv_numba_csr_sync_free_block_coarsening,
    sptrsv_numba_csc_sync_free_thread_coarsening, sptrsv_numba_csr_sync_free_thread_coarsening,
)
from kernels.sptrsv_hlevel import (
    sptrsv_h_level_analyze,
    sptrsv_numba_csc_h_level_solve, sptrsv_numba_csr_h_level_solve,
    sptrsv_numba_csc_h_level_graph, sptrsv_numba_csr_h_level_graph,
    sptrsv_numba_csc_syncfree_h_level_graph, sptrsv_numba_csr_syncfree_h_level_graph,
    sptrsv_numba_csc_syncfree_h_level_solve, sptrsv_numba_csr_syncfree_h_level_solve,
    sptrsv_numba_csr_syncfree_persistent_graph, sptrsv_numba_csr_syncfree_persistent_solve,
    sptrsv_numba_csc_syncfree_persistent_graph, sptrsv_numba_csc_syncfree_persistent_solve,
)

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

VERBOSE = os.environ.get("SPTRSV_VERBOSE", "0") == "1"
SKIP_CORRECTNESS = os.environ.get("SPTRSV_SKIP_CORRECTNESS", "1") != "0"

def maybe_print(msg: str):
    if VERBOSE:
        print(msg)

device = cuda.get_current_device()
maybe_print(f"Using GPU: {device.name}")
maybe_print(f"Compute Capability: {device.compute_capability}")



DEFAULT_MTX_LIST = [
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
mtx_override = os.environ.get("SPTRSV_MATRIX")
mtx_list = [mtx_override] if mtx_override else DEFAULT_MTX_LIST

def matrix_basename(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem.replace(" ", "_")

QUICK = os.environ.get("SPTRSV_QUICK", "1") != "0"
WARMUP = int(os.environ.get("SPTRSV_WARMUP", "2" if QUICK else "25"))
REP = int(os.environ.get("SPTRSV_REP", "5" if QUICK else "100"))
PROVIDER_FILTER = os.environ.get("SPTRSV_PROVIDERS")
# ============================================================================
# Benchmark Configuration
# ============================================================================
# Parameter ranges for kernel variants
unroll_factors = [4, 8, 16, 32, 64]
thread_counts = [32, 64, 128]
block_coarsening_factors = [2, 4, 8, 16]
sync_free_block_factors = [1, 2, 4]
thread_coarsening_factors = [1, 2, 4]
wavefront_coarsening_factors = [1, 2, 4, 8]
band_widths = [1, 2, 4, 8, 16]

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
        f"csc-h-sf-bw{bw}",
        f"csr-h-sf-bw{bw}",
        f"csc-h-sf-persist-bw{bw}",
        f"csr-h-sf-persist-bw{bw}",
        ])

provider_whitelist = None
if PROVIDER_FILTER:
    provider_whitelist = {p.strip() for p in PROVIDER_FILTER.split(",") if p.strip()}
# elif QUICK:
#     provider_whitelist = {"cusparse", "csc-wf-base", "csc-sf-base", "csr-wf-base", "csr-sf-base"}

if provider_whitelist:
    before = len(method_list)
    method_list = [m for m in method_list if m in provider_whitelist]
    maybe_print(f"Filtered providers {before} -> {len(method_list)}")


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
        
        # --- MODIFIED: Upper Triangular ---
        # make it sparse UPPER triangular
        A = triu(A, format="csc")
        
        # add diagonals with 100
        A.setdiag(A.diagonal())
        A.sort_indices() # Ensure sorted indices
        mat_name = mtx_matrix_path.split("/")[-1]
        matrix_signtures.append(get_matrix_signature(A, mat_name))
        b = A.dot(np.ones(n))  # Dense vector b
        
        # Run CPU solver (reference solution)
        x_cpu, cpu_median = cpu_solver(A, b)
        
        # Initialize method correctness dict
        method_correctness[mat_name] = {}
        for method in method_list:
            method_correctness[mat_name][method] = 0
        
        # Test cusparse
        print("Testing cusparse...")
        x_gpu, gpu_median = gpu_solver(A, b)
        if np.allclose(x_cpu, x_gpu, atol=1e-5):
            print("cusparse: PASS")
            method_correctness[mat_name]['cusparse'] = 1
        else:
            print("cusparse: FAIL")

        # Prepare common data structures
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        in_edge = incoming_edge(A, upper_triangular=True)
        
        # Transfer data to the GPU for numba
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_b = cuda.to_device(b)
        waves_gpu = cuda.to_device(waves)
        
        # Test CSC baseline methods
        print("Testing csc-wf-base...")
        y_numba_gpu = sptrsv_numba_csc(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu)
        y_numba_cpu = y_numba_gpu.copy_to_host()
        if np.allclose(x_cpu, y_numba_cpu, atol=1e-5):
            print("csc-wf-base: PASS")
            method_correctness[mat_name]['csc-wf-base'] = 1
        else:
            print("csc-wf-base: FAIL")

        print("Testing csc-sf-base...")
        y_numba_gpu_sync_free = sptrsv_numba_csc_sync_free(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge)
        y_numba_cpu_sync_free = y_numba_gpu_sync_free.copy_to_host()
        if np.allclose(x_cpu, y_numba_cpu_sync_free, atol=1e-5):
            print("csc-sf-base: PASS")
            method_correctness[mat_name]['csc-sf-base'] = 1
        else:
            print("csc-sf-base: FAIL")

        # Build successor structures for mapping variants
        succ_offsets_csc, succ_indices_csc = build_csc_successors(A)
        d_succ_offsets_csc = cuda.to_device(succ_offsets_csc)
        d_succ_indices_csc = cuda.to_device(succ_indices_csc)
        
        # Test sample CSC wavefront unrolling variants (just one example)
        print("Testing csc-wf-u1-t128...")
        try:
            y_opt = sptrsv_numba_csc_wavefront_mapping(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block=128)
            y_opt_cpu = y_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                print("csc-wf-u1-t128: PASS")
                method_correctness[mat_name]['csc-wf-u1-t128'] = 1
            else:
                print("csc-wf-u1-t128: FAIL")
        except Exception as e:
            print(f"csc-wf-u1-t128 failed: {e}")

        # Test sample CSC sync-free mapping variant
        print("Testing csc-sf-u1-t128...")
        try:
            y_opt = sptrsv_numba_csc_sync_free_mapping(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block=128)
            y_opt_cpu = y_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                print("csc-sf-u1-t128: PASS")
                method_correctness[mat_name]['csc-sf-u1-t128'] = 1
            else:
                print("csc-sf-u1-t128: FAIL")
        except Exception as e:
            print(f"csc-sf-u1-t128 failed: {e}")

        # Test one CSC wavefront unrolling variant as an example
        print("Testing csc-wf-u4-t128...")
        try:
            y_opt = sptrsv_numba_csc_wavefront_unrolling_4(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu, threads_per_block=128)
            y_opt_cpu = y_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                print("csc-wf-u4-t128: PASS (assuming other unroll factors work similarly)")
                method_correctness[mat_name]['csc-wf-u4-t128'] = 1
            else:
                print("csc-wf-u4-t128: FAIL")
        except Exception as e:
            print(f"csc-wf-u4-t128 failed: {e}")

        # Test one CSC sync-free unrolling variant as an example
        print("Testing csc-sf-u4-t128...")
        try:
            y_opt = sptrsv_numba_csc_sync_free_unrolling_4(d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz, in_edge, d_succ_offsets_csc, d_succ_indices_csc, threads_per_block=128)
            y_opt_cpu = y_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_opt_cpu, atol=1e-5):
                print("csc-sf-u4-t128: PASS (assuming other unroll factors work similarly)")
                method_correctness[mat_name]['csc-sf-u4-t128'] = 1
            else:
                print("csc-sf-u4-t128: FAIL")
        except Exception as e:
            print(f"csc-sf-u4-t128 failed: {e}")

        # Test CSR baseline methods
        A_csr_numba = A.tocsr()
        A_csr_numba.sort_indices()
        d_data_csr = cuda.to_device(A_csr_numba.data)
        d_indices_csr = cuda.to_device(A_csr_numba.indices)
        d_indptr_csr = cuda.to_device(A_csr_numba.indptr)
        d_b_csr = cuda.to_device(b)
        wavefronts_csr_numba = compute_wavefront(A, upper_triangular=True)
        wave_start_loc_csr_numba, waves_csr_numba = compact_2d_list(wavefronts_csr_numba)
        waves_gpu_csr_numba = cuda.to_device(waves_csr_numba)
        
        print("Testing csr-wf-base...")
        y_numba_csr_wave = sptrsv_numba_csr_wavefront(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, wave_start_loc_csr_numba, waves_gpu_csr_numba)
        y_numba_csr_wave_cpu = y_numba_csr_wave.copy_to_host()
        
        if np.allclose(x_cpu, y_numba_csr_wave_cpu, atol=1e-5):
            print("csr-wf-base: PASS")
            method_correctness[mat_name]['csr-wf-base'] = 1
        else:
            print("csr-wf-base: FAIL")

        # Test numba-csr-sync-free
        print("Testing csr-sf-base...")
        in_edges_csr = incoming_edge_csr_rows(A_csr_numba)
        succ_offsets_csr, succ_indices_csr = build_csr_successors(A_csr_numba)
        y_numba_csr_sync_free = sptrsv_numba_csr_sync_free(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, in_edges_csr, succ_offsets_csr, succ_indices_csr)
        y_numba_csr_sync_free_cpu = y_numba_csr_sync_free.copy_to_host()
        
        if np.allclose(x_cpu, y_numba_csr_sync_free_cpu, atol=1e-5):
            print("csr-sf-base: PASS")
            method_correctness[mat_name]['csr-sf-base'] = 1
        else:
            print("csr-sf-base: FAIL")

        # Test one CSR wavefront mapping variant as an example
        print("Testing csr-wf-u1-t128...")
        try:
            y_csr_opt = sptrsv_numba_csr_wavefront_mapping(d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz, wave_start_loc_csr_numba, waves_gpu_csr_numba, threads_per_block=128)
            y_csr_opt_cpu = y_csr_opt.copy_to_host()
            
            if np.allclose(x_cpu, y_csr_opt_cpu, atol=1e-5):
                print("csr-wf-u1-t128: PASS (assuming other variants work similarly)")
                method_correctness[mat_name]['csr-wf-u1-t128'] = 1
            else:
                print("csr-wf-u1-t128: FAIL")
        except Exception as e:
            print(f"csr-wf-u1-t128 failed: {e}")

        # Test CSC wavefront-based coarsening with block/thread coarsening
        print("\nTesting CSC wavefront-based coarsening methods...")
        original_wavefronts = compute_wavefront(A, upper_triangular=True)
        
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
            
            # Test CSC Sync-Free H-Level (push)
            method_name = f'csc-h-sf-bw{bw}'
            try:
                dep_counts_csc = incoming_edge(A, upper_triangular=True)  # dependencies per column
                y_h_level_sf_csc = sptrsv_numba_csc_syncfree_h_level_graph(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz,
                    waves, wave_start_loc, dep_counts_csc, band_width=bw)
                y_h_level_sf_csc_cpu = y_h_level_sf_csc.copy_to_host()
                
                if np.allclose(x_cpu, y_h_level_sf_csc_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

            # Test CSC Sync-Free H-Level (Persistent Push)
            method_name = f'csc-h-sf-persist-bw{bw}'
            try:
                dep_counts_csc = incoming_edge(A, upper_triangular=True)
                y_h_level_sf_csc_persist = sptrsv_numba_csc_syncfree_persistent_graph(
                    d_data, d_indices, d_indptr, d_b, A.shape[0], A.shape[1], A.nnz,
                    waves, wave_start_loc, dep_counts_csc, band_width=bw)
                y_h_level_sf_csc_persist_cpu = y_h_level_sf_csc_persist.copy_to_host()
                
                if np.allclose(x_cpu, y_h_level_sf_csc_persist_cpu, atol=1e-5):
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

            # Test CSR Sync-Free H-Level
            method_name = f'csr-h-sf-bw{bw}'
            try:
                y_h_level_sf = sptrsv_numba_csr_syncfree_h_level_graph(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    waves_csr_numba, wave_start_loc_csr_numba, band_width=bw)
                y_h_level_sf_cpu = y_h_level_sf.copy_to_host()
                
                if np.allclose(x_cpu, y_h_level_sf_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

            # Test CSR Sync-Free H-Level (Persistent)
            method_name = f'csr-h-sf-persist-bw{bw}'
            try:
                y_h_level_sf_persist = sptrsv_numba_csr_syncfree_persistent_graph(
                    d_data_csr, d_indices_csr, d_indptr_csr, d_b_csr, n, n, A_csr_numba.nnz,
                    waves_csr_numba, wave_start_loc_csr_numba, band_width=bw)
                y_h_level_sf_persist_cpu = y_h_level_sf_persist.copy_to_host()
                
                if np.allclose(x_cpu, y_h_level_sf_persist_cpu, atol=1e-5):
                    print(f"\n{method_name} Solutions match within tolerance!")
                    method_correctness[mat_name][method_name] = 1
                else:
                    print(f"\nWarning: {method_name} Solutions do not match!")
            except Exception as e:
                print(f"\n{method_name} failed: {e}")

        # 3-level coarsening correctness tests already covered above in loop
        
    return matrix_signtures, method_correctness


if SKIP_CORRECTNESS:
    mat_signatures, method_correctness = [], {}
else:
    mat_signatures, method_correctness = correctness_test()
    import pandas as pd
    df = pd.DataFrame(mat_signatures)
    df.to_csv("matrix_signatures.csv")
    df = pd.DataFrame(method_correctness)
    df.to_csv("method_correctness.csv")

configs = []

plot_suffix = ""
if mtx_override:
    plot_suffix = f"-{matrix_basename(mtx_override)}"

configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],
        line_arg="provider",
        line_vals=method_list,
        line_names=method_list,
        ylabel="GFLOP/S",
        plot_name=f"sptrsv-performance{plot_suffix}",
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
    
    # --- MODIFIED: Upper Triangular (Backward Substitution) ---
    # make it sparse UPPER triangular
    A = triu(A, format="csc")
    
    if A.data.dtype != np.float64:
        print(f"Matrix {matrices} is not double precision")
        A.data = A.data.astype(np.float64)

    # add diagonals with 100
    A.setdiag(A.diagonal())
    A.sort_indices() # Ensure sorted indices
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
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csc-sf-base':
        # Transfer data to the GPU
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        dep_count = incoming_edge(A, upper_triangular=True)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csr-wf-base':
        A_csr = A.tocsr()
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider == 'csr-sf-base':
        A_csr = A.tocsr()
        A_csr.sort_indices()
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
        wavefronts = compute_wavefront(A, upper_triangular=True)
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
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
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
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-wf-coarse-block-wc'):
        coarsening_factor = int(provider.replace('csr-wf-coarse-block-wc', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_wavefront_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start, waves_gpu, level_boundaries, block_coarsening_factor=1), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csc-wf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csc-wf-coarse-thread-tc', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, 1)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_wavefront_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, wave_start, waves_gpu, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-wf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csr-wf-coarse-thread-tc', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
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
        wavefronts = compute_wavefront(A, upper_triangular=True)
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
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
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
        dep_count = incoming_edge(A, upper_triangular=True)
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
        A_csr.sort_indices()
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
        dep_count = incoming_edge(A, upper_triangular=True)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_block_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, block_coarsening_factor=block_coarsening_factor), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-sf-coarse-block-bc'):
        block_coarsening_factor = int(provider.replace('csr-sf-coarse-block-bc', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
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
        dep_count = incoming_edge(A, upper_triangular=True)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_sync_free_thread_coarsening(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, dep_count, thread_coarsening_factor=1, threads_per_block=threads_per_block), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-sf-coarse-thread-tc'):
        threads_per_block = int(provider.replace('csr-sf-coarse-thread-tc', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
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
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        # Pre-compute analysis OUTSIDE the benchmark loop
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        # Benchmark ONLY the solver
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csc-h-sf-bw'):
        band_width = int(provider.replace('csc-h-sf-bw', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        dep_counts = incoming_edge(A, upper_triangular=True)
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_syncfree_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, h_level_data, dep_counts), quantiles=quantiles, warmup=warmpup, rep=rep)

    if provider.startswith('csc-h-sf-persist-bw'):
        band_width = int(provider.replace('csc-h-sf-persist-bw', ''))
        d_data = cuda.to_device(A.data)
        d_indices = cuda.to_device(A.indices)
        d_indptr = cuda.to_device(A.indptr)
        d_x = cuda.to_device(b.copy())  # persistent solver expects b copy
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        dep_counts = incoming_edge(A, upper_triangular=True)
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csc_syncfree_persistent_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A.nnz, h_level_data, dep_counts), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-h-bw'):
        band_width = int(provider.replace('csr-h-bw', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)  # Note: CSR uses same wavefronts as CSC
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        # Pre-compute analysis OUTSIDE the benchmark loop
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        # Benchmark ONLY the solver
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)
    
    if provider.startswith('csr-h-sf-bw'):
        band_width = int(provider.replace('csr-h-sf-bw', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_syncfree_h_level_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)

    if provider.startswith('csr-h-sf-persist-bw'):
        band_width = int(provider.replace('csr-h-sf-persist-bw', ''))
        A_csr = A.tocsr()
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        h_level_data = sptrsv_h_level_analyze(d_indptr, d_indices, waves_gpu, wave_start_loc, band_width=band_width)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_syncfree_persistent_solve(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, h_level_data), quantiles=quantiles, warmup=warmpup, rep=rep)

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
        wavefronts = compute_wavefront(A, upper_triangular=True)
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
        A_csr.sort_indices()
        d_data = cuda.to_device(A_csr.data)
        d_indices = cuda.to_device(A_csr.indices)
        d_indptr = cuda.to_device(A_csr.indptr)
        d_x = cuda.to_device(np.zeros(n))
        wavefronts = compute_wavefront(A, upper_triangular=True)
        wave_start_loc, waves, level_boundaries = compact_coarsened_wavefronts(wavefronts, coarsening_factor)
        waves_gpu = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_numba_csr_coarsened_mapping(d_data, d_indices, d_indptr, d_x, A.shape[0], A.shape[1], A_csr.nnz, wave_start_loc, waves_gpu, coarsening_factor, threads_per_block, thread_coarsening_factor=thread_coarsening_factor, level_boundaries=level_boundaries), quantiles=quantiles, warmup=warmpup, rep=rep)

    perf = lambda ms: (A.shape[0] + 2 * A.nnz) / (ms * 1e-3) / 1e9
    print(f"{provider} -- {matrices} -- ms: {ms}")
    return perf(ms), perf(min_ms), perf(max_ms)

# Call the benchmark function (plots disabled, CSV logging enabled)
benchmark.run(print_data=False, save_path=".", show_plots=False)
