"""
Benchmark for supernodal SpTRSV solvers comparing:
- Python CPU sequential implementation
- Numba CPU sequential implementation
- Numba CPU wavefront implementation
- Numba GPU sync-free implementation
- Numba GPU wavefront implementation
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as spio
from sksparse.cholmod import cholesky

import cupy
import triton
import triton.language as tl
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress warnings
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
warnings.filterwarnings('ignore', message='converting matrix of class')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bench_utils import *
from sptrsv_supernodal import *
from utils.symbolic_cholesky_csc import (
    convert_csc_to_supernodal,
    get_supernodal_dag
)
from utils.inspectors import sptrsv_csc_supernodal_analysis_1


# Configuration
RELAXATION = 8
NUM_RUNS = 30

# GPU setup
os.environ['NUMBA_CUDA_PTX_VERSION'] = '84'
print(cupy.show_config())
device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")

# Benchmark configuration
mtx_list = [
    "/home/mad29/projects/testdata/supernodal_matrices/apache1.mtx",
    "/home/mad29/projects/testdata/supernodal_matrices/FEM_3D_thermal1.mtx",
    "/home/mad29/projects/testdata/supernodal_matrices/thermal1.mtx",
]

method_list = [
    "python-cpu-sequential",
    "numba-cpu-sequential",
    "numba-cpu-wavefront",
    "numba-gpu-syncfree",
    "numba-gpu-wavefront"
]


def correctness_test():
    """Run correctness tests for all methods on all matrices."""
    matrix_signatures = []
    method_correctness = {}
    
    for mtx_matrix_path in mtx_list:
        print(f"\n{'=' * 60}")
        print(f"Matrix: {mtx_matrix_path}")
        print(f"{'=' * 60}")
        
        # Load matrix and compute Cholesky factorization
        Ao = spio.mmread(mtx_matrix_path)
        A = sp.tril(Ao).tocsc()
        factor = cholesky(Ao)
        L = factor.L()
        mat_name = mtx_matrix_path.split("/")[-1]
        
        # Create test vector
        b = L.dot(np.ones(L.shape[0]))
        
        # Verify Cholesky factorization
        x = factor.solve_A(b)
        print(f"nnz L: {L.nnz}")
        print(f"Cholesky solve error: {np.linalg.norm(b - Ao.dot(x), ord=np.inf):.2e}")
        
        x_chol = factor.solve_L(b, use_LDLt_decomposition=False)
        print(f"Cholmod solve error: {np.linalg.norm(x_chol - np.ones(L.shape[0]), ord=np.inf):.2e}")
        
        # Convert to supernodal format
        Ls = convert_csc_to_supernodal(L, RELAXATION)
        
        # Compute matrix signatures
        matrix_signatures.append(get_matrix_signature(A, mat_name))
        signature_l = get_L_signature(L, Ls, mat_name)
        matrix_signatures[-1].update(signature_l)
        
        # Compute wavefront structure
        dag_L_super = get_supernodal_dag(Ls)
        wf_sup = compute_wavefront(dag_L_super)
        wave_start_loc, waves = compact_2d_list(wf_sup)
        n_supernodes = len(Ls['supernodes']) - 1
        n_wavefronts = len(wave_start_loc) - 1
        print(f"\nSupernodes: {n_supernodes}, Wavefronts: {n_wavefronts}")
        print(f"Max supernode width: {Ls['max_width']}")
        
        # Test 1: Python CPU sequential
        x_python = b.copy()
        sptrv_csc_supernodal(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['p'], 
                             Ls['x'], b, x_python)
        error_python = np.linalg.norm(x_python - np.ones(L.shape[0]), ord=np.inf)
        print(f"Python CPU sequential error: {error_python:.2e}")
        
        # Test 2: Numba CPU sequential
        x_numba = b.copy()
        sptrsv_csc_supernodal_numba(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['p'], 
                                    Ls['x'], b, x_numba)
        error_numba = np.linalg.norm(x_numba - np.ones(L.shape[0]), ord=np.inf)
        print(f"Numba CPU sequential error: {error_numba:.2e}")
        
        # Test 3: Numba CPU wavefront
        x_numba_wf = b.copy()
        sptrsv_csc_supernodal_wavefront_numba(Ls['supernodes'], Ls['ip'], Ls['i'], Ls['p'], 
                                              Ls['x'], b, x_numba_wf, wave_start_loc, waves)
        error_numba_wf = np.linalg.norm(x_numba_wf - np.ones(L.shape[0]), ord=np.inf)
        print(f"Numba CPU wavefront error: {error_numba_wf:.2e}")
        
        # Test 4: Numba GPU sync-free
        thread_flags = sptrsv_csc_supernodal_analysis_1(Ls['supernodes'], Ls['ip'], 
                                                        Ls['i'], Ls['col_to_super'])
        d_lx = cuda.to_device(Ls['x'])
        d_super = cuda.to_device(Ls['supernodes'])
        d_col_to_sup = cuda.to_device(Ls['col_to_super'])
        d_ip = cuda.to_device(Ls['ip'])
        d_i = cuda.to_device(Ls['i'])
        d_p = cuda.to_device(Ls['p'])
        d_thread_flags = cuda.to_device(thread_flags)
        d_x = cuda.to_device(b)
        
        sptrsv_csc_supernodal_numba_cuda(d_super, d_col_to_sup, d_ip, d_i, d_p, 
                                         d_lx, d_x, d_thread_flags, Ls['max_width'])
        x_cuda = d_x.copy_to_host()
        error_cuda = np.linalg.norm(x_cuda - np.ones(L.shape[0]), ord=np.inf)
        print(f"Numba GPU sync-free error: {error_cuda:.2e}")
        
        # Test 5: Numba GPU wavefront
        d_x_wf = cuda.to_device(b)
        d_wave_ptr = cuda.to_device(wave_start_loc)
        d_waves = cuda.to_device(waves)
        sptrsv_csc_supernodal_wavefront_cuda(d_super, d_ip, d_i, d_p, d_lx, 
                                             d_x_wf, d_wave_ptr, d_waves)
        x_cuda_wf = d_x_wf.copy_to_host()
        error_cuda_wf = np.linalg.norm(x_cuda_wf - np.ones(L.shape[0]), ord=np.inf)
        print(f"Numba GPU wavefront error: {error_cuda_wf:.2e}")
    
    return matrix_signatures, method_correctness


def setup_benchmark_config():
    """Setup triton benchmark configuration."""
    config = triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=mtx_list,
        line_arg="provider",
        line_vals=method_list,
        line_names=[
            "Python CPU Sequential",
            "Numba CPU Sequential",
            "Numba CPU Wavefront",
            "Numba GPU Sync-Free",
            "Numba GPU Wavefront"
        ],
        styles=[
            ("green", "-"),
            ("blue", "-"),
            ("red", "-"),
            ("purple", "-"),
            ("orange", "-")
        ],
        ylabel="GFLOP/S",
        plot_name="sptrsv-supernodal-performance",
        args={}
    )
    return [config]


@triton.testing.perf_report(setup_benchmark_config())
def benchmark(matrices, provider):
    """Benchmark a specific method on a specific matrix."""
    # Load matrix and compute Cholesky factorization
    matrix = spio.mmread(matrices)
    factor = cholesky(matrix)
    L = factor.L()
    b = L.dot(np.ones(L.shape[0]))
    
    # Convert to supernodal format
    Ls = convert_csc_to_supernodal(L, RELAXATION)
    
    # Compute wavefront structure
    dag_L_super = get_supernodal_dag(Ls)
    wf_sup = compute_wavefront(dag_L_super)
    wave_start_loc, waves = compact_2d_list(wf_sup)
    n_supernodes = len(Ls['supernodes']) - 1
    n_wavefronts = len(wave_start_loc) - 1
    print(f"{matrices}: supernodes={n_supernodes}, wavefronts={n_wavefronts}")
    
    # Prepare thread flags for GPU sync-free method
    thread_flags = sptrsv_csc_supernodal_analysis_1(Ls['supernodes'], Ls['ip'],
                                                    Ls['i'], Ls['col_to_super'])
    
    # Benchmark parameters
    quantiles = [0.5, 0.2, 0.8]
    rep, warmup = 10, 2
    
    # Benchmark the requested method
    if provider == 'python-cpu-sequential':
        x = b.copy()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sptrv_csc_supernodal(Ls['supernodes'], Ls['ip'], Ls['i'], 
                                         Ls['p'], Ls['x'], b, x),
            quantiles=quantiles, warmup=warmup, rep=rep
        )
    
    elif provider == 'numba-cpu-sequential':
        x = b.copy()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sptrsv_csc_supernodal_numba(Ls['supernodes'], Ls['ip'], Ls['i'], 
                                                Ls['p'], Ls['x'], b, x),
            quantiles=quantiles, warmup=warmup, rep=rep
        )
    
    elif provider == 'numba-cpu-wavefront':
        x = b.copy()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sptrsv_csc_supernodal_wavefront_numba(Ls['supernodes'], Ls['ip'], Ls['i'], 
                                                          Ls['p'], Ls['x'], b, x, 
                                                          wave_start_loc, waves),
            quantiles=quantiles, warmup=warmup, rep=rep
        )
    
    elif provider == 'numba-gpu-syncfree':
        d_lx = cuda.to_device(Ls['x'])
        d_super = cuda.to_device(Ls['supernodes'])
        d_col_to_sup = cuda.to_device(Ls['col_to_super'])
        d_ip = cuda.to_device(Ls['ip'])
        d_i = cuda.to_device(Ls['i'])
        d_p = cuda.to_device(Ls['p'])
        d_thread_flags = cuda.to_device(thread_flags)
        d_x = cuda.to_device(b)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sptrsv_csc_supernodal_numba_cuda(d_super, d_col_to_sup, d_ip, d_i, 
                                                     d_p, d_lx, d_x, d_thread_flags, 
                                                     Ls['max_width']),
            quantiles=quantiles, warmup=warmup, rep=rep
        )
    
    elif provider == 'numba-gpu-wavefront':
        d_lx = cuda.to_device(Ls['x'])
        d_super = cuda.to_device(Ls['supernodes'])
        d_ip = cuda.to_device(Ls['ip'])
        d_i = cuda.to_device(Ls['i'])
        d_p = cuda.to_device(Ls['p'])
        d_x = cuda.to_device(b)
        d_wave_ptr = cuda.to_device(wave_start_loc)
        d_waves = cuda.to_device(waves)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sptrsv_csc_supernodal_wavefront_cuda(d_super, d_ip, d_i, d_p, 
                                                         d_lx, d_x, d_wave_ptr, d_waves),
            quantiles=quantiles, warmup=warmup, rep=rep
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Compute performance in GFLOP/S
    perf = lambda ms: (L.shape[0] + 2 * L.nnz) / (ms * 1e-3) / 1e9
    print(f"{provider} -- {matrices} -- ms: {ms:.2f}")
    
    return perf(ms), perf(min_ms), perf(max_ms)


if __name__ == "__main__":
    # # Run correctness tests
    # print("\n" + "=" * 60)
    # print("CORRECTNESS TESTS")
    # print("=" * 60)
    # mat_signatures, method_correctness = correctness_test()
    
    # # Save results to CSV
    # df = pd.DataFrame(mat_signatures)
    # df.to_csv("matrix_signatures.csv", index=False)
    # print(f"\nMatrix signatures saved to matrix_signatures.csv")
    
    # df = pd.DataFrame(method_correctness)
    # df.to_csv("method_correctness.csv", index=False)
    # print(f"Method correctness saved to method_correctness.csv")
    
    # Run performance benchmarks
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    benchmark.run(print_data=True, save_path=".", show_plots=True)
