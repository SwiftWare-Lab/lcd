import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import scipy.io as scio
import triton
from numba import cuda

from bench_utils import *
from spilu0 import *

NUM_RUNS = 30
DEVICE = torch.device("cuda:0")

device = cuda.get_current_device()
print(f"Using GPU: {device.name}")
print(f"Compute Capability: {device.compute_capability}")

mtx_list = [
    "/home/mad29/projects/testdata/matrices/1138_bus.mtx"
]

method_list = [
    "numba-csr-wavefront", "numba-csr-sync-free",
    "numba-csc-wavefront", "numba-csc-sync-free",
    "cpu-csr-simple", "cpu-csc-simple"
]

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, min(len(mtx_list), 10))],
        line_arg="provider",
        line_vals=method_list,
        line_names=[
            "Numba CSR Wavefront", "Numba CSR Sync-free",
            "Numba CSC Wavefront", "Numba CSC Sync-free",
            "CPU CSR Simple", "CPU CSC Simple"
        ],
        styles=[
            ("red", "-"), ("purple", "-"),
            ("red", "--"), ("purple", "--"),
            ("orange", "-"), ("orange", "--")
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
    quantiles = [0.5, 0.2, 0.8]
    rep, warmup = 100, 25

    if provider == 'numba-csr-wavefront':
        wavefronts = compute_wavefront(A.tocsc())
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        def run_numba_csr_wave():
            A_copy = A.copy()
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            ilu0_csr_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu)
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_wave, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'numba-csc-wavefront':
        A_csc = A.tocsc()
        wavefronts = compute_wavefront(A_csc)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = cuda.to_device(waves)
        def run_numba_csc_wave():
            A_copy = A_csc.copy()
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            ilu0_csc_numba_wavefront(d_data, d_indices, d_indptr, n, wave_start_loc, waves_gpu)
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csc_wave, quantiles=quantiles, warmup=warmup, rep=rep)

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
            ilu0_csr_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices)
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csr_sync, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'numba-csc-sync-free':
        A_csc = A.tocsc()
        dep_counts = compute_ilu_dependency_counts_csc(A_csc)
        succ_offsets, succ_indices = build_successors_from_csc(A_csc)
        d_succ_offsets = cuda.to_device(succ_offsets)
        d_succ_indices = cuda.to_device(succ_indices)
        def run_numba_csc_sync():
            A_copy = A_csc.copy()
            d_data = cuda.to_device(A_copy.data)
            d_indices = cuda.to_device(A_copy.indices)
            d_indptr = cuda.to_device(A_copy.indptr)
            d_dep_counts = cuda.to_device(dep_counts.copy())
            ilu0_csc_numba_sync_free(d_data, d_indices, d_indptr, n, d_dep_counts, d_succ_offsets, d_succ_indices)
        ms, min_ms, max_ms = triton.testing.do_bench(run_numba_csc_sync, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'cpu-csr-simple':
        def run_cpu_csr_simple():
            A_copy = A.copy()
            simple_ilu0_csr(A_copy)
        ms, min_ms, max_ms = triton.testing.do_bench(run_cpu_csr_simple, quantiles=quantiles, warmup=warmup, rep=rep)

    elif provider == 'cpu-csc-simple':
        def run_cpu_csc_simple():
            A_copy = A.tocsc()
            simple_ilu0_csc(A_copy)
        ms, min_ms, max_ms = triton.testing.do_bench(run_cpu_csc_simple, quantiles=quantiles, warmup=warmup, rep=rep)

    perf = lambda t_ms: (2 * A.nnz) / (t_ms * 1e-3) / 1e9
    print(f"{provider} -- {matrices} -- time: {ms:.3f} ms")
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    # TODO (Da): ADD CORRECTNESS VALIDATION TEST
    benchmark.run(print_data=False, save_path=".", show_plots=False)
