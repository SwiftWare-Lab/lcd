import sys
import os
import time
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
from numba import cuda
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import LinearOperator, cg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.gauss_seidel_hlevel import (
    gs_numba_hlevel_syncfree_persistent_solve,
    gs_h_level_analyze
)
from kernels.gauss_seidel_hlevel_diag import (
    gs_numba_hlevel_syncfree_persistent_solve_diag,
    gs_h_level_analyze_diag,
    gs_numba_hlevel_syncfree_persistent_solve_diag_bwd
)
from kernels.gauss_seidel_utils import extract_diagonal_info_gpu
from kernels.gauss_seidel_cupy import CuPyGaussSeidelSolver 

def compute_wavefront_gs(A, upper_triangular=False):
    """Compute wavefront ordering based on matrix dependencies."""
    n = A.shape[0]
    A_csr = A.tocsr()
    in_degree = np.zeros(n, dtype=np.int32)
    
    if upper_triangular:
        for i in range(n):
            for ptr in range(A_csr.indptr[i], A_csr.indptr[i+1]):
                col = A_csr.indices[ptr]
                if col > i:
                    in_degree[i] += 1
    else:
        for i in range(n):
            for ptr in range(A_csr.indptr[i], A_csr.indptr[i+1]):
                col = A_csr.indices[ptr]
                if col < i:
                    in_degree[i] += 1
    
    wavefronts = []
    processed = np.zeros(n, dtype=bool)
    while not np.all(processed):
        wave = np.where((in_degree == 0) & (~processed))[0].astype(np.int32)
        if len(wave) == 0:
            break
        wavefronts.append(wave)
        processed[wave] = True
        
        for node in wave:
            for ptr in range(A_csr.indptr[node], A_csr.indptr[node+1]):
                col = A_csr.indices[ptr]
                if upper_triangular:
                    if col < node:
                        in_degree[col] -= 1
                else:
                    if col > node:
                        in_degree[col] -= 1
    return wavefronts

def compact_wavefronts(wavefronts):
    wave_start = [0]
    waves = []
    for wave in wavefronts:
        waves.extend(wave.tolist())
        wave_start.append(len(waves))
    return np.array(wave_start, dtype=np.int32), np.array(waves, dtype=np.int32)


# ============================================================================
# 1. OPTIMIZED PRECONDITIONER (Fixed for Numba-CuPy Interop)
# ============================================================================
class OptimizedGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix, bandwidth=8):
        self.shape = A_matrix.shape
        self.dtype = np.float64 
        n = A_matrix.shape[0]
        
        # 1. Setup GPU Data
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        # 2. FORWARD ANALYSIS (L-Part)
        print("  > [Ours] Analyzing FWD DAG...")
        wavefronts_fwd = compute_wavefront_gs(A_matrix)
        wave_start_fwd, waves_fwd = compact_wavefronts(wavefronts_fwd)
        self.d_waves_fwd = cuda.to_device(waves_fwd)
        
        self.analysis_fwd = gs_h_level_analyze_diag(
            self.d_indptr, self.d_indices, self.d_waves_fwd, 
            wave_start_fwd, band_width=bandwidth
        )

        # 3. BACKWARD ANALYSIS (U-Part)
        print("  > [Ours] Analyzing BWD DAG...")
        wavefronts_bwd = compute_wavefront_gs(A_matrix, upper_triangular=True)
        wave_start_bwd, waves_bwd = compact_wavefronts(wavefronts_bwd)
        self.d_waves_bwd = cuda.to_device(waves_bwd)
        
        self.analysis_bwd = gs_h_level_analyze_diag(
            self.d_indptr, self.d_indices, self.d_waves_bwd, 
            wave_start_bwd, band_width=bandwidth
        )

        self.sub_warp_size = 8

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        
        # Zero-Copy Views
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        # 1. FORWARD SWEEP (Uses FWD Analysis)
        gs_numba_hlevel_syncfree_persistent_solve_diag(
            self.d_data, self.d_indices, self.d_indptr, 
            self.d_diag_pos, self.d_diag_vals,
            r_numba, z_numba, 
            self.analysis_fwd, 
            max_iter=1, sub_warp_size=self.sub_warp_size
        )

        # 2. BACKWARD SWEEP (Uses BWD Analysis + BWD Kernel)
        # Note: We pass z_numba as both b and x to update in-place
        gs_numba_hlevel_syncfree_persistent_solve_diag_bwd(
            self.d_data, self.d_indices, self.d_indptr, 
            self.d_diag_pos, self.d_diag_vals,
            r_numba, z_numba, 
            self.analysis_bwd,  # <--- USES REVERSED SCHEDULE
            max_iter=1, sub_warp_size=self.sub_warp_size
        )
        
        return z_cupy


# ============================================================================
# 2. BASELINE PRECONDITIONER (CuPy Wrapper)
# ============================================================================
class BaselineGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        self.solver = CuPyGaussSeidelSolver(max_iters=1) 
        
        # Dummy setup
        n = A_matrix.shape[0]
        b_dummy = np.zeros(n)
        x_dummy = np.zeros(n) 
        self.solver.setup(A_matrix, b_dummy, x_dummy)

    def _matvec(self, r_gpu):
        self.solver.b_gpu = r_gpu
        self.solver.reset_x()
        self.solver.solve()
        return self.solver.x_gpu


# ============================================================================
# 3. BENCHMARK RUNNER
# ============================================================================
def run_pcg_benchmark(matrix_path):
    print(f"\n==================================================")
    print(f"Benchmarking Matrix: {os.path.basename(matrix_path)}")
    print(f"==================================================")

    # 1. Load Data
    matrix = spio.mmread(matrix_path)
    if sp.issparse(matrix):
        A_host = matrix.tocsr().astype(np.float64)
    else:
        A_host = sp.csr_matrix(matrix).astype(np.float64)
        
    print("  > Adjusting diagonal for convergence...")
    A_host.setdiag(A_host.diagonal() + np.abs(A_host).sum(axis=1).A1)
    
    A_gpu = csp.csr_matrix(A_host)
    n = A_host.shape[0]
    b_host = np.random.rand(n).astype(np.float64)
    b_gpu = cp.array(b_host)
    
    MAX_PCG_ITER = 200
    TOL = 1e-5

    # ------------------------------------------------------------------------
    # RUN BASELINE
    # ------------------------------------------------------------------------
    print("\n[Baseline] Setting up cuSPARSE Preconditioner...")
    M_base = BaselineGSPreconditioner(A_host)
    
    print("[Baseline] Running PCG...")
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    # Warmup
    cg(A_gpu, b_gpu, M=M_base, maxiter=5, tol=TOL)
    
    iter_count_base = [0]
    def callback_base(xk): iter_count_base[0] += 1
    
    start_event.record()
    x_sol_base, info_base = cg(A_gpu, b_gpu, M=M_base, maxiter=MAX_PCG_ITER, tol=TOL, callback=callback_base)
    end_event.record()
    end_event.synchronize()
    
    time_base = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    print(f"  > Time: {time_base:.4f} s | Iters: {iter_count_base[0]}")

    # ------------------------------------------------------------------------
    # RUN OURS
    # ------------------------------------------------------------------------
    print("\n[Ours] Setting up H-Level Persistent Preconditioner...")
    M_ours = OptimizedGSPreconditioner(A_host, bandwidth=1)
    
    print("[Ours] Running PCG...")
    
    # Warmup
    cg(A_gpu, b_gpu, M=M_ours, maxiter=5, tol=TOL)
    
    iter_count_ours = [0]
    def callback_ours(xk): iter_count_ours[0] += 1
    
    start_event.record()
    x_sol_ours, info_ours = cg(A_gpu, b_gpu, M=M_ours, maxiter=MAX_PCG_ITER, tol=TOL, callback=callback_ours)
    end_event.record()
    end_event.synchronize()
    
    time_ours = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    print(f"  > Time: {time_ours:.4f} s | Iters: {iter_count_ours[0]}")

    # ------------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------------
    print(f"\n[Result] Speedup: {time_base / time_ours:.2f}x")
    print(f"==================================================")


if __name__ == "__main__":
    matrices = [
        "/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
        "/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
    ]
    
    for mat in matrices:
        try:
            run_pcg_benchmark(mat)
        except Exception as e:
            print(f"Failed on {mat}: {e}")
            import traceback
            traceback.print_exc()