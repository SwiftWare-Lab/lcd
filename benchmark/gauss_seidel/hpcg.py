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

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.gauss_seidel_hlevel import (
    gs_numba_hlevel_syncfree_persistent_solve,
    gs_h_level_analyze
)
from kernels.gauss_seidel_hlevel_diag import (
    gs_numba_hlevel_syncfree_persistent_solve_diag,
    gs_h_level_analyze_diag
)
from kernels.gauss_seidel_baseline import (
    gs_numba_wavefront,
    gs_numba_syncfree
)
from kernels.gauss_seidel_baseline_diag import (
    gs_numba_wavefront_diag,
    gs_numba_syncfree_diag
)
from kernels.gauss_seidel_baseline_bwd import (
    gs_numba_wavefront_bwd,
    gs_numba_syncfree_bwd
)
from kernels.gauss_seidel_baseline_diag_bwd import (
    gs_numba_wavefront_diag_bwd,
    gs_numba_syncfree_diag_bwd
)
from kernels.gauss_seidel_hlevel_bwd import gs_numba_hlevel_syncfree_persistent_solve_bwd
from kernels.gauss_seidel_hlevel_diag_bwd import gs_numba_hlevel_syncfree_persistent_solve_diag_bwd
from kernels.gauss_seidel_baseline_bwd import gs_numba_syncfree_bwd
from kernels.gauss_seidel_baseline_diag_bwd import gs_numba_syncfree_diag_bwd
from kernels.gauss_seidel_unrolled import (
    gs_numba_syncfree_unrolling_4,
    gs_numba_syncfree_unrolling_8,
    gs_numba_syncfree_unrolling_16,
    gs_numba_syncfree_unrolling_32,
    gs_numba_syncfree_unrolling_64
)
from kernels.gauss_seidel_unrolled_bwd import (
    gs_numba_syncfree_unrolling_4_bwd,
    gs_numba_syncfree_unrolling_8_bwd,
    gs_numba_syncfree_unrolling_16_bwd,
    gs_numba_syncfree_unrolling_32_bwd,
    gs_numba_syncfree_unrolling_64_bwd
)
from kernels.gauss_seidel_unrolled_diag import (
    gs_numba_syncfree_unrolling_4_diag,
    gs_numba_syncfree_unrolling_8_diag,
    gs_numba_syncfree_unrolling_16_diag,
    gs_numba_syncfree_unrolling_32_diag,
    gs_numba_syncfree_unrolling_64_diag
)
from kernels.gauss_seidel_unrolled_diag_bwd import (
    gs_numba_syncfree_unrolling_4_diag_bwd,
    gs_numba_syncfree_unrolling_8_diag_bwd,
    gs_numba_syncfree_unrolling_16_diag_bwd,
    gs_numba_syncfree_unrolling_32_diag_bwd,
    gs_numba_syncfree_unrolling_64_diag_bwd
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


def build_csc_upper_triangle(A_csr):
    """Build CSC format for upper triangle (for backward pass dependency tracking)."""
    n = A_csr.shape[0]
    A_csc = A_csr.tocsc()
    
    csc_indices_U = []
    csc_indptr_U = [0]
    base_incom_U = np.zeros(n, dtype=np.int32)
    
    for col in range(n):
        count = 0
        for ptr in range(A_csc.indptr[col], A_csc.indptr[col+1]):
            row = A_csc.indices[ptr]
            if row < col:  # Upper triangle: row < col
                csc_indices_U.append(row)
                base_incom_U[row] += 1
                count += 1
        csc_indptr_U.append(csc_indptr_U[-1] + count)
    
    return (np.array(csc_indices_U, dtype=np.int32), 
            np.array(csc_indptr_U, dtype=np.int32), 
            base_incom_U)

def build_csc_lower_triangle(A_csr):
    """
    Build CSC format for lower triangle (for forward pass dependency tracking).
    
    Args:
        A_csr: scipy.sparse.csr_matrix
    
    Returns:
        csc_indices_L: np.ndarray[int32] - row indices for lower triangle in CSC
        csc_indptr_L: np.ndarray[int32] - column pointers for lower triangle
        base_incom_L: np.ndarray[int32] - incoming edge count for each row
    """
    n = A_csr.shape[0]
    A_csc = A_csr.tocsc()
    
    csc_indices_L = []
    csc_indptr_L = [0]
    base_incom_L = np.zeros(n, dtype=np.int32)
    
    for col in range(n):
        count = 0
        for ptr in range(A_csc.indptr[col], A_csc.indptr[col+1]):
            row = A_csc.indices[ptr]
            if row > col:  # Lower triangle: row > col
                csc_indices_L.append(row)
                base_incom_L[row] += 1
                count += 1
        csc_indptr_L.append(csc_indptr_L[-1] + count)
    
    return (np.array(csc_indices_L, dtype=np.int32), 
            np.array(csc_indptr_L, dtype=np.int32), 
            base_incom_L)


# ============================================================================
# 1. CUPY BASELINE PRECONDITIONER
# ============================================================================
class CuPyGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        self.solver = CuPyGaussSeidelSolver(max_iters=1) 
        
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
# 2. WAVEFRONT BASELINE (gs-wf-base)
# ============================================================================
class WavefrontGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        n = A_matrix.shape[0]
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        # Forward wavefront
        wavefronts_fwd = compute_wavefront_gs(A_matrix)
        wave_start_fwd, waves_fwd = compact_wavefronts(wavefronts_fwd)
        self.d_waves_fwd = cuda.to_device(waves_fwd)
        self.wave_start_fwd = cuda.to_device(wave_start_fwd)
        
        # Backward wavefront
        wavefronts_bwd = compute_wavefront_gs(A_matrix, upper_triangular=True)
        wave_start_bwd, waves_bwd = compact_wavefronts(wavefronts_bwd)
        self.d_waves_bwd = cuda.to_device(waves_bwd)
        self.wave_start_bwd = cuda.to_device(wave_start_bwd)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        # Forward + Backward
        gs_numba_wavefront(self.d_data, self.d_indices, self.d_indptr, 
                          r_numba, z_numba, self.wave_start_fwd, self.d_waves_fwd, max_iter=1)
        gs_numba_wavefront_bwd(self.d_data, self.d_indices, self.d_indptr, 
                               r_numba, z_numba, self.wave_start_bwd, self.d_waves_bwd, max_iter=1)
        
        return z_cupy


# ============================================================================
# 3. WAVEFRONT BASELINE WITH DIAG (gs-wf-base-diag)
# ============================================================================
class WavefrontGSDiagPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        wavefronts_fwd = compute_wavefront_gs(A_matrix)
        wave_start_fwd, waves_fwd = compact_wavefronts(wavefronts_fwd)
        self.d_waves_fwd = cuda.to_device(waves_fwd)
        self.wave_start_fwd = cuda.to_device(wave_start_fwd)
        
        wavefronts_bwd = compute_wavefront_gs(A_matrix, upper_triangular=True)
        wave_start_bwd, waves_bwd = compact_wavefronts(wavefronts_bwd)
        self.d_waves_bwd = cuda.to_device(waves_bwd)
        self.wave_start_bwd = cuda.to_device(wave_start_bwd)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_wavefront_diag(self.d_data, self.d_indices, self.d_indptr, 
                               self.d_diag_pos, self.d_diag_vals,
                               r_numba, z_numba, self.wave_start_fwd, self.d_waves_fwd, max_iter=1)
        gs_numba_wavefront_diag_bwd(self.d_data, self.d_indices, self.d_indptr, 
                                   self.d_diag_pos, self.d_diag_vals,
                                   r_numba, z_numba, self.wave_start_bwd, self.d_waves_bwd, max_iter=1)
        
        return z_cupy


# ============================================================================
# 4. SYNC-FREE BASELINE (gs-sf-base)
# ============================================================================
class SyncFreeGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        # Build CSC lower triangle for forward
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        # Build CSC upper triangle for backward
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree(self.d_data, self.d_indices, self.d_indptr,
                         self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                         r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_bwd(self.d_data, self.d_indices, self.d_indptr,
                             self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                             r_numba, z_numba, max_iter=1)
        
        return z_cupy


# ============================================================================
# 5. SYNC-FREE BASELINE WITH DIAG (gs-sf-base-diag)
# ============================================================================
class SyncFreeGSDiagPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_diag(self.d_data, self.d_indices, self.d_indptr,
                              self.d_diag_pos, self.d_diag_vals,
                              self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                              r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_diag_bwd(self.d_data, self.d_indices, self.d_indptr,
                                  self.d_diag_pos, self.d_diag_vals,
                                  self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                  r_numba, z_numba, max_iter=1)
        
        return z_cupy


# ============================================================================
# 6. LCD-OPTIMIZED (gs-h-sf-persist)
# ============================================================================
class LCDOptimizedGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix, bandwidth=8):
        self.shape = A_matrix.shape
        self.dtype = np.float64 
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        wavefronts_fwd = compute_wavefront_gs(A_matrix)
        wave_start_fwd, waves_fwd = compact_wavefronts(wavefronts_fwd)
        self.d_waves_fwd = cuda.to_device(waves_fwd)
        
        self.analysis_fwd = gs_h_level_analyze(
            self.d_indptr, self.d_indices, self.d_waves_fwd, 
            wave_start_fwd, band_width=bandwidth
        )

        wavefronts_bwd = compute_wavefront_gs(A_matrix, upper_triangular=True)
        wave_start_bwd, waves_bwd = compact_wavefronts(wavefronts_bwd)
        self.d_waves_bwd = cuda.to_device(waves_bwd)
        
        self.analysis_bwd = gs_h_level_analyze(
            self.d_indptr, self.d_indices, self.d_waves_bwd, 
            wave_start_bwd, band_width=bandwidth
        )

        self.sub_warp_size = 8

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_hlevel_syncfree_persistent_solve(
            self.d_data, self.d_indices, self.d_indptr, 
            r_numba, z_numba, 
            self.analysis_fwd, 
            max_iter=1, sub_warp_size=self.sub_warp_size
        )

        gs_numba_hlevel_syncfree_persistent_solve_bwd(
            self.d_data, self.d_indices, self.d_indptr, 
            r_numba, z_numba, 
            self.analysis_bwd,
            max_iter=1, sub_warp_size=self.sub_warp_size
        )
        
        return z_cupy


# ============================================================================
# 7. LCD-OPTIMIZED WITH DIAG (gs-h-sf-persist-diag)
# ============================================================================
class LCDOptimizedGSDiagPreconditioner(LinearOperator):
    def __init__(self, A_matrix, bandwidth=8):
        self.shape = A_matrix.shape
        self.dtype = np.float64 
        n = A_matrix.shape[0]
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        wavefronts_fwd = compute_wavefront_gs(A_matrix)
        wave_start_fwd, waves_fwd = compact_wavefronts(wavefronts_fwd)
        self.d_waves_fwd = cuda.to_device(waves_fwd)
        
        self.analysis_fwd = gs_h_level_analyze_diag(
            self.d_indptr, self.d_indices, self.d_waves_fwd, 
            wave_start_fwd, band_width=bandwidth
        )

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
        
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_hlevel_syncfree_persistent_solve_diag(
            self.d_data, self.d_indices, self.d_indptr, 
            self.d_diag_pos, self.d_diag_vals,
            r_numba, z_numba, 
            self.analysis_fwd, 
            max_iter=1, sub_warp_size=self.sub_warp_size
        )

        gs_numba_hlevel_syncfree_persistent_solve_diag_bwd(
            self.d_data, self.d_indices, self.d_indptr, 
            self.d_diag_pos, self.d_diag_vals,
            r_numba, z_numba, 
            self.analysis_bwd,
            max_iter=1, sub_warp_size=self.sub_warp_size
        )
        
        return z_cupy


# ============================================================================
# 8-17. SYNC-FREE UNROLLED VARIANTS (Regular)
# ============================================================================
class SyncFreeUnrolling4GSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_4(self.d_data, self.d_indices, self.d_indptr,
                                      self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                      r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_4_bwd(self.d_data, self.d_indices, self.d_indptr,
                                         self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                         r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling8GSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_8(self.d_data, self.d_indices, self.d_indptr,
                                      self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                      r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_8_bwd(self.d_data, self.d_indices, self.d_indptr,
                                         self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                         r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling16GSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_16(self.d_data, self.d_indices, self.d_indptr,
                                       self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                       r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_16_bwd(self.d_data, self.d_indices, self.d_indptr,
                                          self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                          r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling32GSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_32(self.d_data, self.d_indices, self.d_indptr,
                                       self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                       r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_32_bwd(self.d_data, self.d_indices, self.d_indptr,
                                          self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                          r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling64GSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_64(self.d_data, self.d_indices, self.d_indptr,
                                       self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                       r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_64_bwd(self.d_data, self.d_indices, self.d_indptr,
                                          self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                          r_numba, z_numba, max_iter=1)
        return z_cupy


# ============================================================================
# 18-27. SYNC-FREE UNROLLED VARIANTS (Diagonal-Optimized)
# ============================================================================
class SyncFreeUnrolling4DiagGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_4_diag(self.d_data, self.d_indices, self.d_indptr,
                                          self.d_diag_pos, self.d_diag_vals,
                                          self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                          r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_4_diag_bwd(self.d_data, self.d_indices, self.d_indptr,
                                              self.d_diag_pos, self.d_diag_vals,
                                              self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                              r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling8DiagGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_8_diag(self.d_data, self.d_indices, self.d_indptr,
                                          self.d_diag_pos, self.d_diag_vals,
                                          self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                          r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_8_diag_bwd(self.d_data, self.d_indices, self.d_indptr,
                                              self.d_diag_pos, self.d_diag_vals,
                                              self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                              r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling16DiagGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_16_diag(self.d_data, self.d_indices, self.d_indptr,
                                           self.d_diag_pos, self.d_diag_vals,
                                           self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                           r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_16_diag_bwd(self.d_data, self.d_indices, self.d_indptr,
                                               self.d_diag_pos, self.d_diag_vals,
                                               self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                               r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling32DiagGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_32_diag(self.d_data, self.d_indices, self.d_indptr,
                                           self.d_diag_pos, self.d_diag_vals,
                                           self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                           r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_32_diag_bwd(self.d_data, self.d_indices, self.d_indptr,
                                               self.d_diag_pos, self.d_diag_vals,
                                               self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                               r_numba, z_numba, max_iter=1)
        return z_cupy


class SyncFreeUnrolling64DiagGSPreconditioner(LinearOperator):
    def __init__(self, A_matrix):
        self.shape = A_matrix.shape
        self.dtype = np.float64
        
        self.d_data = cuda.to_device(A_matrix.data.astype(np.float64))
        self.d_indices = cuda.to_device(A_matrix.indices.astype(np.int32))
        self.d_indptr = cuda.to_device(A_matrix.indptr.astype(np.int32))
        self.d_diag_pos, self.d_diag_vals = extract_diagonal_info_gpu(A_matrix)
        
        csc_indices_L, csc_indptr_L, base_incom_L = build_csc_lower_triangle(A_matrix)
        self.d_csc_indices_L = cuda.to_device(csc_indices_L)
        self.d_csc_indptr_L = cuda.to_device(csc_indptr_L)
        self.d_base_incom_L = cuda.to_device(base_incom_L)
        
        csc_indices_U, csc_indptr_U, base_incom_U = build_csc_upper_triangle(A_matrix)
        self.d_csc_indices_U = cuda.to_device(csc_indices_U)
        self.d_csc_indptr_U = cuda.to_device(csc_indptr_U)
        self.d_base_incom_U = cuda.to_device(base_incom_U)

    def _matvec(self, r_gpu):
        z_cupy = cp.zeros(self.shape[0], dtype=np.float64)
        r_numba = cuda.as_cuda_array(cp.ascontiguousarray(r_gpu))
        z_numba = cuda.as_cuda_array(z_cupy)
        
        gs_numba_syncfree_unrolling_64_diag(self.d_data, self.d_indices, self.d_indptr,
                                           self.d_diag_pos, self.d_diag_vals,
                                           self.d_csc_indices_L, self.d_csc_indptr_L, self.d_base_incom_L,
                                           r_numba, z_numba, max_iter=1)
        gs_numba_syncfree_unrolling_64_diag_bwd(self.d_data, self.d_indices, self.d_indptr,
                                               self.d_diag_pos, self.d_diag_vals,
                                               self.d_csc_indices_U, self.d_csc_indptr_U, self.d_base_incom_U,
                                               r_numba, z_numba, max_iter=1)
        return z_cupy


# ============================================================================
# 28. BENCHMARK RUNNER
# ============================================================================
def run_pcg_benchmark(matrix_path):
    matrix_name = os.path.basename(matrix_path).replace('.mtx', '')
    print(f"\n[{matrix_name}]")

    # Load matrix
    matrix = spio.mmread(matrix_path)
    if sp.issparse(matrix):
        A_host = matrix.tocsr().astype(np.float64)
    else:
        A_host = sp.csr_matrix(matrix).astype(np.float64)
    
    A_gpu = csp.csr_matrix(A_host)
    n = A_host.shape[0]
    b_host = np.random.rand(n).astype(np.float64)
    b_gpu = cp.array(b_host)
    
    MAX_PCG_ITER = 1000
    TOL = 1e-6
    
    methods = [
        ("cupy", lambda: CuPyGSPreconditioner(A_host)),
        ("gs-wf-base", lambda: WavefrontGSPreconditioner(A_host)),
        ("gs-wf-base-diag", lambda: WavefrontGSDiagPreconditioner(A_host)),
        ("gs-sf-base", lambda: SyncFreeGSPreconditioner(A_host)),
        ("gs-sf-base-diag", lambda: SyncFreeGSDiagPreconditioner(A_host)),
        ("gs-sf-unroll-4", lambda: SyncFreeUnrolling4GSPreconditioner(A_host)),
        ("gs-sf-unroll-4-diag", lambda: SyncFreeUnrolling4DiagGSPreconditioner(A_host)),
        ("gs-sf-unroll-8", lambda: SyncFreeUnrolling8GSPreconditioner(A_host)),
        ("gs-sf-unroll-8-diag", lambda: SyncFreeUnrolling8DiagGSPreconditioner(A_host)),
        ("gs-sf-unroll-16", lambda: SyncFreeUnrolling16GSPreconditioner(A_host)),
        ("gs-sf-unroll-16-diag", lambda: SyncFreeUnrolling16DiagGSPreconditioner(A_host)),
        ("gs-sf-unroll-32", lambda: SyncFreeUnrolling32GSPreconditioner(A_host)),
        ("gs-sf-unroll-32-diag", lambda: SyncFreeUnrolling32DiagGSPreconditioner(A_host)),
        ("gs-sf-unroll-64", lambda: SyncFreeUnrolling64GSPreconditioner(A_host)),
        ("gs-sf-unroll-64-diag", lambda: SyncFreeUnrolling64DiagGSPreconditioner(A_host)),
        ("gs-h-sf-persist", lambda: LCDOptimizedGSPreconditioner(A_host, bandwidth=1)),
        ("gs-h-sf-persist-diag", lambda: LCDOptimizedGSDiagPreconditioner(A_host, bandwidth=1)),
    ]
    
    results = {}
    
    for i, (method_name, precond_factory) in enumerate(methods, 1):
        print(f"  [{i}/{len(methods)}] {method_name}...", end=' ', flush=True)
        try:
            M = precond_factory()
            
            # Warmup
            cg(A_gpu, b_gpu, M=M, maxiter=5, tol=TOL)
            
            iter_count = [0]
            def callback(xk): iter_count[0] += 1
            
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            
            start_event.record()
            x_sol, info = cg(A_gpu, b_gpu, M=M, maxiter=MAX_PCG_ITER, tol=TOL, callback=callback)
            end_event.record()
            end_event.synchronize()
            
            elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
            results[method_name] = {
                'time': elapsed_time,
                'iters': iter_count[0],
                'converged': info == 0
            }
            
            print(f"{elapsed_time:.3f}s ({iter_count[0]} iters)")
            
        except Exception as e:
            print(f"FAILED: {e}")
            results[method_name] = {'time': float('inf'), 'iters': -1, 'converged': False}

    return matrix_name, results


if __name__ == "__main__":
    import csv
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='HPCG benchmark with Gauss-Seidel preconditioners')
    parser.add_argument('--matrix', type=str, help='Single matrix file to benchmark')
    parser.add_argument('--matrix-list', type=str, help='File containing list of matrices')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results')
    args = parser.parse_args()
    
    # Determine which matrices to run
    matrices = []
    if args.matrix:
        # Single matrix mode (for bash script)
        matrices = [args.matrix]
    elif args.matrix_list:
        # Read from matrix list file
        with open(args.matrix_list, 'r') as f:
            matrices = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    else:
        # Default test matrices
        matrices = [
            "/home/mad29/projects/testdata/matrices/1138_bus.mtx",
            "/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
            "/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
        ]
    
    all_results = []
    
    if len(matrices) == 1:
        # Single matrix mode - minimal output
        print(f"Benchmarking: {os.path.basename(matrices[0])}")
    else:
        print(f"Starting HPCG benchmark on {len(matrices)} matrices...")
    
    for mat in matrices:
        try:
            matrix_name, results = run_pcg_benchmark(mat)
            all_results.append((matrix_name, results))
        except Exception as e:
            print(f"Failed on {mat}: {e}")
            import traceback
            traceback.print_exc()
    
    # Export to CSV
    all_methods = ["cupy", "gs-wf-base", "gs-wf-base-diag", "gs-sf-base", "gs-sf-base-diag",
                   "gs-sf-unroll-4", "gs-sf-unroll-4-diag", "gs-sf-unroll-8", "gs-sf-unroll-8-diag",
                   "gs-sf-unroll-16", "gs-sf-unroll-16-diag", "gs-sf-unroll-32", "gs-sf-unroll-32-diag",
                   "gs-sf-unroll-64", "gs-sf-unroll-64-diag",
                   "gs-h-sf-persist", "gs-h-sf-persist-diag"]
    
    if len(matrices) == 1:
        # Single matrix mode - output to matrix-specific CSV
        matrix_name = all_results[0][0]
        csv_filename = os.path.join(args.output_dir, f"hpcg_results_{matrix_name}.csv")
    else:
        # Multi-matrix mode - output to timestamped CSV
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        csv_filename = os.path.join(args.output_dir, f"hpcg_results_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['matrix'] + all_methods)
        
        # Write data (time in seconds for each method)
        for matrix_name, results in all_results:
            row = [matrix_name]
            for method in all_methods:
                if method in results:
                    row.append(f"{results[method]['time']:.4f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    
    if len(matrices) == 1:
        print(f"Results saved to: {csv_filename}")
    else:
        print(f"\n{'='*80}")
        print(f"Results exported to: {csv_filename}")
        print(f"{'='*80}")
        
        # Print summary table
        print(f"\n{'Matrix':<20} {'CuPy':<12} {'Best Custom':<12} {'Speedup':<10}")
        print(f"{'-'*80}")
        for matrix_name, results in all_results:
            cupy_time = results.get('cupy', {}).get('time', float('inf'))
            best_custom_time = min([r['time'] for m, r in results.items() if m != 'cupy'], default=float('inf'))
            speedup = cupy_time / best_custom_time if best_custom_time < float('inf') else 0.0
            print(f"{matrix_name:<20} {cupy_time:<12.3f} {best_custom_time:<12.3f} {speedup:<10.2f}x")