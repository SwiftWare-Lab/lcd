import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
import time
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*NumbaPerformanceWarning.*')
warnings.filterwarnings('ignore', message='.*Grid size.*')

import torch
import cupy as cp
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

from bench_utils import incoming_edge, compute_wavefront
from sptrsv import (
    sptrsv_numba_csc,
    sptrsv_numba_csc_sync_free,
    sptrsv_numba_csc_wavefront_mapping,
    sptrsv_numba_csc_sync_free_mapping,
    sptrsv_numba_csr_wavefront,
    sptrsv_numba_csr_sync_free,
    sptrsv_numba_csr_wavefront_mapping,
    sptrsv_numba_csr_sync_free_mapping,
    sptrsv_numba_csr_wavefront_unrolling,
    sptrsv_numba_csr_sync_free_unrolling,
    sptrsv_sync_free_triton,
    sptrsv_sync_free_triton_csr,
    sptrsv_triton,
    sptrsv_triton_csr,
    sptrsv_triton_csc_mapping,
    sptrsv_triton_csr_mapping,
    sptrsv_triton_csr_sync_free_mapping,
    incoming_edge_csr_rows,
    build_csr_successors,
    build_csc_successors,
    REAL_TYPE,
    DEVICE
)


def generate_3d_stencil_matrix(nx, ny, nz):
    """
    Generate a deterministic 3D 27-point Laplacian matrix like in the HPCG benchmark.
    The matrix is symmetric positive definite (SPD) and diagonally dominant.
    
    Each grid point connects to its 26 spatial neighbors with weight -1,
    and the diagonal entry equals the number of neighbors (so A is SPD).
    
    Args:
        nx, ny, nz: Grid dimensions in each direction

    Returns:
        A (csr_matrix): SPD sparse matrix
        n (int): Total number of unknowns
    """
    n = nx * ny * nz

    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    row, col, data = [], [], []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = idx(i, j, k)
                diag = 0.0

                # Loop through 3x3x3 stencil neighborhood
                for dk in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for di in [-1, 0, 1]:
                            ni, nj, nk = i + di, j + dj, k + dk
                            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                                neighbor = idx(ni, nj, nk)
                                if (di, dj, dk) == (0, 0, 0):
                                    continue
                                # Off-diagonal weight: constant -1
                                row.append(center)
                                col.append(neighbor)
                                data.append(-1.0)
                                diag += 1.0

                # Diagonal value: number of valid neighbors
                row.append(center)
                col.append(center)
                data.append(diag)

    A = csr_matrix((data, (row, col)), shape=(n, n))
    return A, n



def incomplete_cholesky(A, fill_factor=1.0):
    """
    Compute incomplete Cholesky factorization: A ≈ L*L^T
    
    Args:
        A: Symmetric positive definite matrix (CSR format)
        fill_factor: Controls sparsity pattern (1.0 = same as A)
    
    Returns:
        L: Lower triangular factor in CSC format
    """
    from scipy.sparse.linalg import spilu
    
    # Use ILU as approximation and extract lower triangular part
    try:
        ilu = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=0.0)
        # NOTE: spilu returns L with unit diagonal (implicitly 1.0), must add explicitly
        L = sp.tril(ilu.L, k=-1) + sp.eye(ilu.L.shape[0])
        return L.tocsc()
    except:
        # Fallback: just use lower triangular part of A
        return sp.tril(A).tocsc()


def conjugate_gradient_preconditioned(A, b, preconditioner, max_iter=1000, tol=1e-6, verbose=False):
    """
    Preconditioned Conjugate Gradient solver.
    
    Args:
        A: System matrix (can be on CPU or GPU)
        b: Right-hand side vector
        preconditioner: Function that applies M^{-1} to a vector
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Print iteration info
    
    Returns:
        x: Solution vector
        iterations: Number of iterations
        residual_norm: Final residual norm
    """
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    
    initial_residual = np.linalg.norm(r)
    
    if initial_residual < 1e-14:
        return x, 0, initial_residual, [initial_residual]
    
    z = preconditioner(r)
    p = z.copy()
    
    rz_old = np.dot(r, z)
    
    residual_norms = []
    
    for iteration in range(max_iter):
        Ap = A.dot(p)
        pAp = np.dot(p, Ap)
        
        if abs(pAp) < 1e-14:
            break
            
        alpha = rz_old / pAp
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        residual_norm = np.linalg.norm(r)
        residual_norms.append(residual_norm)
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: residual = {residual_norm:.6e} (relative: {residual_norm/initial_residual:.6e})")
        
        if residual_norm < tol * initial_residual:
            if verbose:
                print(f"Converged in {iteration+1} iterations")
            return x, iteration + 1, residual_norm, residual_norms
        
        z = preconditioner(r)
        rz_new = np.dot(r, z)
        
        if abs(rz_new) < 1e-14:
            break
            
        beta = rz_new / rz_old
        p = z + beta * p
        
        rz_old = rz_new
    
    if verbose:
        print(f"Did not converge in {max_iter} iterations (final relative residual: {residual_norm/initial_residual:.6e})")
    return x, max_iter, residual_norm, residual_norms


class SymmetricGaussSeidelPreconditioner:
    """
    Symmetric Gauss-Seidel preconditioner: M^{-1} = (L+D)^{-1} * D * (D+L^T)^{-1}
    
    For simplicity, we approximate A ≈ L*L^T and use:
    M^{-1} ≈ L^{-T} * L^{-1}
    
    This requires two triangular solves per application.
    """
    
    def __init__(self, A, sptrsv_lower_fn, sptrsv_upper_fn=None, format='csc'):
        """
        Args:
            A: System matrix
            sptrsv_lower_fn: Function to solve L*x = b (lower triangular)
            sptrsv_upper_fn: Function to solve U*x = b (upper triangular), optional
            format: 'csc' or 'csr'
        """
        self.A = A
        self.sptrsv_lower = sptrsv_lower_fn
        self.sptrsv_upper = sptrsv_upper_fn
        self.format = format
        
        # Extract lower triangular part with diagonal
        if format == 'csc':
            self.L = sp.tril(A).tocsc()
        else:
            self.L = sp.tril(A).tocsr()
        
        # Ensure positive diagonal
        n = A.shape[0]
        for i in range(n):
            if format == 'csc':
                diag_idx = self.L.indptr[i]
                if diag_idx < self.L.indptr[i+1] and self.L.indices[diag_idx] == i:
                    if self.L.data[diag_idx] <= 0:
                        self.L.data[diag_idx] = 1.0
            else:
                row_start = self.L.indptr[i]
                row_end = self.L.indptr[i+1]
                for idx in range(row_start, row_end):
                    if self.L.indices[idx] == i:
                        if self.L.data[idx] <= 0:
                            self.L.data[idx] = 1.0
        
        # Transpose for upper triangular solve
        if format == 'csc':
            self.U = self.L.T.tocsr()
        else:
            self.U = self.L.T.tocsc()
    
    def apply(self, r):
        """
        Apply preconditioner: z = M^{-1} * r
        This requires solving L*y = r, then L^T*z = y
        """
        # Solve L*y = r
        y = self.sptrsv_lower(self.L, r)
        
        # Solve L^T*z = y
        if self.sptrsv_upper is not None:
            z = self.sptrsv_upper(self.U, y)
        else:
            # Use same solver for transposed matrix
            z = self.sptrsv_lower(self.U, y)
        
        return z


def prepare_sptrsv_data(L, format='csc', device='cuda'):
    """
    Prepare data structures needed for various SPTRSV implementations.
    
    Returns dictionary with all necessary arrays on GPU.
    """
    n = L.shape[0]
    nnz = L.nnz
    
    data_dict = {}
    
    # Convert to requested format
    if format == 'csc':
        L_format = L.tocsc()
    else:
        L_format = L.tocsr()
    
    # Move basic arrays to GPU
    if device == 'cuda':
        data_dict['data'] = torch.tensor(L_format.data, dtype=REAL_TYPE, device=DEVICE)
        data_dict['indices'] = torch.tensor(L_format.indices, dtype=torch.int32, device=DEVICE)
        data_dict['indptr'] = torch.tensor(L_format.indptr, dtype=torch.int32, device=DEVICE)
        
        # For numba
        data_dict['data_numba'] = cuda.to_device(L_format.data.astype(np.float64))
        data_dict['indices_numba'] = cuda.to_device(L_format.indices.astype(np.int32))
        data_dict['indptr_numba'] = cuda.to_device(L_format.indptr.astype(np.int32))
    
    # Compute wavefronts
    if format == 'csc':
        L_for_wavefront = L_format
    else:
        L_for_wavefront = L_format.tocsc()
    
    wavefronts_list = list(compute_wavefront(L_for_wavefront))
    wavefronts_flat = np.concatenate(wavefronts_list).astype(np.int32)
    wave_start = np.zeros(len(wavefronts_list) + 1, dtype=np.int32)
    for i, wf in enumerate(wavefronts_list):
        wave_start[i + 1] = wave_start[i] + len(wf)
    
    data_dict['wavefronts'] = wavefronts_flat
    data_dict['wave_start'] = wave_start
    
    if device == 'cuda':
        data_dict['wavefronts_gpu'] = torch.tensor(wavefronts_flat, dtype=torch.int32, device=DEVICE)
        data_dict['wavefronts_numba'] = cuda.to_device(wavefronts_flat)
    
    # Compute dependency counts for sync-free variants
    if format == 'csc':
        in_edges = incoming_edge(L_format)
    else:
        in_edges = incoming_edge_csr_rows(L_format)
    
    data_dict['in_edges'] = in_edges
    
    if device == 'cuda':
        data_dict['dep_counts'] = torch.tensor(in_edges, dtype=torch.int32, device=DEVICE)
    
    # Compute successor structures for CSR sync-free
    if format == 'csr':
        succ_offsets, succ_indices = build_csr_successors(L_format)
        data_dict['succ_offsets'] = succ_offsets
        data_dict['succ_indices'] = succ_indices
        if device == 'cuda':
            data_dict['succ_offsets_numba'] = cuda.to_device(succ_offsets)
            data_dict['succ_indices_numba'] = cuda.to_device(succ_indices)
    elif format == 'csc':
        succ_offsets, succ_indices = build_csc_successors(L_format)
        data_dict['succ_offsets'] = succ_offsets
        data_dict['succ_indices'] = succ_indices
        if device == 'cuda':
            data_dict['succ_offsets_numba'] = cuda.to_device(succ_offsets)
            data_dict['succ_indices_numba'] = cuda.to_device(succ_indices)
    
    data_dict['M'] = n
    data_dict['N'] = n
    data_dict['nnz'] = nnz
    
    # Compute max column/row length for Triton
    if format == 'csc':
        max_length = max(L_format.indptr[i+1] - L_format.indptr[i] for i in range(n))
    else:
        max_length = max(L_format.indptr[i+1] - L_format.indptr[i] for i in range(n))
    data_dict['max_length'] = max_length
    
    return data_dict


def benchmark_hpcg(nx, ny, nz, max_iter=100, tol=1e-6, num_runs=3):
    """
    Run HPCG-style benchmark with different SPTRSV implementations.
    
    Args:
        nx, ny, nz: Grid dimensions
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        num_runs: Number of benchmark runs
    """
    print(f"\n{'='*80}")
    print(f"HPCG Benchmark: {nx}x{ny}x{nz} grid ({nx*ny*nz} unknowns)")
    print(f"{'='*80}\n")
    
    # Generate problem
    print("Generating 3D stencil matrix...")
    A, n = generate_3d_stencil_matrix(nx, ny, nz)
    b = np.random.rand(n)
    
    print(f"Matrix size: {n}x{n}")
    print(f"Non-zeros: {A.nnz}")
    print(f"Density: {A.nnz / (n*n) * 100:.4f}%")
    print(f"Avg non-zeros per row: {A.nnz / n:.2f}")
    print(f"Matrix is sparse: {'YES' if A.nnz / (n*n) < 0.01 else 'NO (density too high)'}\n")
    
    # Test CG with simple Jacobi preconditioner first
    print("Testing CG convergence with Jacobi preconditioner...")
    diag_inv = 1.0 / A.diagonal()
    def jacobi_precond(r):
        return r * diag_inv
    
    x_test, iters_test, res_test, _ = conjugate_gradient_preconditioned(
        A, b, jacobi_precond, max_iter=200, tol=1e-6, verbose=False
    )
    print(f"  Jacobi preconditioner: {iters_test} iterations, residual = {res_test:.6e}")
    
    if iters_test >= 200:
        print("  WARNING: CG did not converge with Jacobi preconditioner!")
        print("  The problem may be ill-conditioned.\n")
    else:
        print(f"  CG works correctly! Relative residual: {res_test / np.linalg.norm(b):.6e}\n")
    
    # Prepare matrices for preconditioner
    L_csc = sp.tril(A).tocsc()
    L_csr = sp.tril(A).tocsr()
    U_csr = L_csc.T.tocsr()
    U_csc = L_csr.T.tocsc()
    
    # Verify L has correct structure for SPTRSV
    print("Verifying triangular matrices...")
    for i in range(min(5, n)):
        col_start_csc = L_csc.indptr[i]
        col_end_csc = L_csc.indptr[i+1]
        if col_end_csc > col_start_csc:
            first_row_idx = L_csc.indices[col_start_csc]
            if first_row_idx != i:
                print(f"  WARNING: L_csc column {i} does not start with diagonal! First row: {first_row_idx}")
                break
    else:
        print("  L_csc structure: OK (diagonal first in each column)")
    
    # Check CSR structure
    for i in range(min(5, n)):
        row_start_csr = L_csr.indptr[i]
        row_end_csr = L_csr.indptr[i+1]
        if row_end_csr > row_start_csr:
            last_col_idx = L_csr.indices[row_end_csr-1]
            if last_col_idx != i:
                print(f"  WARNING: L_csr row {i} does not end with diagonal! Last col: {last_col_idx}")
                break
    else:
        print("  L_csr structure: OK (diagonal last in each row)\n")
    
    # Prepare GPU data
    print("Preparing GPU data structures...")
    data_lower_csc = prepare_sptrsv_data(L_csc, format='csc', device='cuda')
    data_lower_csr = prepare_sptrsv_data(L_csr, format='csr', device='cuda')
    data_upper_csr = prepare_sptrsv_data(U_csr, format='csr', device='cuda')
    data_upper_csc = prepare_sptrsv_data(U_csc, format='csc', device='cuda')
    
    print("Data structures ready.\n")
    
    # Test SPTRSV correctness
    print("Testing SPTRSV correctness...")
    test_rhs = np.ones(n)
    
    # Test with CPU reference (scipy)
    from scipy.sparse.linalg import spsolve_triangular
    x_ref_csc = spsolve_triangular(L_csc, test_rhs, lower=True)
    
    # Test GPU solver
    b_test_gpu = cuda.to_device(test_rhs.astype(np.float64))
    x_test_gpu = sptrsv_numba_csc(
        data_lower_csc['data_numba'],
        data_lower_csc['indices_numba'],
        data_lower_csc['indptr_numba'],
        b_test_gpu,
        data_lower_csc['M'],
        data_lower_csc['N'],
        data_lower_csc['nnz'],
        data_lower_csc['wave_start'],
        data_lower_csc['wavefronts_numba']
    )
    x_test_result = x_test_gpu.copy_to_host()
    
    sptrsv_error = np.linalg.norm(x_test_result - x_ref_csc) / np.linalg.norm(x_ref_csc)
    print(f"  SPTRSV relative error: {sptrsv_error:.6e}")
    
    if sptrsv_error > 1e-6:
        print(f"  WARNING: SPTRSV solver may be incorrect! Error = {sptrsv_error:.6e}")
        print(f"  First 5 elements - GPU: {x_test_result[:5]}")
        print(f"  First 5 elements - CPU: {x_ref_csc[:5]}\n")
    else:
        print("  SPTRSV solver: CORRECT\n")
    
    # Define SPTRSV implementations to benchmark
    implementations = []
    
    # Numba CSC wavefront
    def numba_csc_wavefront_solve(L, b_vec):
        data_dict = data_lower_csc if L.shape == L_csc.shape else data_upper_csc
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csc(
            data_dict['data_numba'], data_dict['indices_numba'], data_dict['indptr_numba'],
            b_gpu, data_dict['M'], data_dict['N'], data_dict['nnz'],
            data_dict['wave_start'], data_dict['wavefronts_numba']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSC Wavefront', numba_csc_wavefront_solve, numba_csc_wavefront_solve))
    
    # Numba CSC sync-free
    def numba_csc_syncfree_solve(L, b_vec):
        data_dict = data_lower_csc if L.shape == L_csc.shape else data_upper_csc
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csc_sync_free(
            data_dict['data_numba'], data_dict['indices_numba'], data_dict['indptr_numba'],
            b_gpu, data_dict['M'], data_dict['N'], data_dict['nnz'], data_dict['in_edges']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSC Sync-Free', numba_csc_syncfree_solve, numba_csc_syncfree_solve))
    
    # Numba CSC wavefront mapping
    def numba_csc_wavefront_mapping_solve(L, b_vec):
        data_dict = data_lower_csc if L.shape == L_csc.shape else data_upper_csc
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csc_wavefront_mapping(
            data_dict['data_numba'], data_dict['indices_numba'], data_dict['indptr_numba'],
            b_gpu, data_dict['M'], data_dict['N'], data_dict['nnz'],
            data_dict['wave_start'], data_dict['wavefronts_numba']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSC Wavefront Mapping', numba_csc_wavefront_mapping_solve, numba_csc_wavefront_mapping_solve))
    
    # Numba CSC sync-free mapping
    def numba_csc_syncfree_mapping_solve(L, b_vec):
        data_dict = data_lower_csc if L.shape == L_csc.shape else data_upper_csc
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csc_sync_free_mapping(
            data_dict['data_numba'], data_dict['indices_numba'], data_dict['indptr_numba'],
            b_gpu, data_dict['M'], data_dict['N'], data_dict['nnz'],
            data_dict['in_edges'], data_dict['succ_offsets'], data_dict['succ_indices']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSC Sync-Free Mapping', numba_csc_syncfree_mapping_solve, numba_csc_syncfree_mapping_solve))
    
    # Numba CSR variants
    def numba_csr_wavefront_solve(L, b_vec):
        data_dict = data_lower_csr if L.shape == L_csr.shape else data_upper_csr
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csr_wavefront(
            data_dict['data_numba'], data_dict['indices_numba'], data_dict['indptr_numba'],
            b_gpu, data_dict['M'], data_dict['N'], data_dict['nnz'],
            data_dict['wave_start'], data_dict['wavefronts_numba']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSR Wavefront', numba_csr_wavefront_solve, numba_csr_wavefront_solve))
    
    def numba_csr_wavefront_mapping_solve(L, b_vec):
        data_dict = data_lower_csr if L.shape == L_csr.shape else data_upper_csr
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csr_wavefront_mapping(
            data_dict['data_numba'], data_dict['indices_numba'], data_dict['indptr_numba'],
            b_gpu, data_dict['M'], data_dict['N'], data_dict['nnz'],
            data_dict['wave_start'], data_dict['wavefronts_numba']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSR Wavefront Mapping', numba_csr_wavefront_mapping_solve, numba_csr_wavefront_mapping_solve))
    
    # Run benchmarks
    results = []
    
    for name, sptrsv_lower, sptrsv_upper in implementations:
        print(f"\n{'-'*80}")
        print(f"Testing: {name}")
        print(f"{'-'*80}")
        
        try:
            # Create preconditioner
            if 'CSC' in name:
                precond = SymmetricGaussSeidelPreconditioner(A, sptrsv_lower, sptrsv_upper, format='csc')
            else:
                precond = SymmetricGaussSeidelPreconditioner(A, sptrsv_lower, sptrsv_upper, format='csr')
            
            # Warmup with diagnostics
            print("Warming up...")
            warmup_x, warmup_iters, warmup_res, _ = conjugate_gradient_preconditioned(
                A, b, precond.apply, max_iter=min(10, max_iter), tol=tol, verbose=False
            )
            print(f"  Warmup: {warmup_iters} iterations, residual = {warmup_res:.6e}")
            
            # Benchmark
            times = []
            iters_list = []
            residuals = []
            
            for run in range(num_runs):
                print(f"Run {run+1}/{num_runs}...", end=' ', flush=True)
                start = time.time()
                x_sol, iters, res_norm, res_history = conjugate_gradient_preconditioned(
                    A, b, precond.apply, max_iter=max_iter, tol=tol, verbose=False
                )
                cuda.synchronize()
                elapsed = time.time() - start
                
                times.append(elapsed)
                iters_list.append(iters)
                residuals.append(res_norm)
                print(f"done ({elapsed:.4f}s, {iters} iters)")
            
            median_time = np.median(times)
            median_iters = int(np.median(iters_list))
            median_residual = np.median(residuals)
            
            results.append({
                'name': name,
                'time': median_time,
                'iterations': median_iters,
                'residual': median_residual,
                'success': True
            })
            
            print(f"\nMedian time: {median_time:.4f}s")
            print(f"Median iterations: {median_iters}")
            print(f"Median residual: {median_residual:.6e}")
            
        except Exception as e:
            print(f"FAILED: {str(e)}")
            results.append({
                'name': name,
                'time': float('inf'),
                'iterations': -1,
                'residual': float('inf'),
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        # Sort by time
        successful_results.sort(key=lambda x: x['time'])
        
        print(f"{'Implementation':<40} {'Time (s)':<12} {'Speedup':<10} {'Iters':<8}")
        print(f"{'-'*80}")
        
        baseline_time = successful_results[0]['time']
        
        for r in successful_results:
            speedup = baseline_time / r['time']
            print(f"{r['name']:<40} {r['time']:<12.4f} {speedup:<10.2f}x {r['iterations']:<8}")
    
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n{'Failed implementations:'}")
        for r in failed_results:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}\n")
    
    return results


if __name__ == '__main__':
    # Default problem sizes (can be adjusted)
    grid_sizes = [
        (16, 16, 16),   # Small: 4,096 unknowns
        (24, 24, 24),   # Medium: 13,824 unknowns
        (32, 32, 32),   # Large: 32,768 unknowns
    ]
    
    print("\n" + "="*80)
    print("HPCG-Style Benchmark for SPTRSV Implementations")
    print("="*80)
    
    for nx, ny, nz in grid_sizes:
        results = benchmark_hpcg(nx, ny, nz, max_iter=100, tol=1e-6, num_runs=3)
        
        # Save results
        output_file = f"hpcg_results_{nx}x{ny}x{nz}.txt"
        with open(output_file, 'w') as f:
            f.write(f"HPCG Benchmark Results: {nx}x{ny}x{nz} grid\n")
            f.write(f"{'='*80}\n\n")
            for r in results:
                f.write(f"{r['name']}\n")
                if r['success']:
                    f.write(f"  Time: {r['time']:.4f}s\n")
                    f.write(f"  Iterations: {r['iterations']}\n")
                    f.write(f"  Residual: {r['residual']:.6e}\n")
                else:
                    f.write(f"  FAILED: {r.get('error', 'Unknown')}\n")
                f.write("\n")
        
        print(f"Results saved to {output_file}")

