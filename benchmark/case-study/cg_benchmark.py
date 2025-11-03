import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
import scipy.io as scio
import time
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    sptrsv_numba_csr_wavefront_cudagraph,
    incoming_edge_csr_rows,
    build_csr_successors,
    build_csc_successors,
    REAL_TYPE,
    DEVICE
)


mtx_list = [
"/home/mad29/projects/testdata/matrices/1138_bus.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova1.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova2.mtx",
# "/home/mad29/projects/testdata/matrices/Dubcova3.mtx",
# "/home/mad29/projects/testdata/matrices/Kuu.mtx",
# "/home/mad29/projects/testdata/matrices/Muu.mtx",
# "/home/mad29/projects/testdata/matrices/Pres_Poisson.mtx",
# "/home/mad29/projects/testdata/matrices/Serena.mtx",
# "/home/mad29/projects/testdata/matrices/StocF-1465.mtx",
# "/home/mad29/projects/testdata/matrices/Trefethen_2000.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk08.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk09.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk10.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk11.mtx",
# "/home/mad29/projects/testdata/matrices/bcsstk12.mtx",
# "/home/mad29/projects/testdata/matrices/nd3k.mtx",
# "/home/mad29/projects/testdata/matrices/nd6k.mtx",
# "/home/mad29/projects/testdata/matrices/parabolic_fem.mtx",
# "/home/mad29/projects/testdata/matrices/shallow_water1.mtx",
# "/home/mad29/projects/testdata/matrices/shallow_water2.mtx",
# "/home/mad29/projects/testdata/matrices/thermal1.mtx",
# "/home/mad29/projects/testdata/matrices/thermal2.mtx",
# "/home/mad29/projects/testdata/matrices/2cubes_sphere.mtx",
]


def conjugate_gradient_preconditioned(A, b, preconditioner, max_iter=1000, tol=1e-6, verbose=False):
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    
    initial_residual = np.linalg.norm(r)
    
    if initial_residual < 1e-14:
        return x, 0, initial_residual, [initial_residual], True
    
    z = preconditioner(r)
    p = z.copy()
    
    rz_old = np.dot(r, z)
    
    residual_norms = []
    converged = False
    
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
            print(f"  Iteration {iteration}: residual = {residual_norm:.6e}")
        
        if residual_norm < tol * initial_residual:
            converged = True
            return x, iteration + 1, residual_norm, residual_norms, converged
        
        z = preconditioner(r)
        rz_new = np.dot(r, z)
        
        if abs(rz_new) < 1e-14:
            break
            
        beta = rz_new / rz_old
        p = z + beta * p
        
        rz_old = rz_new
    
    return x, max_iter, residual_norm, residual_norms, converged


class ILU0Preconditioner:
    def __init__(self, L, U, sptrsv_lower_fn):
        self.L = L
        self.U = U
        self.sptrsv_lower = sptrsv_lower_fn
    
    def apply(self, r):
        from scipy.sparse.linalg import spsolve_triangular
        y = self.sptrsv_lower(self.L, r)
        z = spsolve_triangular(self.U, y, lower=False)
        return z


def prepare_sptrsv_data(L, format='csc', device='cuda', is_lower=True):
    n = L.shape[0]
    nnz = L.nnz
    
    data_dict = {}
    data_dict['is_lower'] = is_lower
    
    if not is_lower:
        L = L.T
        format = 'csr' if format == 'csc' else 'csc'
    
    if format == 'csc':
        L_format = L.tocsc()
    else:
        L_format = L.tocsr()
    
    if device == 'cuda':
        data_dict['data'] = torch.tensor(L_format.data, dtype=REAL_TYPE, device=DEVICE)
        data_dict['indices'] = torch.tensor(L_format.indices, dtype=torch.int32, device=DEVICE)
        data_dict['indptr'] = torch.tensor(L_format.indptr, dtype=torch.int32, device=DEVICE)
        
        data_dict['data_numba'] = cuda.to_device(L_format.data.astype(np.float64))
        data_dict['indices_numba'] = cuda.to_device(L_format.indices.astype(np.int32))
        data_dict['indptr_numba'] = cuda.to_device(L_format.indptr.astype(np.int32))
    
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
    
    if format == 'csc':
        in_edges = incoming_edge(L_format)
    else:
        in_edges = incoming_edge_csr_rows(L_format)
    
    data_dict['in_edges'] = in_edges
    
    if device == 'cuda':
        data_dict['dep_counts'] = torch.tensor(in_edges, dtype=torch.int32, device=DEVICE)
    
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
    data_dict['format'] = format
    
    return data_dict


def benchmark_cg_preconditioned(mtx_path, max_iter=1000, tol=1e-12, num_runs=3):
    mat_name = os.path.basename(mtx_path)
    print(f"\n{'='*80}\nMatrix: {mat_name}\n{'='*80}")
    
    matrix = scio.mmread(mtx_path)
    A = matrix.tocsr()
    if A.data.dtype != np.float64:
        A.data = A.data.astype(np.float64)
    
    n = A.shape[0]
    nnz = A.nnz
    density = nnz / (n * n) * 100
    
    print(f"n={n}, nnz={nnz}, density={density:.4f}%, avg_nnz/row={nnz/n:.2f}")
    
    is_symmetric = np.allclose((A - A.T).data, 0, atol=1e-10)
    if not is_symmetric:
        A = (A + A.T) / 2
    
    min_diag = A.diagonal().min()
    if min_diag <= 0:
        shift = abs(min_diag) + 1.0
        A = A + shift * sp.eye(n)
    
    np.random.seed(42)
    b = np.random.rand(n)
    
    from scipy.sparse.linalg import spsolve, spilu, LinearOperator, cg as scipy_cg
    
    x_direct = spsolve(A, b)
    direct_residual = np.linalg.norm(A.dot(x_direct) - b) / np.linalg.norm(b)
    print(f"Direct solve: residual={direct_residual:.6e}")
    
    ilu_ref = spilu(A.tocsc(), drop_tol=0.0, fill_factor=1.0)
    
    def precond_matvec(x):
        return ilu_ref.solve(x)
    
    M = LinearOperator(A.shape, matvec=precond_matvec)
    x_ref, info = scipy_cg(A, b, M=M, rtol=tol, atol=0, maxiter=max_iter)
    ref_res = np.linalg.norm(A.dot(x_ref) - b) / np.linalg.norm(b)
    
    if info == 0:
        ref_converged = True
        ref_iters = "converged"
        print(f"SciPy CG: converged, residual={ref_res:.6e}")
    elif info > 0:
        ref_converged = False
        ref_iters = info
        print(f"SciPy CG: NOT converged after {info} iters, residual={ref_res:.6e}")
    else:
        ref_converged = False
        ref_iters = "error"
        print(f"SciPy CG: error (info={info}), residual={ref_res:.6e}")
    
    L_ref = (sp.tril(ilu_ref.L, k=-1) + sp.eye(ilu_ref.L.shape[0])).tocsc()
    U_ref = sp.triu(ilu_ref.U).tocsc()
    
    L_ilu = L_ref
    U_ilu = U_ref
    print(f"ILU(0): L_nnz={L_ilu.nnz}, U_nnz={U_ilu.nnz}")
    
    L_csc = L_ilu.tocsc()
    L_csr = L_ilu.tocsr()
    U_csc = U_ilu.tocsc()
    U_csr = U_ilu.tocsr()
    
    data_lower_csc = prepare_sptrsv_data(L_csc, format='csc', device='cuda', is_lower=True)
    data_lower_csr = prepare_sptrsv_data(L_csr, format='csr', device='cuda', is_lower=True)
    
    from scipy.sparse.linalg import spsolve_triangular
    
    test_rhs = np.ones(n)
    x_ref_sptrsv = spsolve_triangular(L_csc, test_rhs, lower=True)
    
    b_test = cuda.to_device(test_rhs.astype(np.float64))
    x_test_gpu = sptrsv_numba_csc(
        data_lower_csc['data_numba'], data_lower_csc['indices_numba'], data_lower_csc['indptr_numba'],
        b_test, data_lower_csc['M'], data_lower_csc['N'], data_lower_csc['nnz'],
        data_lower_csc['wave_start'], data_lower_csc['wavefronts_numba']
    )
    x_test_result = x_test_gpu.copy_to_host()
    
    sptrsv_error = np.linalg.norm(x_test_result - x_ref_sptrsv) / np.linalg.norm(x_ref_sptrsv)
    print(f"GPU SPTRSV validation: error={sptrsv_error:.6e}")
    
    implementations = []
    
    def numba_csc_wavefront_solve(L, b_vec):
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csc(
            data_lower_csc['data_numba'], data_lower_csc['indices_numba'], data_lower_csc['indptr_numba'],
            b_gpu, data_lower_csc['M'], data_lower_csc['N'], data_lower_csc['nnz'],
            data_lower_csc['wave_start'], data_lower_csc['wavefronts_numba']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSC Wavefront', numba_csc_wavefront_solve, 'csc'))
    
    def numba_csc_syncfree_solve(L, b_vec):
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csc_sync_free(
            data_lower_csc['data_numba'], data_lower_csc['indices_numba'], data_lower_csc['indptr_numba'],
            b_gpu, data_lower_csc['M'], data_lower_csc['N'], data_lower_csc['nnz'], data_lower_csc['in_edges']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSC Sync-Free', numba_csc_syncfree_solve, 'csc'))
    
    def numba_csr_wavefront_solve(L, b_vec):
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csr_wavefront(
            data_lower_csr['data_numba'], data_lower_csr['indices_numba'], data_lower_csr['indptr_numba'],
            b_gpu, data_lower_csr['M'], data_lower_csr['N'], data_lower_csr['nnz'],
            data_lower_csr['wave_start'], data_lower_csr['wavefronts_numba']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSR Wavefront', numba_csr_wavefront_solve, 'csr'))
    
    def numba_csr_syncfree_solve(L, b_vec):
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csr_sync_free(
            data_lower_csr['data_numba'], data_lower_csr['indices_numba'], data_lower_csr['indptr_numba'],
            b_gpu, data_lower_csr['M'], data_lower_csr['N'], data_lower_csr['nnz'],
            data_lower_csr['in_edges'], data_lower_csr['succ_offsets'], data_lower_csr['succ_indices']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSR Sync-Free', numba_csr_syncfree_solve, 'csr'))
    
    def numba_csr_wavefront_cudagraph_solve(L, b_vec):
        b_gpu = cuda.to_device(b_vec.astype(np.float64))
        result = sptrsv_numba_csr_wavefront_cudagraph(
            data_lower_csr['data_numba'], data_lower_csr['indices_numba'], data_lower_csr['indptr_numba'],
            b_gpu, data_lower_csr['M'], data_lower_csr['N'], data_lower_csr['nnz'],
            data_lower_csr['wave_start'], data_lower_csr['wavefronts']
        )
        return result.copy_to_host()
    
    implementations.append(('Numba CSR Wavefront CUDAGraph', numba_csr_wavefront_cudagraph_solve, 'csr'))
    
    results = []
    
    for name, sptrsv_lower, format_type in implementations:
        print(f"\n{'-'*80}\n{name}\n{'-'*80}")
        
        L_matrix = L_csc if format_type == 'csc' else L_csr
        U_matrix = U_csc if format_type == 'csc' else U_csr
        precond = ILU0Preconditioner(L_matrix, U_matrix, sptrsv_lower)
        
        warmup_x, warmup_iters, warmup_res, _, warmup_conv = conjugate_gradient_preconditioned(
            A, b, precond.apply, max_iter=min(20, max_iter), tol=tol, verbose=False
        )
        
        times = []
        iters_list = []
        residuals = []
        converged_list = []
        
        for run in range(num_runs):
            start = time.time()
            x_sol, iters, res_norm, res_history, converged = conjugate_gradient_preconditioned(
                A, b, precond.apply, max_iter=max_iter, tol=tol, verbose=False
            )
            cuda.synchronize()
            elapsed = time.time() - start
            
            times.append(elapsed)
            iters_list.append(iters)
            residuals.append(res_norm)
            converged_list.append(converged)
        
        median_time = np.median(times)
        median_iters = int(np.median(iters_list))
        median_residual = np.median(residuals)
        all_converged = all(converged_list)
        
        correctness_error = None
        if ref_converged and all_converged:
            correctness_error = np.linalg.norm(x_sol - x_ref) / np.linalg.norm(x_ref)
        
        results.append({
            'name': name,
            'time': median_time,
            'iterations': median_iters,
            'residual': median_residual,
            'converged': all_converged,
            'correctness_error': correctness_error,
            'success': True
        })
        
        print(f"time={median_time:.4f}s, iters={median_iters}, converged={all_converged}, residual={median_residual:.6e}")
    
    print(f"\n{'='*80}\nSUMMARY - {mat_name}\n{'='*80}")
    
    successful_results = [r for r in results if r['success'] and r['converged']]
    if successful_results:
        successful_results.sort(key=lambda x: x['time'])
        
        print(f"\n{'Implementation':<35} {'Time (s)':<12} {'Speedup':<10} {'Iters':<8}")
        print(f"{'-'*80}")
        
        baseline_time = successful_results[0]['time']
        
        for r in successful_results:
            speedup = baseline_time / r['time']
            print(f"{r['name']:<35} {r['time']:<12.4f} {speedup:<10.2f}x {r['iterations']:<8}")
    
    non_converged = [r for r in results if r['success'] and not r['converged']]
    if non_converged:
        print(f"\nDid not converge:")
        for r in non_converged:
            print(f"  {r['name']}: {r['iterations']} iters, residual={r['residual']:.6e}")
    
    return {
        'matrix': mat_name,
        'n': n,
        'nnz': nnz,
        'density': density,
        'reference_converged': ref_converged,
        'reference_iters': ref_iters,
        'results': results
    }


def main():
    print("\n" + "="*80)
    print("CG Preconditioned Benchmark - SPTRSV Performance")
    print("="*80)
    
    all_results = []
    
    for mtx_path in mtx_list:
        if not os.path.exists(mtx_path):
            print(f"\nSkipping {mtx_path} (not found)")
            continue
        
        result = benchmark_cg_preconditioned(mtx_path, max_iter=1000, tol=1e-12, num_runs=3)
        if result is not None:
            all_results.append(result)
    
    print(f"\n\n{'='*80}\nFINAL SUMMARY - ALL MATRICES\n{'='*80}")
    
    import pandas as pd
    
    summary_data = []
    for res in all_results:
        for impl_res in res['results']:
            if impl_res['success'] and impl_res['converged']:
                summary_data.append({
                    'Matrix': res['matrix'],
                    'Size': res['n'],
                    'NNZ': res['nnz'],
                    'Ref_Iters': res.get('reference_iters', 'N/A'),
                    'Implementation': impl_res['name'],
                    'Time (s)': impl_res['time'],
                    'Iterations': impl_res['iterations'],
                    'Correctness_Error': impl_res.get('correctness_error', 'N/A'),
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        pivot_time = df.pivot_table(
            values='Time (s)',
            index='Matrix',
            columns='Implementation',
            aggfunc='first'
        )
        
        print("\nExecution Time (seconds):")
        print(pivot_time.to_string())
        
        output_file = "cg_precond_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        print("\n" + "="*80)
        print("AVERAGE PERFORMANCE")
        print("="*80)
        
        avg_times = df.groupby('Implementation')['Time (s)'].mean().sort_values()
        baseline = avg_times.iloc[0]
        
        for impl, avg_time in avg_times.items():
            speedup = baseline / avg_time
            print(f"{impl:<35} {avg_time:<12.4f}s (speedup: {speedup:.2f}x)")
        
        print("\n" + "="*80)
        print("CORRECTNESS")
        print("="*80)
        
        correctness_df = df[df['Correctness_Error'] != 'N/A'].copy()
        if not correctness_df.empty:
            correctness_df['Correctness_Error'] = pd.to_numeric(correctness_df['Correctness_Error'])
            avg_errors = correctness_df.groupby('Implementation')['Correctness_Error'].agg(['mean', 'max'])
            
            print(f"\n{'Implementation':<35} {'Avg Error':<15} {'Max Error':<15}")
            print(f"{'-'*80}")
            for impl in avg_errors.index:
                mean_err = avg_errors.loc[impl, 'mean']
                max_err = avg_errors.loc[impl, 'max']
                status = "OK" if max_err < 1e-4 else "WARN"
                print(f"{impl:<35} {mean_err:<15.6e} {max_err:<15.6e} [{status}]")
    else:
        print("No converged results.")


if __name__ == '__main__':
    main()
