import cupy as cp
import cupyx.scipy.sparse as cpx
import cupyx.scipy.sparse.linalg as cpx_linalg
import numpy as np

class CuPyGaussSeidelSolver:
    """
    Gauss-Seidel Baseline using standard cuSPARSE routines via CuPy.
    Performs iterative triangular solves.
    """
    def __init__(self, max_iters=10):
        self.max_iters = max_iters
        self.is_setup = False

    def setup(self, A_csr, b, x_init):
        # 1. Transfer raw data to GPU (CuPy arrays)
        self.d_data = cp.asarray(A_csr.data)
        self.d_indices = cp.asarray(A_csr.indices)
        self.d_indptr = cp.asarray(A_csr.indptr)
        self.n = A_csr.shape[0]
        
        # 2. Create CuPy Sparse Matrix
        # This is a thin wrapper around cusparse CSR
        self.A_gpu = cpx.csr_matrix(
            (self.d_data, self.d_indices, self.d_indptr), 
            shape=A_csr.shape
        )
        
        # 3. Split into Lower+Diagonal (L_D) and Upper (U)
        # GS Iteration: (L + D) * x_new = b - U * x_old
        # Note: 'format="csr"' ensures we stay in sparse format for fast SpMV/SpSV
        self.L_D = cpx.tril(self.A_gpu, k=0, format="csr") 
        self.U = cpx.triu(self.A_gpu, k=1, format="csr")
        
        self.b_gpu = cp.asarray(b)
        self.x_gpu = cp.asarray(x_init)
        self.x_init_gpu = self.x_gpu.copy()
        
        self.is_setup = True

    def reset_x(self):
        cp.copyto(self.x_gpu, self.x_init_gpu)

    def solve(self):
        for _ in range(self.max_iters):
            # 1. SpMV: Calculate RHS = b - U * x_old
            # CuPy calls cusparseSpMV here
            rhs = self.b_gpu - self.U.dot(self.x_gpu)
            
            # 2. Triangular Solve: x_new = (L+D) \ rhs
            # CuPy calls cusparseSpSV (or csrsv2) here
            # overwrite_b=True allows inplace optimization for temporary vectors
            self.x_gpu = cpx_linalg.spsolve_triangular(
                self.L_D, rhs, lower=True, overwrite_b=True
            )
            
        # Ensure completion for timing safety
        # (Though benchmark harness usually syncs, explicit sync here doesn't hurt)
        cp.cuda.Stream.null.synchronize()

    def download_result(self):
        return cp.asnumpy(self.x_gpu)

    def destroy(self):
        # CuPy handles GC automatically, but we can explicit clear to be safe
        self.A_gpu = None
        self.L_D = None
        self.U = None
        self.is_setup = False
