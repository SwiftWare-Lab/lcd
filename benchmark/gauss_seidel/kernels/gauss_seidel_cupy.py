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
        self.n = A_csr.shape[0]
        self.b_gpu = cp.asarray(b)
        self.x_gpu = cp.asarray(x_init)
        self.x_init_gpu = self.x_gpu.copy()

        # Transfer Data
        d_data = cp.asarray(A_csr.data)
        d_indices = cp.asarray(A_csr.indices)
        d_indptr = cp.asarray(A_csr.indptr)
        self.A_gpu = cpx.csr_matrix((d_data, d_indices, d_indptr), shape=A_csr.shape)

        # --- SETUP FOR SYMMETRIC GS ---
        # Forward needs: (L+D) and U
        # Backward needs: (U+D) and L
        
        # L + D (Lower Triangular + Diagonal)
        self.L_D = cpx.tril(self.A_gpu, k=0, format="csr")
        # U (Strict Upper Triangular)
        self.U = cpx.triu(self.A_gpu, k=1, format="csr")
        
        # U + D (Upper Triangular + Diagonal) - NEEDED FOR BACKWARD SWEEP
        self.U_D = cpx.triu(self.A_gpu, k=0, format="csr")
        # L (Strict Lower Triangular) - NEEDED FOR BACKWARD SWEEP
        self.L = cpx.tril(self.A_gpu, k=-1, format="csr")

        self.is_setup = True

    def reset_x(self):
        cp.copyto(self.x_gpu, self.x_init_gpu)

    def solve(self):
        for _ in range(self.max_iters):
            # ----------------------------------------
            # 1. Forward Sweep: (L+D) * x = b - U * x
            # ----------------------------------------
            rhs_fwd = self.b_gpu - self.U.dot(self.x_gpu)
            self.x_gpu = cpx_linalg.spsolve_triangular(
                self.L_D, rhs_fwd, lower=True, overwrite_b=True
            )

            # ----------------------------------------
            # 2. Backward Sweep: (U+D) * x = b - L * x
            # ----------------------------------------
            # This missing piece makes it Symmetric GS
            rhs_bwd = self.b_gpu - self.L.dot(self.x_gpu)
            self.x_gpu = cpx_linalg.spsolve_triangular(
                self.U_D, rhs_bwd, lower=False, overwrite_b=True
            )

        cp.cuda.Stream.null.synchronize()

    def download_result(self):
        return cp.asnumpy(self.x_gpu)

    def destroy(self):
        # CuPy handles GC automatically, but we can explicit clear to be safe
        self.A_gpu = None
        self.L_D = None
        self.U = None
        self.L = None
        self.is_setup = False
