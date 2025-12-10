import numpy as np
from numba import cuda

try:
    import pyamgx
    PYAMGX_AVAILABLE = False
    # pyamgx.initialize()
except ImportError:
    PYAMGX_AVAILABLE = False

# ============================================================================
# CPU Baseline Implementations
# ============================================================================

def gauss_seidel_csr(A, IA, JA, b, x, tol=1e-6, max_iter=100):
    n = len(IA) - 1
    for iter in range(max_iter):
        norm = 0.0
        for i in range(n):
            x_old = x[i]
            sum = 0.0
            for j in range(IA[i], IA[i + 1]):
                if i != JA[j]:
                    sum += A[j] * x[JA[j]]
                else:
                    diag = A[j]
            x[i] = (b[i] - sum) / diag
            norm += (x[i] - x_old) ** 2
        if norm < tol:
            break
    return x, iter


def gauss_seidel_diag_csr(A, Ap, Ai, Diag, b, x, tol=1e-6, max_iter=100):
    n, iter = len(Ap) - 1, 0
    for iter in range(max_iter):
        norm = 0.0
        for i in range(n):
            x_old = x[i]
            sum = 0.0
            for j in range(Ap[i], Ap[i + 1]):
                if Ai[j] != i:  # Skip diagonal
                    sum += A[j] * x[Ai[j]]
            x[i] = (b[i] - sum) / Diag[i]
            norm += (x[i] - x_old) ** 2
        if norm < tol:
            break
    return x, iter


# ============================================================================
# GPU Baseline Implementations
# ============================================================================

@cuda.jit
def gs_wavefront_kernel(data, indices, indptr, b, x, 
                        wavefronts, wave_start, wave_idx):
    # One Thread per Row
    tid = cuda.grid(1)
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if tid < num_rows:
        row = wavefronts[start_wave + tid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]

        sigma = 0.0
        diag = 1.0
        
        for i in range(row_start, row_end):
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]

        x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront(data, indices, indptr, b, x, wave_start, wavefronts, max_iter=10):
    num_waves = len(wave_start) - 1
    threads_per_block = 128
    
    for k in range(max_iter):
        for i in range(num_waves):
            grid_len = wave_start[i+1] - wave_start[i]
            if grid_len > 0:
                blocks = (grid_len + threads_per_block - 1) // threads_per_block
                gs_wavefront_kernel[blocks, threads_per_block](
                    data, indices, indptr, b, x, wavefronts, wave_start, i
                )


@cuda.jit
def gs_syncfree_kernel(data, indices, indptr, 
                       csc_indices_L, csc_indptr_L,
                       incom_edges, b, x, num_rows):
    # 1 Warp per Row
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block

    for row in range(global_warp_id, num_rows, total_warps):

        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()

        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        sigma = 0.0
        diag = 1.0

        for i in range(row_start + lane_id, row_end, 32):
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]

        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)

        temp_diag = diag
        for offset in [16, 8, 4, 2, 1]:
            other_diag = cuda.shfl_down_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        temp_diag = diag
        for offset in [1, 2, 4, 8, 16]:
            other_diag = cuda.shfl_up_sync(mask, temp_diag, offset)
            if other_diag != 1.0:
                diag = other_diag
            temp_diag = diag
        if lane_id == 0 and diag == 1.0:
            for i in range(row_start, row_end):
                if indices[i] == row:
                    diag = data[i]
                    break
        diag = cuda.shfl_sync(mask, diag, 0)

        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag

        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()

        sig_start = csc_indptr_L[row]
        sig_end = csc_indptr_L[row+1]
        
        for p in range(sig_start + lane_id, sig_end, 32):
            dependent_row = csc_indices_L[p]
            cuda.atomic.add(incom_edges, dependent_row, -1)


@cuda.jit
def copy_device_array_kernel(src, dst, n):
    tid = cuda.grid(1)
    if tid < n:
        dst[tid] = src[tid]


def gs_numba_syncfree(data, indices, indptr, csc_indices_L, csc_indptr_L, 
                      base_incom_edges, b, x, max_iter=10, threads_per_block=128):
    M = len(indptr) - 1
    num_blocks = min(M, 1024)
    
    d_incom_edges = cuda.device_array_like(base_incom_edges)
    copy_blocks = (M + threads_per_block - 1) // threads_per_block
    
    for k in range(max_iter):
        # RESET Dependencies for this iteration
        copy_device_array_kernel[copy_blocks, threads_per_block](
            base_incom_edges, d_incom_edges, M
        )
        cuda.synchronize()
        
        gs_syncfree_kernel[num_blocks, threads_per_block](
            data, indices, indptr, csc_indices_L, csc_indptr_L, 
            d_incom_edges, b, x, M
        )
        cuda.synchronize()


# ============================================================================
# Library Baseline: AMGX (NVIDIA's Industrial-Strength GPU Library)
# ============================================================================

# _AMGX_INITIALIZED = False

class AMGXGaussSeidelSolver:
    """
    Wrapper for NVIDIA AMGX. 
    Assumes pyamgx.initialize() is called ONCE in the main script.
    """
    def __init__(self, max_iters=10, tol=1e-6):
        if not PYAMGX_AVAILABLE:
            raise RuntimeError("pyamgx is required")
        self.max_iters = int(max_iters) # Ensure int
        self.tol = float(tol)           # Ensure float
        self.is_setup = False

    def setup(self, A_csr, b, x_init):
        import json
        
        # Standard Dictionary Config
        # We explicitly set the determinant to ensure correct resource allocation
        config_dict = {
            "config_version": 2, 
            "solver": {
                "solver": "MULTICOLOR_GS",
                "max_iters": self.max_iters,
                "monitor_residual": 1,
                "tolerance": self.tol,
                "print_solve_stats": 0,
                "obtain_timings": 0
            }
        }
        
        # Create Config and Resources
        self.cfg = pyamgx.Config()
        self.cfg.create(json.dumps(config_dict))
        self.resources = pyamgx.Resources(self.cfg)

        # Create Matrix/Vectors
        self.solver = pyamgx.Solver(self.resources, self.cfg)
        self.d_A = pyamgx.Matrix()
        self.d_b = pyamgx.Vector()
        self.d_x = pyamgx.Vector()

        # Create handles (dDDI = double data, Device, Device, Int indices)
        self.d_A.create(self.resources, mode='dDDI')
        self.d_b.create(self.resources, mode='dDDI')
        self.d_x.create(self.resources, mode='dDDI')

        # Type enforcement
        if A_csr.indices.dtype != np.int32: A_csr.indices = A_csr.indices.astype(np.int32)
        if A_csr.indptr.dtype != np.int32: A_csr.indptr = A_csr.indptr.astype(np.int32)
        if A_csr.data.dtype != np.float64: A_csr.data = A_csr.data.astype(np.float64)

        # Upload
        self.d_A.upload_CSR(A_csr)
        self.solver.setup(self.d_A)
        self.d_b.upload(b.astype(np.float64))
        self.x_init_host = x_init.astype(np.float64)
        self.d_x.upload(self.x_init_host)
        
        self.is_setup = True

    def reset_x(self):
        self.d_x.upload(self.x_init_host)

    def solve(self):
        self.solver.solve(self.d_b, self.d_x)

    def download_result(self):
        return self.d_x.download()

    def destroy(self):
        if self.is_setup:
            self.d_A.destroy()
            self.d_b.destroy()
            self.d_x.destroy()
            self.solver.destroy()
            self.resources.destroy()
            self.cfg.destroy()
            self.is_setup = False

# ============================================================================
# Library Baseline: PETSc
# ============================================================================

try:
    import sys
    from petsc4py import PETSc
    PETSC_AVAILABLE = False
except ImportError:
    PETSC_AVAILABLE = False

class PETScGaussSeidelSolver:
    """
    Wrapper for PETSc GPU Gauss-Seidel (SOR).
    Uses 'aijcusparse' backend which leverages NVIDIA cuSPARSE.
    """
    def __init__(self, max_iters=10, tol=1e-6):
        if not PETSC_AVAILABLE:
            raise RuntimeError("petsc4py is required")
        self.max_iters = max_iters
        self.tol = tol
        self.is_setup = False

    def setup(self, A_csr, b, x_init):
        # 1. Create Matrix on GPU
        # We create explicitly as 'aijcusparse' to force CUDA backend
        self.A = PETSc.Mat().create()
        self.A.setSizes([A_csr.shape[0], A_csr.shape[1]])
        self.A.setType('aijcusparse') # <--- CRITICAL for GPU
        
        # Preallocate and fill (efficient bulk set)
        self.A.setUp()
        self.A.setValuesCSR(A_csr.indptr, A_csr.indices, A_csr.data)
        self.A.assemble()

        # 2. Create Vectors on GPU
        # 'cuda' type ensures data resides on device
        self.b = PETSc.Vec().create()
        self.b.setSizes(len(b))
        self.b.setType('cuda') 
        self.b.setValues(range(len(b)), b)
        self.b.assemble()

        self.x = self.b.duplicate()
        self.x.setValues(range(len(x_init)), x_init)
        self.x.assemble()
        
        # Save initial x for reset
        self.x_init_vals = x_init.copy()
        
        self.is_setup = True

    def reset_x(self):
        # Reset x to initial guess
        self.x.setValues(range(len(self.x_init_vals)), self.x_init_vals)
        self.x.assemble()

    def solve(self):
        # Option 1: Direct SOR call
        # omega=1.0 is Gauss-Seidel. 
        # its=max_iters, lits=1 (local iters)
        # FIX: Use keyword 'sortype' instead of 'flag'
        self.A.SOR(self.b, self.x, omega=1.0, 
                   its=self.max_iters, lits=1,
                   sortype=PETSc.Mat.SORType.FORWARD_SWEEP)
        
        # Note: PETSc GPU operations are asynchronous. 
        # The benchmark script must call cuda.synchronize() after this function returns.

    def download_result(self):
        return self.x.getArray()

    def destroy(self):
        if self.is_setup:
            self.A.destroy()
            self.b.destroy()
            self.x.destroy()
            self.is_setup = False
