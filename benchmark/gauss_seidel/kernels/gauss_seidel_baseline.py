import numpy as np
from numba import cuda

try:
    import pyamgx
    PYAMGX_AVAILABLE = True
    # Initialize ONCE at module level to guarantee it exists
    # This prevents "Invalid C Wrapper" errors caused by uninitialized contexts
    pyamgx.initialize()
except ImportError:
    PYAMGX_AVAILABLE = False
    print("Warning: pyamgx not found. AMGX baseline will be skipped.")

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
    """
    Standard Wavefront. One Thread per Row.
    """
    tid = cuda.grid(1)
    
    start_wave = wave_start[wave_idx]
    end_wave = wave_start[wave_idx + 1]
    num_rows = end_wave - start_wave
    
    if tid < num_rows:
        row = wavefronts[start_wave + tid]
        
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        # Compute Sum (A[i,:] * x[:])
        sigma = 0.0
        diag = 1.0
        
        for i in range(row_start, row_end):
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
        
        # Update x[i]
        x[row] = (b[row] - sigma) / diag


def gs_numba_wavefront(data, indices, indptr, b, x, wave_start, wavefronts, max_iter=10):
    """
    Driver: Loops over iterations, and inside that, loops over wavefronts.
    """
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
    """
    Sync-Free GS.
    1 Warp per Row.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x
    
    warp_id = tid // 32
    lane_id = tid % 32
    warps_per_block = block_dim // 32
    global_warp_id = (bid * warps_per_block) + warp_id
    total_warps = grid_dim * warps_per_block
    
    # Persistent Threads Loop
    for row in range(global_warp_id, num_rows, total_warps):
        
        # Wait (Dependency Check)
        if lane_id == 0:
            while cuda.atomic.add(incom_edges, row, 0) != 0:
                pass
        cuda.syncwarp()
        
        # Compute Row
        row_start = indptr[row]
        row_end = indptr[row + 1]
        
        sigma = 0.0
        diag = 1.0
        
        # Parallel Reduction over the row
        for i in range(row_start + lane_id, row_end, 32):
            col = indices[i]
            val = data[i]
            
            if col == row:
                diag = val
            else:
                sigma += val * x[col]
        
        # Warp Shuffle Reduction for sigma
        mask = 0xFFFFFFFF
        sigma += cuda.shfl_down_sync(mask, sigma, 16)
        sigma += cuda.shfl_down_sync(mask, sigma, 8)
        sigma += cuda.shfl_down_sync(mask, sigma, 4)
        sigma += cuda.shfl_down_sync(mask, sigma, 2)
        sigma += cuda.shfl_down_sync(mask, sigma, 1)
        
        # Reduce diag
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
        
        # Lane 0 updates
        if lane_id == 0:
            x[row] = (b[row] - sigma) / diag
            
        # Fence
        cuda.syncwarp()
        if lane_id == 0:
            cuda.threadfence()
        cuda.syncwarp()
        
        # Signal Dependents (Lower Triangle Only)
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
    """
    Driver. Must reset dependencies every iteration.
    threads_per_block: Number of threads per block (must be multiple of 32).
    """
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

class AMGXGaussSeidelSolver:
    """
    Robust Wrapper for NVIDIA AMGX Gauss-Seidel.
    Uses module-level initialization for stability.
    """
    def __init__(self, max_iters=10, tol=1e-6):
        if not PYAMGX_AVAILABLE:
            raise RuntimeError("pyamgx is required")
        self.max_iters = max_iters
        self.tol = tol
        self.is_setup = False

    def setup(self, A_csr, b, x_init):
        """
        Setup solver for a given matrix and pre-upload vectors.
        Call once per matrix before benchmarking.
        """
        # 1. Config: Use MULTICOLOR_GS
        # 'GAUSS_SEIDEL' is often just a smoother; 'MULTICOLOR_GS' is the top-level solver
        self.cfg = pyamgx.Config()
        self.cfg.create_from_dict({
            "config_version": 2,
            "solver": {
                "solver": "MULTICOLOR_GS",  # More robust than GAUSS_SEIDEL
                "max_iters": self.max_iters,
                "monitor_residual": 1, 
                "tolerance": self.tol,
                "print_solve_stats": 0,
            }
        })

        # 2. Create Resources
        # Because we initialized globally, this should now be valid
        self.resources = pyamgx.Resources(self.cfg)
        
        # 3. Create Solver
        self.solver = pyamgx.Solver(self.resources, self.cfg)

        # 4. Create Matrix/Vectors with Explicit Mode
        # Mode 'dDDI' = device, Double-Matrix, Double-Vector, Int-Indices
        self.d_A = pyamgx.Matrix()
        self.d_A.create(self.resources, mode='dDDI')
        
        self.d_b = pyamgx.Vector()
        self.d_b.create(self.resources, mode='dDDI')
        
        self.d_x = pyamgx.Vector()
        self.d_x.create(self.resources, mode='dDDI')

        # 5. Type Enforcement (Critical for dDDI mode)
        if A_csr.indices.dtype != np.int32: A_csr.indices = A_csr.indices.astype(np.int32)
        if A_csr.indptr.dtype != np.int32: A_csr.indptr = A_csr.indptr.astype(np.int32)
        if A_csr.data.dtype != np.float64: A_csr.data = A_csr.data.astype(np.float64)

        # 6. Upload
        self.d_A.upload_CSR(A_csr)
        self.solver.setup(self.d_A)
        self.d_b.upload(b.astype(np.float64))
        
        self.x_init_host = x_init.astype(np.float64)
        self.d_x.upload(self.x_init_host)
        
        self.is_setup = True

    def reset_x(self):
        """Reset x vector to initial value (excluded from benchmark timing)."""
        self.d_x.upload(self.x_init_host)

    def solve(self):
        """
        The hot kernel. This is what you should benchmark.
        """
        self.solver.solve(self.d_b, self.d_x)

    def download_result(self):
        """Download result from device."""
        return self.d_x.download()

    def destroy(self):
        """Clean up solver resources (per matrix)."""
        if self.is_setup:
            self.d_A.destroy()
            self.d_b.destroy()
            self.d_x.destroy()
            self.solver.destroy()
            self.resources.destroy()
            self.cfg.destroy()
            self.is_setup = False