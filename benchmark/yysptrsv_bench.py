import sys
import scipy.io as scio
import numpy as np
from scipy.sparse import csc_matrix, tril
from numba import cuda
import time
import math
import cupy as cp
import sys

# sys.path.append("/home/salehm32/projects/YuenyeungSpTRSV/cmake-build-debug/")
print(sys.path)
import yysptrsv_wrapper


def matrix_warp_numpy(m, csrRowPtr, border, WARP_SIZE=32):
    """
    NumPy implementation of matrix_warp preprocessing.

    Parameters
    ----------
    m : int
        Number of rows
    csrRowPtr : np.ndarray (int32/int64)
        CSR row pointer array of size m+1
    border : float
        Threshold for average nonzeros per row
    WARP_SIZE : int
        Warp size (default 32)

    Returns
    -------
    warp_num : np.ndarray (int32)
        Preallocated array of size m+1
    len_add : int
        Valid length of warp_num (must be copied to device)
    """

    csrRowPtr = np.asarray(csrRowPtr)

    # Number of warp blocks
    n_blocks = (m + WARP_SIZE - 1) // WARP_SIZE

    # Precompute block starts and ends
    block_starts = np.arange(0, m, WARP_SIZE, dtype=np.int32)
    block_ends = np.minimum(block_starts + WARP_SIZE, m)

    # Compute elements per block (vectorized)
    element_n = csrRowPtr[block_ends] - csrRowPtr[block_starts]

    rows_per_block = block_ends - block_starts
    avg_element_n = element_n / rows_per_block

    # Condition mask
    dense_mask = avg_element_n >= border

    # Preallocate like original C++ (size m+1)
    warp_num = np.zeros(m + 1, dtype=np.int32)

    row = 0
    k = 1  # warp_num[0] already 0

    for b in range(n_blocks):
        block_size = rows_per_block[b]

        if dense_mask[b]:
            # Each row separate
            for _ in range(block_size):
                row += 1
                warp_num[k] = row
                k += 1
        else:
            # Entire block grouped
            row += block_size
            warp_num[k] = row
            k += 1

    len_add = k  # actual valid length

    return warp_num, len_add


@cuda.jit
def yySpTRSV_csr_kernel(
        d_csrRowPtr,
        d_csrColIdx,
        d_csrVal,
        d_get_value,
        m,
        nnz,
        d_b,
        d_x,
        begin,
        d_warp_num,
        Len,
        d_id_extractor
):

    # --------------------------------------------------
    # global id via atomic
    # --------------------------------------------------
    global_id = cuda.atomic.add(d_id_extractor, 0, 1)

    warp_id = global_id // WARP_SIZE

    if warp_id >= (Len - 1):
        return

    lane_id = cuda.threadIdx.x & (WARP_SIZE - 1)

    # ==================================================
    # Thread-level Syncfree SpTRSV
    # ==================================================
    if d_warp_num[warp_id + 1] > (d_warp_num[warp_id] + 1):

        row = d_warp_num[warp_id] + lane_id

        if row >= m:
            return

        left_sum = 0.0
        i = row
        j = d_csrRowPtr[i]

        while j < d_csrRowPtr[i + 1]:

            col = d_csrColIdx[j]

            while d_get_value[col] == 1:
                left_sum += d_csrVal[j] * d_x[col]
                j += 1
                if j >= d_csrRowPtr[i + 1]:
                    break
                col = d_csrColIdx[j]

            if j >= d_csrRowPtr[i + 1]:
                break

            if i == col:
                diag = d_csrVal[d_csrRowPtr[i + 1] - 1]
                xi = (d_b[i] - left_sum) / diag
                d_x[i] = xi
                cuda.threadfence()
                d_get_value[i] = 1
                j += 1

    # ==================================================
    # Warp-level Syncfree SpTRSV
    # ==================================================
    else:

        row = d_warp_num[warp_id]

        if row >= m:
            return

        sum_val = 0.0

        j = d_csrRowPtr[row] + lane_id

        while j < d_csrRowPtr[row + 1] - 1:

            col = d_csrColIdx[j]

            while d_get_value[col] != 1:
                cuda.threadfence_block()

            sum_val += d_x[col] * d_csrVal[j]

            j += WARP_SIZE

        # Warp reduction
        offset = WARP_SIZE // 2
        mask = cuda.activemask()
        while offset > 0:
            sum_val += cuda.shfl_down_sync(mask, sum_val, offset)
            offset //= 2

        # lane 0 writes result
        if lane_id == 0:
            diag = d_csrVal[d_csrRowPtr[row + 1] - 1]
            xi = (d_b[row] - sum_val) / diag
            d_x[row] = xi
            cuda.threadfence()
            d_get_value[row] = 1

# -------------------------
# Read matrix
# -------------------------
mat_file = sys.argv[1]
matrix = scio.mmread(mat_file)
A = matrix.tocsr()

m = A.shape[0]
b = A.dot(np.ones(m))

# Keep only lower triangular part
A = tril(A, format="csr")
if A.data.dtype != np.float64:
    print(f"Matrix {mat_file} is not double precision")
    A.data = A.data.astype(np.float64)


# -------------------------
# Preprocessing (matrix_warp)
# -------------------------

WARP_SIZE = 32
border = 10   # <-- choose the same value you used in C++

warp_num_host, len_add = matrix_warp_numpy(
    m=m,
    csrRowPtr=A.indptr,
    border=border,
    WARP_SIZE=WARP_SIZE
)

# Keep only valid part
warp_num_host = warp_num_host[:len_add].astype(np.int32)



# # -------------------------
# # Move everything to GPU
# # -------------------------
# d_data_csr    = cuda.to_device(A.data)
# d_indices_csr = cuda.to_device(A.indices)
# d_indptr_csr  = cuda.to_device(A.indptr)
# d_b           = cuda.to_device(b)
#
#
# # Move warp_num to device (correct size only)
# d_warp_num = cuda.to_device(warp_num_host)
# # ==========================================
# # Allocate remaining device arrays
# # ==========================================
#
#
# # Vector x
# d_x = cuda.device_array(m, dtype=A.data.dtype)
#
#
# # get_value (initialized to zero)
# d_get_value = cuda.device_array(m, dtype=np.int32)
#
# # id extractor (single int)
# d_id_extractor = cuda.device_array(1, dtype=np.int32)

# -------------------------
# Move everything to GPU (CuPy)
# -------------------------

d_data_csr    = cp.asarray(A.data)
d_indices_csr = cp.asarray(A.indices)
d_indptr_csr  = cp.asarray(A.indptr)
d_b           = cp.asarray(b)

# warp_num_host → device
d_warp_num = cp.asarray(warp_num_host)

# ==========================================
# Allocate remaining device arrays
# ==========================================

# Vector x (uninitialized, like cuda.device_array)
d_x = cp.empty(m, dtype=A.data.dtype)

# get_value (initialized to zero)
d_get_value = cp.zeros(m, dtype=np.int32)

# id extractor (single int)
d_id_extractor = cp.empty(1, dtype=np.int32)





# ==========================================
# Kernel launch configuration
# ==========================================

WARP_SIZE = 32
WARP_PER_BLOCK = 32
num_threads = WARP_PER_BLOCK * WARP_SIZE

num_blocks = math.ceil(((len_add - 1) * WARP_SIZE) / num_threads)

BENCH_REPEAT = 10
time_cuda_solve = 0.0

# ==========================================
# Benchmark loop (equivalent to C++)
# ==========================================

for _ in range(BENCH_REPEAT):

    # reset device arrays (equivalent to cudaMemset)

    d_get_value[:] = 0
    d_x[:] = 0
    d_id_extractor[:] = 0

    cuda.synchronize()

    t1 = time.perf_counter()

    yysptrsv_wrapper.YYSpTRSV(
        d_indptr_csr.data.ptr,
        d_indices_csr.data.ptr,
        d_data_csr.data.ptr,
        d_get_value.data.ptr,
        m,
        A.nnz,
        d_b.data.ptr,
        d_x.data.ptr,
        0,
        d_warp_num.data.ptr,
        len_add,
        d_id_extractor.data.ptr,
        num_threads,
        num_blocks
    )

    cuda.synchronize()

    t2 = time.perf_counter()

    time_cuda_solve += (t2 - t1) * 1000.0  # milliseconds

    # Copy x back (like cudaMemcpy)
    # x_host = d_x.copy_to_host()

time_cuda_solve /= BENCH_REPEAT

# print("Average Solve Time (ms):", time_cuda_solve)

perf = lambda ms: (A.shape[0] + 2 * A.nnz) / (ms * 1e-3) / 1e9
print("{},{},{}".format(mat_file,time_cuda_solve,perf(time_cuda_solve)))