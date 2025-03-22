import numpy as np
import numba
from numba import njit
from compiler.loop_carried_compiler import MyCompiler

import numpy as np
from scipy.sparse.linalg import spsolve_triangular  # CPU sparse solver
import cupy as cp  # For GPU operations
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix  # GPU sparse matrix
# import csr_matrix as cp_csr_matrix  # GPU sparse matrix
from cupyx.scipy.sparse.linalg import spsolve_triangular as cp_spsolve  # GPU sparse solver
import time
from scipy.sparse import csc_matrix
import torch
import triton
import triton.language as tl

from numba import cuda
from numba import config

config.CUDA_ENABLE_PYNVJITLINK = 1

from bench_utils import incoming_edge
from bench_utils import compute_wavefront


REAL_TYPE = torch.float64
REAL_TYPE_TL = tl.float64

NUM_RUNS=1



DEVICE = torch.device("cuda:0")



@njit(numba.float64[:](numba.float32[:], numba.int32[:], numba.int32[:], numba.float32[:]), pipeline_class=MyCompiler)
def sptrsv_csr_numba(A, IA, JA, b):
    n = len(IA) - 1
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(IA[i], IA[i + 1]):
            x[i] -= A[j] * x[JA[j]]
        x[i] /= A[IA[i + 1] - 1]
    return x



@triton.jit
def sptrsv_sync_free(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    row = tl.program_id(axis=0)  # Get the row id based on the program grid
    is_valid_row = row < M
    while tl.load(count_dep + row, mask=is_valid_row) > 1:
        pass  # Busy-wait: Do nothing, just keep looping
    # Load start and end pointers safely (use `mask` for invalid threads)
    start = tl.load(indptr_ptr + row, mask=is_valid_row, other=0)
    print("row: ", row)
    end = tl.load(indptr_ptr + row + 1, mask=is_valid_row, other=0)
    # define an array of float64 with size 2
    y_val_row = tl.load(rhs_vec + row)  / tl.load(data_ptr + end-1) # b[i] / A[i, i]
    y_val =  tl.cast(0.0, REAL_TYPE_TL)
    for i in range(start, end-1):
        col = tl.load(indices_ptr + i)
        val = tl.load(data_ptr + i) # A[j, i]
        x_val = tl.load(y_ptr + col)
        y_val += val * x_val
        # atomid update dount_dep[col]
        tl.atomic_add(count_dep + col, -1)
    y_val = (y_val_row - y_val) / tl.load(data_ptr + end-1)
    print("val: ", y_val)
    tl.store(y_ptr + row, y_val)


@triton.jit
def sptrsv_sync_free_csc2(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    col = tl.program_id(axis=0)  # Get the row id based on the program grid
    is_valid_row = col < M
    if is_valid_row:
        while tl.atomic_add(count_dep + col, 0) != 1:
            pass   # Busy-wait: Do nothing, just keep looping
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1 , mask=is_valid_row, other=0)

        tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)  )/ tl.load(data_ptr + start))
        for j in range(start+1, end):
            val = tl.load(data_ptr + j) # A[j, i]
            x_val = tl.load(y_ptr + col)
            r_i = tl.load(indices_ptr + j)
            tl.atomic_add(y_ptr + r_i, val * x_val)
            tl.atomic_add(count_dep + r_i, -1)


@triton.jit
def sptrsv_sync_free_csc(
        data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
        M, N, nnz, BLOCK_SIZE: tl.constexpr, count_dep
):
    col = tl.program_id(axis=0)  # Get the row id based on the program grid
    is_valid_row = col < M
    if is_valid_row:
        while tl.atomic_add(count_dep + col, 0) != 1:
            #print("dep:", tl.load(count_dep + col))
            pass  # Busy-wait: Do nothing, just keep looping
        #print("dep:", col)
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
        col_length = end - start

        # diagonal = tl.load(data_ptr + start)
        # nonzero_mask = diagonal != 0
        # tl.store(y_ptr + col,
        #          ((tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / diagonal).to(REAL_TYPE_TL),
        #          mask=nonzero_mask)
        # diagonal_val = tl.load(data_ptr + start)
        # rhs_val = tl.load(rhs_vec + col)
        # tl.store(y_ptr + col, rhs_val / diagonal_val)  # Initial solve step
        y_col_sol = (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + start)
        tl.store(y_ptr + col, y_col_sol)
        #tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)) / tl.load(data_ptr + start))# b[i] / A[i, i]
        # diagonal_val = tl.load(data_ptr + start)
        # rhs_val = tl.load(rhs_vec + col)
        # solution = rhs_val / diagonal_val
        # tl.store(y_ptr + col, solution)

        #print("row: ", tl.load(y_ptr + col))
        # Thread-level processing
        thread_idx = tl.arange(0, BLOCK_SIZE)
        block_start = start + 1 + thread_idx

        mask = block_start < end
        data_col = tl.load(data_ptr + block_start, mask=mask, other=0)
        #y_col_scalar = tl.load(y_ptr + col).to(REAL_TYPE_TL)

        # Thread-specific partial contributions
        #partial_sum = data_col * tl.load(y_ptr + col)
        partial_sum = data_col * y_col_sol #tl.atomic_add(y_ptr + col, 0)
        #partial_sum = data_col * solution

        indices_col = tl.load(indices_ptr + block_start, mask=mask, other=0)

        valid_indices = mask

        # Scatter-add contributions to y_ptr
        tl.atomic_add(y_ptr + indices_col, partial_sum, mask=valid_indices)
        #tl.fence()
        # add fence to make sure all threads finish the computation

        # tl.debug_barrier()
        # # Decrement count_dep only after y_ptr is updated
        tl.atomic_add(count_dep + indices_col, -1, mask=valid_indices)
        # tl.debug_barrier()
        # for j in range(start + 1, end):
        #     r_i = tl.load(indices_ptr + j)
        #     tl.atomic_add(count_dep + r_i, -1)
            #print ("count", tl.atomic_add(count_dep + r_i, 0))



def sptrsv_sync_free_triton(data, indices, indptr, x, M, N, nnz, dep_counts, max_col_length):
    # Allocate result vector on GPU
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    d_dep_count = dep_counts.clone().detach()
    grid = (M,)  # One thread per row
    # copy wavefront to GPU
    sptrsv_sync_free_csc[grid](
        data,
        indices,
        indptr,
        x,
        y,
        M,
        N,
        nnz,
        BLOCK_SIZE= max_col_length,
        count_dep=d_dep_count)
    return y



@triton.jit
def sptrsv_kernel_csc(
    data_ptr, indices_ptr, indptr_ptr, rhs_vec, y_ptr,
    M, N, nnz, BLOCK_SIZE: tl.constexpr, wavefront
):
    row_loc = tl.program_id(axis=0)  # Get the row id based on the program grid
    #print("row_loc: ", row_loc)
    col = tl.load(wavefront + row_loc)
    is_valid_row = col < M

    if is_valid_row:
        #print("row: ", col)
    # Load start and end pointers safely (use `mask` for invalid threads)
        start = tl.load(indptr_ptr + col, mask=is_valid_row, other=0)
        end = tl.load(indptr_ptr + col + 1, mask=is_valid_row, other=0)
        # define an array of float64 with size 2
        tl.store(y_ptr + col, (tl.load(rhs_vec + col) - tl.load(y_ptr + col)  )/ tl.load(data_ptr + start))# b[i] / A[i, i]
        for j in range(start+1, end):
            val = tl.load(data_ptr + j)  # A[j, i]
            x_val = tl.load(y_ptr + col)
            r_i = tl.load(indices_ptr + j)
            tl.atomic_add(y_ptr + r_i, val * x_val)


def sptrsv_triton(data, indices, indptr, x, M, N, nnz, wave_start, wavefronts):
    # Allocate result vector on GPU
    y = torch.zeros(M, dtype=REAL_TYPE, device=DEVICE)
    #wavefront = torch.tensor(wavefronts, dtype=torch.int32, device=DEVICE)
    # for wavefront in wavefronts:
    #     # Launch Triton kernel
    #     grid = (len(wavefront),)
    #     # copy wavefront to GPU
    #     wavefront_gpu = torch.tensor(wavefront, dtype=torch.int32, device=DEVICE)
    num_waves = len(wave_start)-1
    for i in range(num_waves):
        grid_len = wave_start[i+1] - wave_start[i]
        grid = (grid_len,)
        sptrsv_kernel_csc[grid](
            data,
            indices,
            indptr,
            x,
            y,
            M,
            N,
            nnz,
            BLOCK_SIZE=128,
            wavefront=wavefronts[wave_start[i]])
    return y


@cuda.jit
def sptrsc_numba_csc_sync_free_kernel(data, indices, indptr, b, y, M, N, nnz, incom_edges):
    col = cuda.grid(1)
    #col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if col < M:
        while cuda.atomic.add(incom_edges, col, 0) != 1:
            pass
        start = indptr[col]
        end = indptr[col + 1]
        y[col] = (b[col] - y[col]) / data[start]
        for j in range(start + 1, end):
            val = data[j]
            #x_val = y[col]
            r_i = indices[j]
            cuda.atomic.add(y, r_i, val * cuda.atomic.add(y, col, 0)) # x[r_i] += val * x_val
            cuda.atomic.add(incom_edges, r_i, -1)


def sptrsc_numba_csc_sync_free(data, indices, indptr, b, M, N, nnz, in_edges):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_in_edges = cuda.to_device(in_edges) # avoid race in multiple kernel calls
    threads_per_block = 1
    blocks_per_grid = M+1 #(M + threads_per_block - 1) // threads_per_block
    sptrsc_numba_csc_sync_free_kernel[blocks_per_grid, threads_per_block](data, indices, indptr, b, d_y, M, N, nnz, d_in_edges)
    return d_y


@cuda.jit
def sptrsc_numba_csc_kernel(data, indices, indptr, b, y, M, N, nnz, wavefronts, wave_start):
    wave_loc = cuda.grid(1)
    col = wavefronts[wave_start + wave_loc]
    if col < M:
        start = indptr[col]
        end = indptr[col + 1]
        y[col] = (b[col] - y[col]) / data[start]
        #print("col: ", y[col])
        for j in range(start + 1, end):
            val = data[j]
            x_val = y[col]
            r_i = indices[j]
            cuda.atomic.add(y, r_i, val * x_val) # x[r_i] += val * x_val


def sptrsc_numba_csc(data, indices, indptr, b, M, N, nnz, wave_start, wavefronts):
    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_waves = len(wave_start) - 1
    for i in range(num_waves):
        grid_len = wave_start[i + 1] - wave_start[i]
        #print("grid_len: ", grid_len)
        threads_per_block = 1
        blocks_per_grid = grid_len #(threads_per_block - 1) // threads_per_block
        sptrsc_numba_csc_kernel[blocks_per_grid, threads_per_block](data, indices, indptr, b, d_y, M, N, nnz, wavefronts, wave_start[i])
    return d_y


# Function to generate a random sparse triangular matrix
def generate_sparse_triangular_matrix(n, density=0.2):
    """Generates a sparse lower triangular matrix with given density."""
    rand_matrix = np.random.rand(n, n) < density
    dense_matrix = np.multiply(rand_matrix, np.random.rand(n, n))
    # Make it lower triangular
    sparse_lower_tri = np.tril(dense_matrix)
    # add 100 to all diagonal elements to avoid singular matrix
    np.fill_diagonal(sparse_lower_tri, sparse_lower_tri.diagonal() + 100)
    return csc_matrix(sparse_lower_tri)


# Function to benchmark CPU solver
def cpu_solver(matrix, b):
    print("Running CPU-based solver...")
    time_arr = []
    for i in range(NUM_RUNS):
        start_time = time.time()
        x_cpu = spsolve_triangular(matrix, b)
        end_time = time.time()
        time_arr.append(end_time - start_time)
    print(f"CPU solution took: {np.median(time_arr):.6f} seconds")
    return x_cpu, np.median(time_arr)


def cusparse_solver(matrix, b):
    x_gpu = cp_spsolve(matrix, b)
    return x_gpu


# Function to benchmark GPU solver
def gpu_solver(matrix, b):
    print("\nRunning GPU-based solver...")
    # nake it csr
    #matrix_gpu_csr = cp_csr_matrix(matrix.tocsr())
    matrix_gpu_csc = cp_csc_matrix(matrix.tocsc())
    b_gpu = cp.array(b)  # Move dense vector to GPU
    time_measures = []
    for i in range(NUM_RUNS):
        start_time = time.time()
        x_gpu = cp_spsolve(matrix_gpu_csc, b_gpu)  # Solves Ax = b on GPU
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        time_measures.append(end_time - start_time)
    print(f"GPU solution took: {np.median(time_measures):.6f} seconds")
    return cp.asnumpy(x_gpu), np.median(time_measures)  # Convert result back to host


def simple_sptrsv_csc(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[i]) / L.data[L.indptr[i] ]
        for j in range(L.indptr[i]+1, L.indptr[i + 1]):
            x[L.indices[j]] += L.data[j] * x[i]
    return x


def simple_sptrsv_csc_waves(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for wave in compute_wavefront(L):
        for i in wave:
            x[i] = (b[i] - x[i]) / L.data[L.indptr[i]]
            for j in range(L.indptr[i]+1, L.indptr[i + 1]):
                x[L.indices[j]] += L.data[j] * x[i]
    return x


def simple_sptrsv_csr(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(L.indptr[i], L.indptr[i + 1]):
            x[i] -= L.data[j] * x[L.indices[j]]
        x[i] /= L.data[L.indptr[i + 1] - 1]
    return x

def simple_sptrsv_csc_syncfree(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    in_edges = incoming_edge(L)
    ready_cols = np.where(in_edges == 1)[0]

    while ready_cols.size > 0:
        i = ready_cols[0]
        # remove i from ready_cols
        ready_cols = ready_cols[1:]
        x[i] = (b[i] - x[i]) / L.data[L.indptr[i] ]
        for j in range(L.indptr[i]+1, L.indptr[i + 1]):
            x[L.indices[j]] += L.data[j] * x[i]
            in_edges[L.indices[j]] -= 1
            if in_edges[L.indices[j]] == 1:
                ready_cols = np.append(ready_cols, L.indices[j])
    return x