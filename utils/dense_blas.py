
import triton
import triton.language as tl
import numpy as np
from numba import cuda, njit, prange

def mv_prod(A, x):
    return A @ x

def lsolve(L, x, b):
    for i in range(L.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x





def lsolve_dense_col_order(ldm, col, M, rhs):
    """
    Solve a dense lower triangular system in column order with synchronization.

    Parameters:
    ldm (int): offset to the next col.
    col (int): Number of columns.
    M (numpy.ndarray): The matrix M.
    rhs (numpy.ndarray): The right-hand side vector.
    """
    for i in range(col):
        rhs[i] /= M[i * ldm + i]
        for j in range(i + 1, col):
            tmp = M[i * ldm + j] * rhs[i]
            rhs[j] -= tmp

@njit(parallel=True, fastmath=True)
def lsolve_dense_col_order_numba(ldm, col, M, rhs):
    """
    Solve a dense lower triangular system in column order with synchronization.

    Parameters:
    ldm (int): offset to the next col.
    col (int): Number of columns.
    M (numpy.ndarray): The matrix M.
    rhs (numpy.ndarray): The right-hand side vector.
    """
    for i in range(col):
        rhs[i] /= M[i * ldm + i]
        for j in range(i + 1, col):
            tmp = M[i * ldm + j] * rhs[i]
            rhs[j] -= tmp


import numpy as np
from numba import njit

@njit(fastmath=True, cache=True )
def lsolve_BLAS_python(dim, num_col, M, b):
    """
    Python implementation of lsolve_BLAS from C.

    Parameters:
    dim (int): Dimension of the matrix (number of rows and columns).
    num_col (int): Number of columns to process.
    M (numpy.ndarray): The lower triangular matrix M (flattened).
    b (numpy.ndarray): The right-hand side vector b.
    """
    col = 0
    M0 = 0  # Index to M

    while col <= num_col - 8:
        Mi0 = M0
        Mi1 = Mi0 + dim + 1
        Mi2 = Mi1 + dim + 1
        Mi3 = Mi2 + dim + 1
        Mi4 = Mi3 + dim + 1
        Mi5 = Mi4 + dim + 1
        Mi6 = Mi5 + dim + 1
        Mi7 = Mi6 + dim + 1

        x0 = b[col] / M[Mi0]
        Mi0 += 1
        x1 = (b[col + 1] - x0 * M[Mi0]) / M[Mi1]
        Mi0 += 1
        Mi1 += 1
        x2 = (b[col + 2] - x0 * M[Mi0] - x1 * M[Mi1]) / M[Mi2]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        x3 = (b[col + 3] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2]) / M[Mi3]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        Mi3 += 1
        x4 = (b[col + 4] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2] - x3 * M[Mi3]) / M[Mi4]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        Mi3 += 1
        Mi4 += 1
        x5 = (b[col + 5] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2] - x3 * M[Mi3] - x4 * M[Mi4]) / M[Mi5]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        Mi3 += 1
        Mi4 += 1
        Mi5 += 1
        x6 = (b[col + 6] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2] - x3 * M[Mi3] - x4 * M[Mi4] - x5 * M[Mi5]) / M[Mi6]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        Mi3 += 1
        Mi4 += 1
        Mi5 += 1
        Mi6 += 1
        x7 = (b[col + 7] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2] - x3 * M[Mi3] - x4 * M[Mi4] - x5 * M[Mi5] - x6 * M[Mi6]) / M[Mi7]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        Mi3 += 1
        Mi4 += 1
        Mi5 += 1
        Mi6 += 1
        Mi7 += 1

        b[col] = x0
        col += 1
        b[col] = x1
        col += 1
        b[col] = x2
        col += 1
        b[col] = x3
        col += 1
        b[col] = x4
        col += 1
        b[col] = x5
        col += 1
        b[col] = x6
        col += 1
        b[col] = x7
        col += 1

        k = col
        while k < num_col:
            b[k] = b[k] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2] - x3 * M[Mi3] - x4 * M[Mi4] - x5 * M[Mi5] - x6 * M[Mi6] - x7 * M[Mi7]
            Mi0 += 1
            Mi1 += 1
            Mi2 += 1
            Mi3 += 1
            Mi4 += 1
            Mi5 += 1
            Mi6 += 1
            Mi7 += 1
            k += 1

        M0 += 8 * dim + 8

    while col <= num_col - 4:
        Mi0 = M0
        Mi1 = Mi0 + dim + 1
        Mi2 = Mi1 + dim + 1
        Mi3 = Mi2 + dim + 1

        x0 = b[col] / M[Mi0]
        Mi0 += 1
        x1 = (b[col + 1] - x0 * M[Mi0]) / M[Mi1]
        Mi0 += 1
        Mi1 += 1
        x2 = (b[col + 2] - x0 * M[Mi0] - x1 * M[Mi1]) / M[Mi2]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        x3 = (b[col + 3] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2]) / M[Mi3]
        Mi0 += 1
        Mi1 += 1
        Mi2 += 1
        Mi3 += 1

        b[col] = x0
        col += 1
        b[col] = x1
        col += 1
        b[col] = x2
        col += 1
        b[col] = x3
        col += 1

        k = col
        while k < num_col:
            b[k] = b[k] - x0 * M[Mi0] - x1 * M[Mi1] - x2 * M[Mi2] - x3 * M[Mi3]
            Mi0 += 1
            Mi1 += 1
            Mi2 += 1
            Mi3 += 1
            k += 1

        M0 += 4 * dim + 4

    if col <= num_col - 2:
        Mi0 = M0
        Mi1 = Mi0 + dim + 1

        x0 = b[col] / M[Mi0]
        Mi0 += 1
        x1 = (b[col + 1] - x0 * M[Mi0]) / M[Mi1]
        Mi0 += 1
        Mi1 += 1

        b[col] = x0
        col += 1
        b[col] = x1
        col += 1

        k = col
        while k < num_col:
            b[k] = b[k] - x0 * M[Mi0] - x1 * M[Mi1]
            Mi0 += 1
            Mi1 += 1
            k += 1

        M0 += 2 * dim + 2

    if col == num_col - 1:
        Mi0 = M0
        x0 = b[col] / M[Mi0]
        b[col] = x0


def dense_matvec_col_order(ldm, row, col, M, x):
    """
    Compute a dense matrix-vector product in column order with synchronization.

    Parameters:
    ldm (int): offset to the next col.
    col (int): Number of columns.
    M (numpy.ndarray): The matrix M.
    x (numpy.ndarray): The vector x.
    """
    y = np.zeros(row)
    for i in range(col):
        for j in range(row):
            y[j] += M[i * ldm + j] * x[i]
    return y


@triton.jit
def dense_matvec_col_order_kernel(M, x, y, ldm, row, col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row

    y_block = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(col):
        M_block = tl.load(M + i * ldm + offsets, mask=mask)
        x_val = tl.load(x + i)
        y_block += M_block * x_val
    tl.store(y + offsets, y_block, mask=mask)



@njit(parallel=True, fastmath=True)
def dense_matvec_col_order_numba(ldm, row, col, M, x, y):
    """
    Compute a dense matrix-vector product in column order with synchronization.

    Parameters:
    ldm (int): offset to the next col.
    col (int): Number of columns.
    M (numpy.ndarray): The matrix M.
    x (numpy.ndarray): The vector x.
    """
    #y = np.zeros(row)
    for i in range(col):
        for j in range(row):
            y[j] += M[i * ldm + j] * x[i]
    #return y

@njit(parallel=True, fastmath=True)
def dense_matvec_col_order_numba_unrolled_ij(ldm, row, col, M, x, y):
    """
    Compute a dense matrix-vector product in column order with synchronization and loop unrolling (i factor 2, j factor 4).

    Parameters:
    ldm (int): offset to the next col.
    col (int): Number of columns.
    M (numpy.ndarray): The matrix M.
    x (numpy.ndarray): The vector x.
    y (numpy.ndarray): The output vector y.
    """
    for i in range(0, col - 1, 2):
        for j in range(0, row - 3, 4):
            x_i, x_ip1 = x[i], x[i + 1]
            y[j] += M[i * ldm + j] * x_i
            y[j + 1] += M[i * ldm + j + 1] * x_i
            y[j + 2] += M[i * ldm + j + 2] * x_i
            y[j + 3] += M[i * ldm + j + 3] * x_i

            y[j] += M[(i + 1) * ldm + j] * x_ip1
            y[j + 1] += M[(i + 1) * ldm + j + 1] * x_ip1
            y[j + 2] += M[(i + 1) * ldm + j + 2] * x_ip1
            y[j + 3] += M[(i + 1) * ldm + j + 3] * x_ip1

        # Handle remaining rows (j)
        for k in range(row % 4):
            y[row - (row % 4) + k] += M[i * ldm + row - (row % 4) + k] * x[
                i]
            y[row - (row % 4) + k] += M[(i + 1) * ldm + row - (
                        row % 4) + k] * x[i + 1]

    # Handle remaining columns (i)
    if col % 2 != 0:
        for j in range(0, row - 3, 4):
            y[j] += M[(col - 1) * ldm + j] * x[col - 1]
            y[j + 1] += M[(col - 1) * ldm + j + 1] * x[col - 1]
            y[j + 2] += M[(col - 1) * ldm + j + 2] * x[col - 1]
            y[j + 3] += M[(col - 1) * ldm + j + 3] * x[col - 1]
        for k in range(row % 4):
            y[row - (row % 4) + k] += M[(col - 1) * ldm + row - (
                        row % 4) + k] * x[col - 1]


@njit(fastmath=True)
def mattvec_BLAS_python(dim, nrow, ncol, M, x, b):
    """
    Python implementation of mattvec_BLAS from C.

    Parameters:
    dim (int): Dimension of the matrix (number of rows).
    nrow (int): Number of rows.
    ncol (int): Number of columns.
    M (numpy.ndarray): The matrix M (flattened, column-major).
    x (numpy.ndarray): The vector x.
    b (numpy.ndarray): The result vector b (initialized to zeros).
    """
    M0 = 0  # Index to M
    col = 0
    k = 0

    while col <= ncol - 8:
        Mi0 = M0
        Mi1 = Mi0 + dim
        Mi2 = Mi1 + dim
        Mi3 = Mi2 + dim
        Mi4 = Mi3 + dim
        Mi5 = Mi4 + dim
        Mi6 = Mi5 + dim
        Mi7 = Mi6 + dim

        for k in range(nrow):
            b[col] += M[Mi0] * x[k]
            Mi0 += 1
            b[col + 1] += M[Mi1] * x[k]
            Mi1 += 1
            b[col + 2] += M[Mi2] * x[k]
            Mi2 += 1
            b[col + 3] += M[Mi3] * x[k]
            Mi3 += 1
            b[col + 4] += M[Mi4] * x[k]
            Mi4 += 1
            b[col + 5] += M[Mi5] * x[k]
            Mi5 += 1
            b[col + 6] += M[Mi6] * x[k]
            Mi6 += 1
            b[col + 7] += M[Mi7] * x[k]
            Mi7 += 1

        M0 += 8 * dim
        col += 8

    while col <= ncol - 4:
        Mi0 = M0
        Mi1 = Mi0 + dim
        Mi2 = Mi1 + dim
        Mi3 = Mi2 + dim

        for k in range(nrow):
            b[col] += M[Mi0] * x[k]
            Mi0 += 1
            b[col + 1] += M[Mi1] * x[k]
            Mi1 += 1
            b[col + 2] += M[Mi2] * x[k]
            Mi2 += 1
            b[col + 3] += M[Mi3] * x[k]
            Mi3 += 1

        M0 += 4 * dim
        col += 4

    while col < ncol:
        Mi0 = M0
        for k in range(nrow):
            tmp = M[Mi0] * x[k]
            b[col] += tmp
            Mi0 += 1
        M0 += dim
        col += 1




@njit(fastmath=True)
def matvec_BLAS_python(dim, nrow, ncol, M, x, b):
    """
    Python implementation of matvec_BLAS from C.

    Parameters:
    dim (int): Dimension of the matrix (number of rows).
    nrow (int): Number of rows.
    ncol (int): Number of columns.
    M (numpy.ndarray): The matrix M (flattened, column-major).
    x (numpy.ndarray): The vector x.
    b (numpy.ndarray): The result vector b (initialized to zeros).
    """
    M0 = 0  # Index to M
    col = 0
    k = 0

    while col <= ncol - 8:
        Mi0 = M0
        Mi1 = Mi0 + dim
        Mi2 = Mi1 + dim
        Mi3 = Mi2 + dim
        Mi4 = Mi3 + dim
        Mi5 = Mi4 + dim
        Mi6 = Mi5 + dim
        Mi7 = Mi6 + dim

        x0 = x[col]
        col += 1
        x1 = x[col]
        col += 1
        x2 = x[col]
        col += 1
        x3 = x[col]
        col += 1
        x4 = x[col]
        col += 1
        x5 = x[col]
        col += 1
        x6 = x[col]
        col += 1
        x7 = x[col]
        col += 1

        for k in range(nrow):
            b[k] += x0 * M[Mi0] + x1 * M[Mi1] + x2 * M[Mi2] + x3 * M[Mi3] + \
                    x4 * M[Mi4] + x5 * M[Mi5] + x6 * M[Mi6] + x7 * M[Mi7]
            Mi0 += 1
            Mi1 += 1
            Mi2 += 1
            Mi3 += 1
            Mi4 += 1
            Mi5 += 1
            Mi6 += 1
            Mi7 += 1

        M0 += 8 * dim

    while col <= ncol - 4:
        Mi0 = M0
        Mi1 = Mi0 + dim
        Mi2 = Mi1 + dim
        Mi3 = Mi2 + dim

        x0 = x[col]
        col += 1
        x1 = x[col]
        col += 1
        x2 = x[col]
        col += 1
        x3 = x[col]
        col += 1

        for k in range(nrow):
            b[k] += x0 * M[Mi0] + x1 * M[Mi1] + x2 * M[Mi2] + x3 * M[Mi3]
            Mi0 += 1
            Mi1 += 1
            Mi2 += 1
            Mi3 += 1

        M0 += 4 * dim

    while col < ncol:
        Mi0 = M0
        x0 = x[col]
        col += 1

        for k in range(nrow):
            b[k] += x0 * M[Mi0]
            Mi0 += 1

        M0 += dim