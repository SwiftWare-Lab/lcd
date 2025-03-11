
import numpy as np

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


def dense_matvec_col_order(A, x):
    """
    Perform dense matrix-vector multiplication in column order.

    Parameters:
    A (numpy.ndarray): The dense matrix.
    x (numpy.ndarray): The vector to be multiplied.

    Returns:
    numpy.ndarray: The result of the matrix-vector multiplication.
    """
    n = A.shape[1]
    result = np.zeros_like(x)
    for j in range(n):
        result += A[:, j] * x[j]
    return result