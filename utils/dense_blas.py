
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
