"""Blocked, serial dense Cholesky using contiguous upper-triangle updates."""

import operator

import numpy as np
from numba import njit

from benchmark.cholesky_dense import _validate_dense_input

DEFAULT_BLOCK_SIZE = 64


@njit(inline="always")
def _subtract_scaled_row(destination, source, multiplier):
    """Contiguous row views expose independent elements to LLVM vectorization."""
    for j in range(destination.size):
        destination[j] -= multiplier * source[j]


@njit(inline="always")
def _subtract_four_rows(destination, row0, row1, row2, row3, m0, m1, m2, m3):
    """Reuse each destination value for four updates, preserving their order."""
    for j in range(destination.size):
        value = destination[j]
        value -= m0 * row0[j]
        value -= m1 * row1[j]
        value -= m2 * row2[j]
        value -= m3 * row3[j]
        destination[j] = value


@njit(cache=True)
def _cholesky_dense_blocked_kernel(a, block_size):
    """Factor a prepared float64 matrix; block_size must be positive.

    Work on U = L.T so that the innermost update walks contiguous rows.
    Factor each diagonal panel and its rows to the right, then apply the
    panel's contributions to cache-sized tiles in the trailing upper triangle.
    Only scalar Numba loops are used: no BLAS, fastmath, or parallel dispatch.
    """
    n = a.shape[0]
    # Transposition makes the input's lower triangle authoritative, just as in
    # the baseline, even when the validator accepts roundoff-level asymmetry.
    work = a.T.copy()
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        for k in range(start, end):
            pivot = work[k, k]
            if pivot <= 0.0 or not np.isfinite(pivot):
                raise np.linalg.LinAlgError("Matrix is not positive definite")
            diagonal = np.sqrt(pivot)
            work[k, k] = diagonal
            for j in range(k + 1, n):
                work[k, j] /= diagonal
            # Updates within this panel also perform the off-diagonal solve.
            for i in range(k + 1, end):
                multiplier = work[k, i]
                _subtract_scaled_row(work[i, i:n], work[k, i:n], multiplier)

        # Keep an output tile hot while accumulating all columns of the panel.
        for row_start in range(end, n, block_size):
            row_end = min(row_start + block_size, n)
            for col_start in range(row_start, n, block_size):
                col_end = min(col_start + block_size, n)
                for i in range(row_start, row_end):
                    first_col = max(i, col_start)
                    four_end = start + (end - start) // 4 * 4
                    destination = work[i, first_col:col_end]
                    for k in range(start, four_end, 4):
                        _subtract_four_rows(
                            destination, work[k, first_col:col_end],
                            work[k + 1, first_col:col_end], work[k + 2, first_col:col_end],
                            work[k + 3, first_col:col_end], work[k, i], work[k + 1, i],
                            work[k + 2, i], work[k + 3, i])
                    for k in range(four_end, end):
                        multiplier = work[k, i]
                        _subtract_scaled_row(destination, work[k, first_col:col_end], multiplier)

    # Return a conventional lower-triangular factor in the same allocation.
    for i in range(n):
        for j in range(i + 1, n):
            work[j, i] = work[i, j]
            work[i, j] = 0.0
    return work


def cholesky_dense_numba_optimized(a, block_size=DEFAULT_BLOCK_SIZE):
    """Return L with the same input/error contract as cholesky_dense_numba.

    This is a serial, blocked factorization with O(n^3) work and O(n^2) storage.
    The input is preserved. block_size must be a positive integer (not bool);
    sizes larger than the matrix are allowed and produce one panel. Partial
    panels are supported. Warm up the function before timing it.
    """
    if isinstance(block_size, (bool, np.bool_)):
        raise TypeError("block_size must be a positive integer")
    try:
        block_size = operator.index(block_size)
    except TypeError:
        raise TypeError("block_size must be a positive integer") from None
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    a = _validate_dense_input(a)
    return _cholesky_dense_blocked_kernel(a, min(block_size, a.shape[0]))
