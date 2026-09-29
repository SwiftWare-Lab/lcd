"""Serial, unblocked dense Cholesky baseline for real SPD matrices."""

import numpy as np
from numba import njit


@njit(cache=True)
def _cholesky_dense_kernel(a):
    """Factor a validated float64 matrix; allocate L without modifying a."""
    n = a.shape[0]
    lower = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            value = a[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            if i == j:
                if value <= 0.0 or not np.isfinite(value):
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                lower[i, j] = np.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]
    return lower


def cholesky_dense_numba(a):
    """Return float64 lower-triangular L such that a is approximately L @ L.T.

    Accept real numeric array-like inputs, including noncontiguous arrays.
    The input must be finite, square, nonempty, symmetric (rtol=1e-12,
    atol=1e-14), and positive definite. Values are converted to float64;
    accepted roundoff-level asymmetry is resolved using the lower triangle.
    The input is never modified. Complex and nonnumeric inputs raise TypeError;
    invalid shape, nonfinite values, or asymmetry raise ValueError; a nonpositive
    pivot raises numpy.linalg.LinAlgError.

    This baseline uses O(n^3) work and O(n^2) output storage, with serial loops
    and no fastmath. The first invocation also pays Numba compilation/cache
    loading costs, so warm it up before timing.
    """
    return _cholesky_dense_kernel(_validate_dense_input(a))


def _validate_dense_input(a):
    """Shared contract for the baseline and optimized dense implementations."""
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("Expected a nonempty square matrix")
    if a.dtype.kind not in "iuf":
        raise TypeError("Expected a real numeric matrix")
    a = np.ascontiguousarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Matrix must contain only finite values")
    if not np.allclose(a, a.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Matrix must be symmetric")
    return a
