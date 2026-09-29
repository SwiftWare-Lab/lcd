"""Standalone sparse CSC Cholesky: independent symbolic + existing numeric kernel."""

import numpy as np
import scipy.sparse as sp

from benchmark.cholesky import cholesky_left_looking_numba
from utils.cholesky_symbolic import (
    CholeskySymbolic, _analyze_prepared, _prepare_csc_input, analyze_cholesky_csc,
)

__all__ = ["CholeskySymbolic", "analyze_cholesky_csc", "cholesky_csc"]


def _factor_prepared(lower, symbolic):
    values = np.zeros(symbolic.nnz, dtype=np.float64)
    success = cholesky_left_looking_numba(
        symbolic.n, lower.indptr, lower.indices, lower.data,
        symbolic.indptr, symbolic.indices, values,
        symbolic.prune_indptr, symbolic.prune_indices,
    )
    if not success or not np.isfinite(values).all():
        raise np.linalg.LinAlgError("Matrix is not positive definite or factor is nonfinite")
    # Separate index ownership keeps edits to a returned factor from corrupting
    # reusable symbolic analysis. Allocation/copies belong to numeric API timing.
    return sp.csc_matrix((values, symbolic.indices.copy(), symbolic.indptr.copy()),
                         shape=(symbolic.n, symbolic.n))


def cholesky_csc(a, symbolic=None):
    """Return sparse lower L with A approximately L @ L.T, in natural order.

    A may use full symmetric sparse storage or store only its lower triangle.
    Lower-triangle storage is interpreted as symmetric. Values are converted to
    float64; duplicates are summed and indices sorted without modifying A. No
    dense conversion, CHOLMOD, ordering library, or reference factor is used.

    Supply analyze_cholesky_csc(A)'s result to reuse analysis for changed values.
    The canonical lower input pattern must match exactly, including explicit zeros.
    Changed patterns raise ValueError; nonpositive pivots raise LinAlgError.
    """
    lower = _prepare_csc_input(a)
    if symbolic is None:
        symbolic = _analyze_prepared(lower)
    elif not isinstance(symbolic, CholeskySymbolic):
        raise TypeError("symbolic must be returned by analyze_cholesky_csc")
    elif (symbolic.n != lower.shape[0]
          or not np.array_equal(symbolic.input_indptr, lower.indptr)
          or not np.array_equal(symbolic.input_indices, lower.indices)):
        raise ValueError("Matrix pattern differs from the reusable symbolic analysis")
    return _factor_prepared(lower, symbolic)
