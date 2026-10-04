"""Independent natural-order symbolic Cholesky for symmetric sparse input.

Analysis uses only the input structure: a symmetric elimination tree followed
by two marked ancestor traversals (count, then fill). No numeric factor, dense
matrix, or external symbolic-factorization library is used.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numba import njit


@dataclass(frozen=True)
class CholeskySymbolic:
    """Analysis returned by analyze_cholesky_csc; array fields are read-only.

    indptr/indices describe L in CSC with its diagonal first in each column.
    prune_indptr/prune_indices describe L by row, with its diagonal last. This
    matches the existing left-looking numeric kernel's pruning convention.
    parent contains the symmetric elimination forest (-1 marks roots).
    input_* describe the canonical lower input pattern for safe value-only reuse.
    """

    n: int
    parent: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    prune_indptr: np.ndarray
    prune_indices: np.ndarray
    input_indptr: np.ndarray
    input_indices: np.ndarray

    @property
    def nnz(self):
        return int(self.indptr[-1])

    @property
    def column_counts(self):
        return np.diff(self.indptr)


def _prepare_csc_input(a):
    """Validate sparse A and return its canonical stored lower triangle.

    A may contain either both symmetric triangles or only its lower triangle.
    If any material upper off-diagonal value is stored, full symmetry is required
    with rtol=1e-12 and atol=1e-14. Explicit zeros remain structural entries in
    the lower triangle. Duplicates are summed in float64, columns are sorted, and
    the caller's storage is unchanged.
    """
    if not sp.issparse(a):
        raise TypeError("Expected a real SciPy sparse matrix or sparse array")
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("Expected a nonempty square sparse matrix")
    if a.dtype.kind not in "iuf":
        raise TypeError("Expected real numeric sparse values")
    if a.shape[0] > np.iinfo(np.int32).max or a.nnz > np.iinfo(np.int32).max:
        raise OverflowError("CSC Cholesky currently supports 32-bit index capacity")
    matrix = sp.csc_matrix(a.astype(np.float64, copy=True))
    matrix.check_format(full_check=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Matrix must contain only finite values")
    upper = sp.triu(matrix, k=1, format="csc")
    if np.any(upper.data != 0.0):
        difference = (matrix - matrix.T).tocoo()
        if difference.nnz:
            reference = np.asarray(matrix[difference.row, difference.col]).ravel()
            if np.any(np.abs(difference.data) > 1e-14 + 1e-12 * np.abs(reference)):
                raise ValueError("Expected full symmetric storage or a stored lower triangle")
    lower = sp.tril(matrix, format="csc")
    lower.sort_indices()
    lower.indptr = lower.indptr.astype(np.int32)
    lower.indices = lower.indices.astype(np.int32)
    return lower


@njit(cache=True)
def _symbolic_pattern(n, upper_ptr, upper_rows):
    # Tree of symmetric A, not A.T @ A. Ancestor shortcuts compress paths.
    parent = np.full(n, -1, dtype=np.int32)
    ancestor = np.full(n, -1, dtype=np.int32)
    for column in range(n):
        for p in range(upper_ptr[column], upper_ptr[column + 1]):
            node = upper_rows[p]
            while node != -1 and node < column:
                previous = ancestor[node]
                ancestor[node] = column
                if previous == -1:
                    parent[node] = column
                node = previous

    # Each row reaches earlier columns along the tree. Count each once and
    # reserve every diagonal, even if absent in A (numeric SPD checking is later).
    column_counts = np.ones(n, dtype=np.int64)
    row_counts = np.ones(n, dtype=np.int64)
    marked = np.full(n, -1, dtype=np.int32)
    for row in range(n):
        marked[row] = row
        for p in range(upper_ptr[row], upper_ptr[row + 1]):
            node = upper_rows[p]
            while marked[node] != row:
                marked[node] = row
                column_counts[node] += 1
                row_counts[row] += 1
                node = parent[node]

    nnz = np.sum(column_counts)
    if nnz > 2147483647:
        raise OverflowError("Factor exceeds 32-bit index capacity")
    indptr = np.zeros(n + 1, dtype=np.int32)
    prune_ptr = np.zeros(n + 1, dtype=np.int32)
    for column in range(n):
        indptr[column + 1] = indptr[column] + column_counts[column]
        prune_ptr[column + 1] = prune_ptr[column] + row_counts[column]
    indices = np.empty(nnz, dtype=np.int32)
    next_slot = indptr[:-1].copy()
    for column in range(n):
        indices[next_slot[column]] = column
        next_slot[column] += 1

    # Repeat reach in ascending row order: output columns are automatically sorted.
    marked[:] = -1
    for row in range(n):
        marked[row] = row
        for p in range(upper_ptr[row], upper_ptr[row + 1]):
            node = upper_rows[p]
            while marked[node] != row:
                marked[node] = row
                indices[next_slot[node]] = row
                next_slot[node] += 1
                node = parent[node]

    # Transpose the pattern into sorted dependency lists; each diagonal is last.
    prune_indices = np.empty(nnz, dtype=np.int32)
    next_slot = prune_ptr[:-1].copy()
    for column in range(n):
        for p in range(indptr[column], indptr[column + 1]):
            row = indices[p]
            prune_indices[next_slot[row]] = column
            next_slot[row] += 1
    return parent, indptr, indices, prune_ptr, prune_indices


def _analyze_prepared(lower):
    upper = lower.T.tocsc()
    arrays = _symbolic_pattern(lower.shape[0], upper.indptr, upper.indices)
    input_arrays = (lower.indptr.copy(), lower.indices.copy())
    for array in (*arrays, *input_arrays):
        array.flags.writeable = False
    return CholeskySymbolic(lower.shape[0], *arrays, *input_arrays)


def analyze_cholesky_csc(a):
    """Predict L's fill and dependencies directly from sparse A, in natural order.

    Accept full symmetric or lower-triangle real SciPy sparse matrices/arrays.
    A stored lower triangle is interpreted as a symmetric matrix. This does not
    test positive definiteness or compute any factor values. It retains explicit
    lower-triangle zeros structurally, so numeric cancellation never removes
    allocated fill. Reuse the result for new values with the same canonical
    lower-triangle pattern.
    """
    return _analyze_prepared(_prepare_csc_input(a))
