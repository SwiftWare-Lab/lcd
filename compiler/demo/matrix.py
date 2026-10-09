"""Matrix Market input handling shared by the executable demos."""

from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_MATRIX = (
    REPOSITORY_ROOT / "data" / "ss" / "bcsstk17" / "bcsstk17.mtx"
)
DEFAULT_MAX_N = 128


def resolve_matrix_path(matrix_path=None):
    """Resolve a user path relative to the repository when appropriate."""
    path = DEFAULT_MATRIX if matrix_path is None else Path(matrix_path)
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path


def load_matrix_market(matrix_path=None, max_size=DEFAULT_MAX_N):
    """Load and regularize a square real Matrix Market input.

    The complete SuiteSparse matrices are large enough to make conservative
    inspectors expensive. By default, demos use a 128-by-128 leading principal
    sample. Setting ``max_size=0`` disables this limit.

    Values are normalized and the diagonal is made strictly dominant. This
    preserves the real sparse off-diagonal pattern while avoiding zero pivots
    and conditioning failures unrelated to generated-code correctness.
    """
    if max_size < 0:
        raise ValueError("max_size must be nonnegative.")
    path = resolve_matrix_path(matrix_path)
    if not path.is_file():
        raise FileNotFoundError(f"Matrix Market input not found: {path}")

    matrix = mmread(path)
    if not issparse(matrix):
        raise ValueError(f"Expected a sparse Matrix Market input: {path}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Matrix Market input must be square, got {matrix.shape}."
        )
    if np.iscomplexobj(matrix.data):
        raise ValueError("Complex Matrix Market inputs are not supported yet.")

    matrix = matrix.tocsr().astype(np.float64)
    if max_size:
        size = min(max_size, matrix.shape[0])
        matrix = matrix[:size, :size].copy()
    else:
        matrix = matrix.copy()

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not matrix.nnz:
        raise ValueError(f"Matrix Market sample is empty: {path}")

    scale = float(np.max(np.abs(matrix.data)))
    if scale:
        matrix.data /= scale
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    off_diagonal_sum = np.asarray(
        np.abs(matrix).sum(axis=1)
    ).ravel()
    matrix.setdiag(off_diagonal_sum + 1.0)
    matrix.sum_duplicates()
    # setdiag may append entries while leaving SciPy's cached sorted-indices
    # flag set.  Clear it so sort_indices really restores the property that
    # generated segmented binary searches rely on.
    matrix.has_sorted_indices = False
    matrix.sort_indices()
    return matrix
