"""Load a configurable SuiteSparse matrix for numerical GPU demos."""

import os
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_MATRIX = (
    REPOSITORY_ROOT / "data" / "ss" / "bcsstk17" / "bcsstk17.mtx"
)
MATRIX_ENVIRONMENT_VARIABLE = "GPU_BACKEND_E2E_MATRIX"
SIZE_ENVIRONMENT_VARIABLE = "GPU_BACKEND_E2E_MAX_N"
DEFAULT_MAX_N = 128


def configured_matrix_path():
    configured = os.environ.get(MATRIX_ENVIRONMENT_VARIABLE)
    if configured is None:
        return DEFAULT_MATRIX
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path


def configured_max_size():
    value = int(os.environ.get(SIZE_ENVIRONMENT_VARIABLE, DEFAULT_MAX_N))
    if value < 0:
        raise ValueError(f"{SIZE_ENVIRONMENT_VARIABLE} must be nonnegative.")
    return value


def load_suitesparse_test_matrix():
    """Return a stable square CSR sample while retaining the input pattern.

    The complete SuiteSparse matrices are large enough to make conservative
    inspectors expensive. By default, tests use a 128-by-128 leading principal
    sample. Setting ``GPU_BACKEND_E2E_MAX_N=0`` disables this limit.

    Values are normalized and the diagonal is made strictly dominant. This
    preserves the real sparse off-diagonal pattern while avoiding zero pivots
    and conditioning failures unrelated to generated-code correctness.
    """
    path = configured_matrix_path()
    if not path.is_file():
        raise FileNotFoundError(f"SuiteSparse test matrix not found: {path}")

    matrix = mmread(path)
    if not issparse(matrix):
        raise ValueError(f"Expected a sparse Matrix Market input: {path}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"SuiteSparse test matrix must be square, got {matrix.shape}."
        )
    if np.iscomplexobj(matrix.data):
        raise ValueError("Complex SuiteSparse inputs are not supported yet.")

    matrix = matrix.tocsr().astype(np.float64)
    max_size = configured_max_size()
    if max_size:
        size = min(max_size, matrix.shape[0])
        matrix = matrix[:size, :size].copy()
    else:
        matrix = matrix.copy()

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not matrix.nnz:
        raise ValueError(f"SuiteSparse test sample is empty: {path}")

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
    matrix.sort_indices()
    return matrix
