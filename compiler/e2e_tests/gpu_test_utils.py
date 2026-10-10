"""Shared helpers for end-to-end tests that require a physical CUDA GPU."""

import os

from compiler.demo.matrix import (
    DEFAULT_MATRIX,
    DEFAULT_MAX_N,
    load_matrix_market,
)


# Select the binding before physical_cuda_status imports Numba. Previously
# this happened as a side effect of importing compiler.transform_tests.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

MATRIX_ENVIRONMENT_VARIABLE = "GPU_BACKEND_E2E_MATRIX"
SIZE_ENVIRONMENT_VARIABLE = "GPU_BACKEND_E2E_MAX_N"


def load_test_matrix():
    """Load the environment-selected matrix used by physical-GPU tests."""
    matrix_path = os.environ.get(MATRIX_ENVIRONMENT_VARIABLE, DEFAULT_MATRIX)
    max_size = int(
        os.environ.get(SIZE_ENVIRONMENT_VARIABLE, DEFAULT_MAX_N)
    )
    return load_matrix_market(matrix_path, max_size)


def physical_cuda_status():
    try:
        import z3  # noqa: F401
    except Exception as error:
        return False, f"z3 is unavailable: {error}"

    try:
        from numba import config, cuda
        if config.ENABLE_CUDASIM:
            return False, "Physical-GPU test skipped under CUDA simulation."
        if not cuda.is_available():
            return False, "Numba CUDA is unavailable."
        cuda.current_context()
    except Exception as error:
        return False, f"Numba CUDA could not initialize a GPU: {error}"

    return True, ""
