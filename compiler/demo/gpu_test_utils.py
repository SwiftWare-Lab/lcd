"""Shared helpers for demo tests that require a physical CUDA GPU."""

import os


# Select the binding before physical_cuda_status imports Numba. Previously
# this happened as a side effect of importing compiler.transform_tests.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")


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
