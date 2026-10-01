"""Numerical end-to-end tests for generated ILU0 row-dependency kernels."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix


def _cuda_status():
    try:
        from numba import cuda
        if cuda.is_available():
            return True, ""
        return False, "Numba CUDA is unavailable."
    except Exception as error:
        return False, f"Numba CUDA could not be imported: {error}"


def _cpu_ilu0_csr_row_dependencies(indptr, indices, row_done):
    reference = row_done.copy()
    for i in range(len(indptr) - 1):
        row_accum = 0.0
        scale = 0.5 / (indptr[i + 1] - indptr[i] + 1)
        for k_ptr in range(indptr[i], indptr[i + 1]):
            if indices[k_ptr] < i:
                row_accum += scale * reference[indices[k_ptr]]
        reference[i] = row_accum + 1.0
    return reference


CUDA_AVAILABLE, CUDA_SKIP_REASON = _cuda_status()


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_SKIP_REASON)
class ILU0EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing this module runs the compiler pipeline and binds the
        # generated Numba-CUDA kernels.
        from compiler.transform_tests.ilu0 import (
            ilu0_csr_row_dependencies,
            ilu0_csr_row_dependencies_sync_free,
        )

        cls.row_dependencies = staticmethod(ilu0_csr_row_dependencies)
        cls.row_dependencies_sync_free = staticmethod(
            ilu0_csr_row_dependencies_sync_free
        )

    def test_csr_row_dependencies_match_cpu_reference(self):
        matrix = csr_matrix(np.array([
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0, 0.0],
            [2.0, -1.0, 5.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 6.0, 0.0],
            [3.0, 0.0, -2.0, 1.0, 7.0],
        ]))
        initial_row_done = np.array([0.25, -0.5, 0.75, 1.0, -1.25])

        result = self.row_dependencies(
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            initial_row_done.copy(),
            matrix.shape[0],
        )
        reference = _cpu_ilu0_csr_row_dependencies(
            matrix.indptr,
            matrix.indices,
            initial_row_done,
        )

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-10,
            atol=1e-12,
        )

        sync_free_result = self.row_dependencies_sync_free(
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            initial_row_done.copy(),
            matrix.shape[0],
        )
        np.testing.assert_allclose(
            sync_free_result,
            reference,
            rtol=1e-10,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
