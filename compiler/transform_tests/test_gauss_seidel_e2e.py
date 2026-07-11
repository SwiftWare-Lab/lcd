"""Numerical end-to-end tests for generated Gauss-Seidel sweeps."""

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


def _cpu_gauss_seidel_csr_sweep(data, indptr, indices, b, x):
    reference = x.copy()
    for i in range(len(indptr) - 1):
        row_sum = 0.0
        diag = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            if i != indices[j]:
                row_sum += data[j] * reference[indices[j]]
            else:
                diag = data[j]
        reference[i] = (b[i] - row_sum) / diag
    return reference


def _cpu_gauss_seidel_diag_csr_sweep(data, indptr, indices, diag, b, x):
    reference = x.copy()
    for i in range(len(indptr) - 1):
        row_sum = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            if i != indices[j]:
                row_sum += data[j] * reference[indices[j]]
        reference[i] = (b[i] - row_sum) / diag[i]
    return reference


CUDA_AVAILABLE, CUDA_SKIP_REASON = _cuda_status()


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_SKIP_REASON)
class GaussSeidelEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing this module runs the compiler pipeline and binds the
        # generated Numba-CUDA kernels.
        from compiler.transform_tests.gauss_seidel import (
            gauss_seidel_csr_sweep,
            gauss_seidel_diag_csr_sweep,
        )

        cls.csr_sweep = staticmethod(gauss_seidel_csr_sweep)
        cls.diag_csr_sweep = staticmethod(gauss_seidel_diag_csr_sweep)

    def setUp(self):
        self.matrix = np.array([
            [4.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [2.0, -1.0, 5.0, 0.0],
            [0.0, 2.0, 1.0, 6.0],
        ])
        self.b = np.array([8.0, 7.0, 4.0, 3.0])
        self.initial_x = np.array([0.25, -0.5, 0.75, 1.0])

    def test_csr_sweep_matches_cpu_reference(self):
        matrix = csr_matrix(self.matrix)
        result = self.csr_sweep(
            matrix.data,
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            self.b,
            self.initial_x.copy(),
            matrix.shape[0],
        )
        reference = _cpu_gauss_seidel_csr_sweep(
            matrix.data,
            matrix.indptr,
            matrix.indices,
            self.b,
            self.initial_x,
        )

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_diag_csr_sweep_matches_cpu_reference(self):
        matrix = csr_matrix(self.matrix)
        diagonal = matrix.diagonal()
        offdiag_matrix = matrix.copy()
        offdiag_matrix.setdiag(0.0)
        offdiag_matrix.eliminate_zeros()

        result = self.diag_csr_sweep(
            offdiag_matrix.data,
            offdiag_matrix.indptr.astype(np.int32),
            offdiag_matrix.indices.astype(np.int32),
            diagonal,
            self.b,
            self.initial_x.copy(),
            matrix.shape[0],
        )
        reference = _cpu_gauss_seidel_diag_csr_sweep(
            offdiag_matrix.data,
            offdiag_matrix.indptr,
            offdiag_matrix.indices,
            diagonal,
            self.b,
            self.initial_x,
        )

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-10,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
