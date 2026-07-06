"""Numerical end-to-end tests for generated sparse triangular solvers."""

import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve_triangular


def _cuda_status():
    try:
        from numba import cuda
        if cuda.is_available():
            return True, ""
        return False, "Numba CUDA is unavailable."
    except Exception as error:
        return False, f"Numba CUDA could not be imported: {error}"


CUDA_AVAILABLE, CUDA_SKIP_REASON = _cuda_status()


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_SKIP_REASON)
class SpTRSVEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing this module runs the compiler pipeline and binds the
        # generated Numba-CUDA kernels.
        from compiler.transform_tests.sptrsv import (
            simple_sptrsv_csc,
            simple_sptrsv_csr,
        )

        cls.csc_solver = staticmethod(simple_sptrsv_csc)
        cls.csr_solver = staticmethod(simple_sptrsv_csr)

    def setUp(self):
        self.lower_triangular = np.array([
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0, 1.0],
        ])
        expected_solution = np.array([1.0, 2.0, 3.0, 4.0])
        self.right_hand_side = (
            self.lower_triangular @ expected_solution
        )

    def test_csr_matches_scipy_triangular_solve(self):
        matrix = csr_matrix(self.lower_triangular)
        result = self.csr_solver(
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
            self.right_hand_side,
            matrix.shape[0],
        )
        reference = spsolve_triangular(
            matrix,
            self.right_hand_side,
            lower=True,
        )

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_csc_matches_scipy_triangular_solve(self):
        matrix = csc_matrix(self.lower_triangular)
        result = self.csc_solver(
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
            self.right_hand_side,
            matrix.shape[0],
        )
        reference = spsolve_triangular(
            matrix,
            self.right_hand_side,
            lower=True,
        )

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-10,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
