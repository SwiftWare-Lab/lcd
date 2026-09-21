"""Numerical end-to-end tests for generated sparse triangular solvers."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import tril
from scipy.sparse.linalg import spsolve_triangular

from compiler.demo.gpu_test_utils import physical_cuda_status
from compiler.demo.suitesparse_test_matrix import (
    load_suitesparse_test_matrix,
)


CUDA_AVAILABLE, CUDA_SKIP_REASON = physical_cuda_status()


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_SKIP_REASON)
class SpTRSVEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing this module runs the compiler pipeline and binds the
        # generated Numba-CUDA kernels.
        with redirect_stdout(io.StringIO()):
            from compiler.demo.sptrsv import (
                simple_sptrsv_csc,
                simple_sptrsv_csr,
            )

        cls.csc_solver = staticmethod(simple_sptrsv_csc)
        cls.csr_solver = staticmethod(simple_sptrsv_csr)

    def setUp(self):
        source_matrix = load_suitesparse_test_matrix()
        self.lower_triangular = tril(source_matrix, format="csr")
        self.lower_triangular.sort_indices()
        expected_solution = np.linspace(
            1.0,
            2.0,
            self.lower_triangular.shape[0],
        )
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
