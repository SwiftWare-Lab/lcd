"""Numerical end-to-end tests for generated Gauss-Seidel CUDA solvers."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from scipy.sparse import tril

from compiler.demo.gpu_test_utils import physical_cuda_status
from compiler.demo.suitesparse_test_matrix import (
    load_suitesparse_test_matrix,
)


PIPELINE_AVAILABLE, PIPELINE_SKIP_REASON = physical_cuda_status()


def _simple_gauss_seidel_reference(Ap, Ai, Ax, D, b, x, n, max_iter):
    x = x.copy()
    for sweep in range(max_iter):
        for i in range(n):
            x[i] = b[i]
            for j in range(Ap[i], Ap[i + 1]):
                x[i] -= Ax[j] * x[Ai[j]]
            x[i] /= D[i]
    return x


def _full_gauss_seidel_reference(Ap, Ai, Ax, b, x, n, max_iter):
    x = x.copy()
    for sweep in range(max_iter):
        for i in range(n):
            sigma = 0.0
            diag = 1.0
            for j in range(Ap[i], Ap[i + 1]):
                if Ai[j] == i:
                    diag = Ax[j]
                else:
                    sigma += Ax[j] * x[Ai[j]]
            x[i] = (b[i] - sigma) / diag
    return x


@unittest.skipUnless(PIPELINE_AVAILABLE, PIPELINE_SKIP_REASON)
class GaussSeidelEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing this module runs the decorator pipeline and binds the
        # generated CUDA kernels.
        with redirect_stdout(io.StringIO()):
            from compiler.demo.gauss_seidel import (
                full_gauss_seidel_csr,
                simple_gauss_seidel_csr,
            )

        cls.full_gauss_seidel_csr = staticmethod(full_gauss_seidel_csr)
        cls.simple_gauss_seidel_csr = staticmethod(simple_gauss_seidel_csr)

    def setUp(self):
        source_matrix = load_suitesparse_test_matrix()
        full_lower = tril(source_matrix, format="csr")
        full_lower.sort_indices()
        strict_lower = tril(source_matrix, k=-1, format="csr")
        strict_lower.sort_indices()

        self.n = source_matrix.shape[0]
        self.max_iter = 2
        expected = np.linspace(1.0, 2.0, self.n)
        self.b = np.asarray(full_lower @ expected).ravel()
        self.x0 = np.zeros(self.n, dtype=np.float64)

        # Strictly lower CSR part for simple_gauss_seidel_csr; D stores the
        # diagonal separately.
        self.simple_Ap = strict_lower.indptr.astype(np.int32)
        self.simple_Ai = strict_lower.indices.astype(np.int32)
        self.simple_Ax = strict_lower.data.copy()
        self.D = source_matrix.diagonal().copy()

        # Full lower-triangular CSR matrix, including the diagonal.
        self.full_Ap = full_lower.indptr.astype(np.int32)
        self.full_Ai = full_lower.indices.astype(np.int32)
        self.full_Ax = full_lower.data.copy()

    def test_simple_csr_matches_sequential_reference(self):
        result = self.simple_gauss_seidel_csr(
            self.simple_Ap,
            self.simple_Ai,
            self.simple_Ax,
            self.D,
            self.b,
            self.x0.copy(),
            self.n,
            self.max_iter,
        )
        reference = _simple_gauss_seidel_reference(
            self.simple_Ap,
            self.simple_Ai,
            self.simple_Ax,
            self.D,
            self.b,
            self.x0,
            self.n,
            self.max_iter,
        )

        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-12)

    def test_full_csr_matches_sequential_reference(self):
        result = self.full_gauss_seidel_csr(
            self.full_Ap,
            self.full_Ai,
            self.full_Ax,
            self.b,
            self.x0.copy(),
            self.n,
            self.max_iter,
        )
        reference = _full_gauss_seidel_reference(
            self.full_Ap,
            self.full_Ai,
            self.full_Ax,
            self.b,
            self.x0,
            self.n,
            self.max_iter,
        )

        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
