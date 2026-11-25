"""Numerical end-to-end tests for generated Gauss-Seidel CUDA solvers."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from scipy.sparse import tril

from compiler.e2e_tests.gpu_test_utils import (
    load_test_matrix,
    physical_cuda_status,
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
        source_matrix = load_test_matrix()
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

    def test_nonsymmetric_pattern_matches_sequential_reference(self):
        from scipy.sparse import coo_matrix

        from compiler.decorator import parallel

        # Rows 1..m-1 form a dependence chain, while rows m..2m-1 have no
        # lower entries and run first. Each chain row i also reads the old
        # value x[m + i], which only an anti dependence protects. Rows
        # m..2m-1 produce the same value in every sweep, so one sweep is
        # needed for the old and new values to differ.
        m = 64
        n = 2 * m
        chain = np.arange(1, m)
        rows = np.concatenate([chain, chain])
        columns = np.concatenate([chain - 1, chain + m])
        matrix = coo_matrix(
            (np.full(rows.size, 0.5), (rows, columns)),
            shape=(n, n),
        ).tocsr()
        matrix.sort_indices()
        b = np.linspace(1.0, 2.0, n)
        x0 = np.linspace(-1.0, 1.0, n)
        arguments = (
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data.copy(),
            np.full(n, 4.0),
            b,
        )
        max_iter = 1
        reference = _simple_gauss_seidel_reference(
            *arguments, x0, n, max_iter,
        )
        for protocol in ("pull", "counter"):
            with self.subTest(protocol=protocol):
                with redirect_stdout(io.StringIO()):
                    solver = parallel(
                        block_map="i",
                        thread_map="j",
                        function_props={
                            "Ap": ["monotonically_increasing"],
                            "Ai": [],
                        },
                        execution_mode="sync_free",
                        threads_per_block=128,
                        thread_unroll=1,
                        dependence_protocol=protocol,
                    )(self.simple_gauss_seidel_csr.original_function)
                for _ in range(3):
                    result = solver(*arguments, x0.copy(), n, max_iter)
                    np.testing.assert_allclose(
                        result, reference, rtol=1e-10, atol=1e-12,
                    )

    def test_sync_free_schedules_match_reference_over_several_sweeps(self):
        from compiler.decorator import parallel

        # Consecutive sweeps are ordered only by the stream, so every
        # sync-free executor must reset its counters before each sweep.
        schedules = {
            "thread per row": dict(threads_per_block=128),
            "warp per row": dict(
                thread_map="j", threads_per_block=128, thread_unroll=1,
            ),
            "block per row": dict(
                thread_map="j", threads_per_block=128, thread_unroll=2,
                n_warp=4,
            ),
        }
        max_iter = 4
        reference = _simple_gauss_seidel_reference(
            self.simple_Ap,
            self.simple_Ai,
            self.simple_Ax,
            self.D,
            self.b,
            self.x0,
            self.n,
            max_iter,
        )
        for name, options in schedules.items():
            with self.subTest(schedule=name):
                with redirect_stdout(io.StringIO()):
                    solver = parallel(
                        block_map="i",
                        function_props={
                            "Ap": ["monotonically_increasing"],
                            "Ai": [],
                        },
                        execution_mode="sync_free",
                        **options,
                    )(self.simple_gauss_seidel_csr.original_function)
                result = solver(
                    self.simple_Ap,
                    self.simple_Ai,
                    self.simple_Ax,
                    self.D,
                    self.b,
                    self.x0.copy(),
                    self.n,
                    max_iter,
                )
                np.testing.assert_allclose(
                    result, reference, rtol=1e-10, atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
