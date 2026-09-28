"""Run with python -m unittest discover -s tests -v."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import cholesky

from benchmark.cholesky_dense import cholesky_dense_numba
from benchmark.cholesky_dense_optimized import cholesky_dense_numba_optimized


class DenseCholeskyTests(unittest.TestCase):
    factor = staticmethod(cholesky_dense_numba)

    def check_factor(self, a, **kwargs):
        before = np.array(a, copy=True)
        lower = self.factor(a, **kwargs)
        assert_array_equal(a, before)
        self.assertEqual(lower.dtype, np.float64)
        assert_array_equal(lower, np.tril(lower))
        self.assertTrue(np.all(np.diag(lower) > 0))
        reference = np.asarray(a, dtype=np.float64)
        assert_allclose(lower, np.linalg.cholesky(reference), rtol=1e-11, atol=1e-12)
        assert_allclose(lower, cholesky(reference, lower=True), rtol=1e-11, atol=1e-12)
        reconstructed = np.einsum("ik,jk->ij", lower, lower, optimize=False)
        residual = np.linalg.norm(reference - reconstructed) / np.linalg.norm(reference)
        self.assertLess(residual, 1e-12)

    def test_known_factor(self):
        a = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        assert_allclose(self.factor(a), [[2, 0, 0], [6, 1, 0], [-8, 5, 3]])
        self.check_factor(a)

    def test_scalar_identity_and_diagonal(self):
        for a in (np.array([[9.0]]), np.eye(7), np.diag([1e-8, 1.0, 1e8])):
            with self.subTest(shape=a.shape):
                self.check_factor(a)

    def test_seeded_spd_matrices(self):
        rng = np.random.default_rng(42)
        for n in (2, 5, 16, 33, 64):
            with self.subTest(n=n):
                m = rng.standard_normal((n, n))
                self.check_factor(np.einsum("ik,jk->ij", m, m) + n * np.eye(n))

    def test_layout_dtype_and_readonly(self):
        rng = np.random.default_rng(7)
        m = rng.standard_normal((12, 12))
        a = np.einsum("ik,jk->ij", m, m) + 12 * np.eye(12)
        readonly = a.copy()
        readonly.flags.writeable = False
        for value in (a.astype(np.float32), np.asfortranarray(a), a[::2, ::2], readonly):
            with self.subTest(dtype=value.dtype, strides=value.strides):
                self.check_factor(value)

    def test_ill_conditioned_spd(self):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        a = np.einsum("ik,jk->ij", q * np.geomspace(1.0, 1e-8, 8), q)
        lower = self.factor(a)
        assert_allclose(lower, np.linalg.cholesky(a), rtol=1e-7, atol=1e-10)
        reconstructed = np.einsum("ik,jk->ij", lower, lower)
        self.assertLess(np.linalg.norm(a - reconstructed) / np.linalg.norm(a), 1e-12)

    def test_non_positive_definite(self):
        for a in (np.zeros((2, 2)), np.ones((2, 2)), np.diag([1., -1.]),
                  np.array([[1., 2.], [2., 1.]])):
            with self.subTest(a=a):
                with self.assertRaises(np.linalg.LinAlgError):
                    self.factor(a)

    def test_invalid_inputs(self):
        for a in ([], [1, 2], np.empty((0, 0)), np.ones((2, 3)),
                  np.ones((2, 2, 2)), [[1., 2.], [0., 1.]],
                  [[np.nan]], [[np.inf]], [[-np.inf]]):
            with self.subTest(a=a):
                with self.assertRaises(ValueError):
                    self.factor(a)
        for a in (np.eye(2, dtype=complex), [["1"]], np.eye(2, dtype=object)):
            with self.subTest(dtype=np.asarray(a).dtype):
                with self.assertRaises(TypeError):
                    self.factor(a)

    def test_roundoff_asymmetry_uses_lower_triangle(self):
        a = np.array([[4., 2. + 1e-13], [2., 3.]])
        before = a.copy()
        expected = np.linalg.cholesky(np.array([[4., 2.], [2., 3.]]))
        assert_allclose(self.factor(a), expected, rtol=1e-14, atol=0)
        assert_array_equal(a, before)


class OptimizedDenseCholeskyTests(DenseCholeskyTests):
    factor = staticmethod(cholesky_dense_numba_optimized)

    def test_block_boundaries_and_remainders(self):
        rng = np.random.default_rng(91)
        for n in (1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129):
            m = rng.standard_normal((n, n))
            a = np.einsum("ik,jk->ij", m, m) + n * np.eye(n)
            for block_size in (1, 2, 3, 7, 16, 32, 64, 128, 256):
                with self.subTest(n=n, block_size=block_size):
                    self.check_factor(a, block_size=block_size)

    def test_later_panel_failure_preserves_input(self):
        a = np.eye(65)
        a[64, 64] = -1
        before = a.copy()
        with self.assertRaises(np.linalg.LinAlgError):
            self.factor(a, block_size=32)
        assert_array_equal(a, before)

    def test_ill_conditioned_multiple_panels(self):
        rng = np.random.default_rng(72)
        q, _ = np.linalg.qr(rng.standard_normal((67, 67)))
        for condition in (1e4, 1e8, 1e10):
            a = np.einsum("ik,jk->ij", q * np.geomspace(1.0, 1.0 / condition, 67), q)
            reference = np.linalg.cholesky(a)
            for block_size in (3, 16, 32, 64):
                with self.subTest(condition=condition, block_size=block_size):
                    lower = self.factor(a, block_size=block_size)
                    assert_allclose(lower, reference, rtol=1e-4, atol=1e-9)
                    reconstructed = np.einsum("ik,jk->ij", lower, lower)
                    self.assertLess(np.linalg.norm(a - reconstructed) / np.linalg.norm(a), 1e-12)

    def test_block_size_contract(self):
        for block_size in (0, -1):
            with self.subTest(block_size=block_size):
                with self.assertRaises(ValueError):
                    self.factor(np.eye(3), block_size=block_size)
        for block_size in (True, np.bool_(False), 2.5, "32", None):
            with self.subTest(block_size=block_size):
                with self.assertRaises(TypeError):
                    self.factor(np.eye(3), block_size=block_size)
        for block_size in (np.int64(3), 10**30):
            self.check_factor(np.eye(3), block_size=block_size)

    def test_linear_system_residual(self):
        rng = np.random.default_rng(25)
        m = rng.standard_normal((67, 67))
        a = np.einsum("ik,jk->ij", m, m) + 67 * np.eye(67)
        b = rng.standard_normal((67, 3))
        lower = self.factor(a)
        from scipy.linalg import solve_triangular
        y = solve_triangular(lower, b, lower=True)
        x = solve_triangular(lower.T, y, lower=False)
        residual = np.linalg.norm(np.einsum("ij,jk->ik", a, x) - b) / np.linalg.norm(b)
        self.assertLess(residual, 1e-12)


if __name__ == "__main__":
    unittest.main()
