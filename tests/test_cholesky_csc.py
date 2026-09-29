"""Independent graph-pattern and numeric tests for the full CSC pipeline."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from benchmark.cholesky_csc import analyze_cholesky_csc, cholesky_csc
from benchmark.cholesky import cholesky_left_looking_numba


def graph_fill_oracle(a):
    """Small-test oracle: explicitly connect each eliminated vertex's neighbors.

    This graph clique algorithm is independent of the production etree/reach code.
    It preserves explicit structural zeros, so cancellation cannot hide fill.
    """
    lower = sp.tril(a, format="csc")
    n = a.shape[0]
    graph = [set() for _ in range(n)]
    for col in range(n):
        for row in lower.indices[lower.indptr[col]:lower.indptr[col + 1]]:
            if row != col:
                graph[col].add(int(row))
                graph[row].add(col)
    columns = []
    for col in range(n):
        neighbors = sorted(row for row in graph[col] if row > col)
        columns.append([col, *neighbors])
        for i, row in enumerate(neighbors):
            for other in neighbors[i + 1:]:
                graph[row].add(other)
                graph[other].add(row)
    return columns


def random_spd(n, seed, density=0.15):
    rng = np.random.default_rng(seed)
    edges = sp.random(n, n, density=density, random_state=rng, format="csc",
                      data_rvs=lambda size: rng.uniform(-1., 1., size))
    edges = sp.tril(edges, k=-1, format="csc")
    a = edges + edges.T
    return (a + sp.diags(np.asarray(abs(a).sum(axis=1)).ravel() + 1.)).tocsc()


class CSCCholeskyTests(unittest.TestCase):
    def check_factor(self, a):
        before = a.copy()
        symbolic = analyze_cholesky_csc(a)
        lower = cholesky_csc(a, symbolic)
        self.assertIsInstance(lower, sp.csc_matrix)
        self.assertTrue(lower.has_canonical_format)
        self.assertEqual(lower.dtype, np.float64)
        self.assertEqual(sp.triu(lower, k=1).nnz, 0)
        self.assertTrue(np.all(lower.diagonal() > 0))
        dense = a.toarray().astype(np.float64)
        assert_allclose(lower.toarray(), np.linalg.cholesky(dense), rtol=1e-11, atol=1e-12)
        assert_allclose(lower.toarray(), la.cholesky(dense, lower=True), rtol=1e-11, atol=1e-12)
        # SciPy's sparse norm may canonicalize duplicates in place; use a copy
        # so the test itself does not mutate the input whose preservation we check.
        self.assertLess(sp.linalg.norm(a - lower @ lower.T) / sp.linalg.norm(a.copy()), 1e-12)
        self.assertEqual((before != a).nnz, 0)
        return symbolic, lower

    def check_pattern(self, a):
        symbolic = analyze_cholesky_csc(a)
        columns = graph_fill_oracle(a)
        expected_parent = []
        for col, rows in enumerate(columns):
            assert_array_equal(symbolic.indices[symbolic.indptr[col]:symbolic.indptr[col + 1]], rows)
            expected_parent.append(rows[1] if len(rows) > 1 else -1)
        assert_array_equal(symbolic.parent, expected_parent)
        assert_array_equal(symbolic.column_counts, [len(rows) for rows in columns])
        for row in range(a.shape[0]):
            expected = [col for col, rows in enumerate(columns) if row in rows]
            actual = symbolic.prune_indices[symbolic.prune_indptr[row]:symbolic.prune_indptr[row + 1]]
            assert_array_equal(actual, expected)
            self.assertEqual(actual[-1], row)
        self.assertEqual(symbolic.nnz, sum(map(len, columns)))

    def test_known_fill_cycle(self):
        a = sp.csc_matrix([[4., 1., 1., 0.], [1., 4., 0., 1.],
                           [1., 0., 4., 1.], [0., 1., 1., 4.]])
        symbolic, lower = self.check_factor(a)
        self.check_pattern(a)
        self.assertEqual(symbolic.nnz, sp.tril(a).nnz + 1)
        self.assertEqual(a[2, 1], 0.)
        self.assertNotEqual(lower[2, 1], 0.)

    def test_random_patterns_against_graph_elimination(self):
        for n in (1, 2, 5, 12, 25):
            for seed in range(5):
                with self.subTest(n=n, seed=seed):
                    a = random_spd(n, seed, density=0.3)
                    self.check_pattern(a)
                    self.check_factor(a)

    def test_diagonal_tridiagonal_and_disconnected(self):
        tridiagonal = sp.diags([-np.ones(8), 4 * np.ones(9), -np.ones(8)], [-1, 0, 1], format="csc")
        for a in (sp.csc_matrix([[9.]]), sp.diags([1e-8, 1., 1e8], format="csc"),
                  tridiagonal, sp.block_diag((tridiagonal, np.eye(3), tridiagonal), format="csc")):
            with self.subTest(shape=a.shape):
                self.check_pattern(a)
                self.check_factor(a)

    def test_value_independence_and_structural_cancellation(self):
        a = sp.csc_matrix([[4., 2., 2.], [2., 5., 1.], [2., 1., 5.]])
        symbolic, lower = self.check_factor(a)
        self.assertEqual(lower[2, 1], 0.)
        self.assertEqual(lower.nnz, 6)  # A zero numeric entry remains allocated.
        other = analyze_cholesky_csc(-a)
        assert_array_equal(symbolic.indptr, other.indptr)
        assert_array_equal(symbolic.indices, other.indices)
        assert_array_equal(symbolic.parent, other.parent)
        with self.assertRaises(np.linalg.LinAlgError):
            cholesky_csc(-a, symbolic)

    def test_explicit_zeros_are_structural(self):
        a = sp.csc_matrix((np.array([2., 0., 0., 3.]),
                           np.array([0, 1, 0, 1]), np.array([0, 2, 4])), shape=(2, 2))
        symbolic, _ = self.check_factor(a)
        self.check_pattern(a)
        self.assertEqual(symbolic.nnz, 3)
        changed = a.copy()
        changed.data[[1, 2]] = 0.5
        assert_allclose(cholesky_csc(changed, symbolic).toarray(), np.linalg.cholesky(changed.toarray()))
        removed = a.copy()
        removed.eliminate_zeros()
        with self.assertRaises(ValueError):
            cholesky_csc(removed, symbolic)

    def test_unsorted_duplicates_and_input_preservation(self):
        a = sp.csc_matrix((np.array([1, 2, 1, 1, 2, 1, 1]),
                           np.array([1, 0, 0, 1, 1, 0, 0]),
                           np.array([0, 3, 7])), shape=(2, 2))
        # Canonical matrix is [[3, 2], [1, 3]] until the first value becomes 2.
        a.data[0] = 2
        before = (a.data.copy(), a.indices.copy(), a.indptr.copy())
        self.check_factor(a)
        for actual, original in zip((a.data, a.indices, a.indptr), before):
            assert_array_equal(actual, original)

    def test_sparse_formats_and_float32(self):
        a = random_spd(12, 42)
        for value in (a.astype(np.float32), a.tocsr(), a.tocoo(), sp.csc_array(a)):
            with self.subTest(format=value.format, dtype=value.dtype):
                self.check_factor(value)

    def test_lower_triangle_storage_and_cross_form_reuse(self):
        full = random_spd(24, 52)
        stored_lower = sp.tril(full, format="csc")
        reference = np.linalg.cholesky(full.toarray())
        full_analysis = analyze_cholesky_csc(full)
        lower_analysis = analyze_cholesky_csc(stored_lower)
        assert_array_equal(full_analysis.indptr, lower_analysis.indptr)
        assert_array_equal(full_analysis.indices, lower_analysis.indices)

        forms = (stored_lower, stored_lower.tocsr(), stored_lower.tocoo(),
                 sp.csc_array(stored_lower))
        for value in forms:
            before = value.copy()
            with self.subTest(format=value.format):
                factor = cholesky_csc(value, full_analysis)
                assert_allclose(factor.toarray(), reference, rtol=1e-11, atol=1e-12)
                self.assertLess(sp.linalg.norm(full - factor @ factor.T) / sp.linalg.norm(full),
                                1e-12)
                self.assertEqual((before != value).nnz, 0)

        # Analysis reuse depends on the canonical lower pattern, not whether the
        # caller supplied one or both triangles.
        assert_allclose(cholesky_csc(full, lower_analysis).toarray(), reference,
                        rtol=1e-11, atol=1e-12)

    def test_analysis_reuse_and_ownership(self):
        a = random_spd(18, 8)
        symbolic = analyze_cholesky_csc(a)
        with patch("benchmark.cholesky_csc._analyze_prepared", side_effect=AssertionError("analysis repeated")):
            for scale in (0.5, 2., 10.):
                lower = cholesky_csc(a * scale, symbolic)
                assert_allclose(lower.toarray(), np.linalg.cholesky((a * scale).toarray()), atol=1e-12)
        lower.indices[:] = 0
        lower.indptr[:] = 0
        # Modifying an output factor cannot corrupt the analysis for the next call.
        assert_allclose(cholesky_csc(a, symbolic).toarray(), np.linalg.cholesky(a.toarray()), atol=1e-12)
        with self.assertRaises(ValueError):
            symbolic.indices[0] = 99
        changed = a.tolil()
        absent = next((i, j) for i in range(18) for j in range(i) if a[i, j] == 0)
        changed[absent] = changed[absent[::-1]] = 0.01
        with self.assertRaises(ValueError):
            cholesky_csc(changed.tocsc(), symbolic)
        with self.assertRaises(ValueError):
            cholesky_csc(sp.eye(3, format="csc"), symbolic)
        with self.assertRaises(TypeError):
            cholesky_csc(a, symbolic="invalid")

    def test_no_dense_or_reference_factor_in_pipeline(self):
        a = random_spd(30, 1)
        with patch.object(sp.csc_matrix, "toarray", side_effect=AssertionError("dense conversion")), \
             patch.object(sp.csc_matrix, "todense", side_effect=AssertionError("dense conversion")), \
             patch("numpy.linalg.cholesky", side_effect=AssertionError("reference factor")), \
             patch("scipy.linalg.cholesky", side_effect=AssertionError("reference factor")):
            symbolic = analyze_cholesky_csc(a)
            lower = cholesky_csc(a, symbolic)
            self.assertLess(sp.linalg.norm(a - lower @ lower.T) / sp.linalg.norm(a), 1e-12)

    def test_large_sparse_diagonal(self):
        a = sp.eye(10000, format="csc")
        symbolic = analyze_cholesky_csc(a)
        self.assertEqual(symbolic.nnz, 10000)
        self.assertTrue(np.all(symbolic.parent == -1))
        lower = cholesky_csc(a, symbolic)
        self.assertEqual(lower.nnz, 10000)
        assert_array_equal(lower.data, np.ones(10000))

    def test_sparse_linear_solve(self):
        a = random_spd(80, 37, density=0.05)
        b = np.random.default_rng(37).standard_normal(80)
        lower = cholesky_csc(a)
        y = spsolve_triangular(lower, b, lower=True)
        x = spsolve_triangular(lower.T.tocsc(), y, lower=False)
        self.assertLess(np.linalg.norm(a @ x - b) / np.linalg.norm(b), 1e-12)

    def test_ill_conditioned_spd(self):
        rng = np.random.default_rng(19)
        q, _ = np.linalg.qr(rng.standard_normal((33, 33)))
        dense = np.einsum("ik,jk->ij", q * np.geomspace(1., 1e-8, 33), q)
        a = sp.csc_matrix(dense)
        lower = cholesky_csc(a)
        assert_allclose(lower.toarray(), np.linalg.cholesky(dense), rtol=1e-5, atol=1e-9)
        self.assertLess(sp.linalg.norm(a - lower @ lower.T) / sp.linalg.norm(a), 1e-12)

    def test_invalid_shape_type_and_values(self):
        for a in (np.eye(2), [[1.]]):
            with self.assertRaises(TypeError):
                analyze_cholesky_csc(a)
        for a in (sp.csc_matrix((0, 0)), sp.csc_matrix((2, 3)),
                  sp.csc_matrix([[1., 2.], [0., 1.]]), sp.csc_matrix([[np.nan]]),
                  sp.csc_matrix([[np.inf]])):
            with self.subTest(shape=a.shape):
                with self.assertRaises(ValueError):
                    cholesky_csc(a)
        for dtype in (complex, bool):
            with self.assertRaises(TypeError):
                cholesky_csc(sp.eye(3, dtype=dtype, format="csc"))

    def test_nonpositive_or_missing_pivots(self):
        for a in (sp.csc_matrix((3, 3)), sp.csc_matrix(np.ones((3, 3))),
                  sp.diags([1., 0., 2.], format="csc"),
                  sp.diags([1., 2., -1.], format="csc"),
                  sp.csc_matrix([[1., 2.], [2., 1.]])):
            symbolic = analyze_cholesky_csc(a)  # Structure is valid even when not SPD.
            with self.assertRaises(np.linalg.LinAlgError):
                cholesky_csc(a, symbolic)

    def test_roundoff_asymmetry_uses_lower(self):
        a = sp.csc_matrix([[4., 2. + 1e-13], [2., 3.]])
        assert_allclose(cholesky_csc(a).toarray(), np.linalg.cholesky([[4., 2.], [2., 3.]]), atol=1e-14)

    def test_numeric_kernel_rejects_inconsistent_dependency(self):
        # A claimed contribution absent from L must not fall through to index -1.
        a = sp.eye(2, format="csc")
        values = np.zeros(2)
        valid = cholesky_left_looking_numba(2, a.indptr, a.indices, a.data,
            a.indptr, a.indices, values, np.array([0, 1, 3], dtype=np.int32),
            np.array([0, 0, 1], dtype=np.int32))
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
