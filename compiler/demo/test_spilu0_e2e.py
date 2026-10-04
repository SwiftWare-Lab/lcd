"""Numerical physical-GPU tests for generated CSR and CSC ILU0."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from compiler.demo.gpu_test_utils import physical_cuda_status
from compiler.demo.suitesparse_test_matrix import (
    load_suitesparse_test_matrix,
)


CUDA_AVAILABLE, CUDA_SKIP_REASON = physical_cuda_status()


def _ilu0_csr_reference(data, indices, indptr):
    result = data.copy()
    for i in range(len(indptr) - 1):
        for k_ptr in range(indptr[i], indptr[i + 1]):
            k = indices[k_ptr]
            if k >= i:
                break
            diag_val = 0.0
            for diag_ptr in range(indptr[k], indptr[k + 1]):
                if indices[diag_ptr] == k:
                    diag_val = result[diag_ptr]
                    break
            if diag_val != 0.0:
                result[k_ptr] /= diag_val
                l_ik = result[k_ptr]
                for j_ptr in range(k_ptr + 1, indptr[i + 1]):
                    j = indices[j_ptr]
                    for kj_ptr in range(indptr[k], indptr[k + 1]):
                        if indices[kj_ptr] == j:
                            result[j_ptr] -= l_ik * result[kj_ptr]
                            break
    return result


def _ilu0_csc_reference(data, indices, indptr):
    result = data.copy()
    n = len(indptr) - 1
    for k in range(n):
        diag_val = 0.0
        for dptr in range(indptr[k], indptr[k + 1]):
            if indices[dptr] == k:
                diag_val = result[dptr]
                break
        for i_ptr in range(indptr[k], indptr[k + 1]):
            i = indices[i_ptr]
            if i <= k:
                continue
            if diag_val != 0.0:
                result[i_ptr] /= diag_val
                l_ik = result[i_ptr]
                for j in range(k + 1, n):
                    u_kj = 0.0
                    for kj_ptr in range(indptr[j], indptr[j + 1]):
                        if indices[kj_ptr] == k:
                            u_kj = result[kj_ptr]
                            break
                    if u_kj != 0.0:
                        for ij_ptr in range(indptr[j], indptr[j + 1]):
                            if indices[ij_ptr] == i:
                                result[ij_ptr] -= l_ik * u_kj
                                break
    return result


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_SKIP_REASON)
class SpILU0EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from compiler.demo.spilu0 import (
            compile_ilu0_csc,
            compile_ilu0_csr,
        )

        # The compiler prints its intermediate dependences and generated DAG.
        # Keep numerical test output focused on failures.
        with redirect_stdout(io.StringIO()):
            cls.csr_ilu0 = staticmethod(compile_ilu0_csr())
            cls.csc_ilu0 = staticmethod(compile_ilu0_csc())

    def setUp(self):
        self.matrix = load_suitesparse_test_matrix()

    def test_csr_matches_sequential_reference(self):
        matrix = csr_matrix(self.matrix)
        indptr = matrix.indptr.astype(np.int32)
        indices = matrix.indices.astype(np.int32)
        data = matrix.data.copy()

        result = self.csr_ilu0(data.copy(), indices, indptr)
        reference = _ilu0_csr_reference(data, indices, indptr)

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_csc_matches_sequential_reference(self):
        matrix = csc_matrix(self.matrix)
        indptr = matrix.indptr.astype(np.int32)
        indices = matrix.indices.astype(np.int32)
        data = matrix.data.copy()

        result = self.csc_ilu0(data.copy(), indices, indptr)
        reference = _ilu0_csc_reference(data, indices, indptr)

        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
