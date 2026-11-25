"""Numerical end-to-end tests for generated sparse triangular solvers."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import tril
from scipy.sparse.linalg import spsolve_triangular

from compiler.e2e_tests.gpu_test_utils import (
    load_test_matrix,
    physical_cuda_status,
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
        source_matrix = load_test_matrix()
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

    def test_prepared_launches_are_repeatable_on_device_data(self):
        from numba import cuda

        from compiler.decorator import parallel

        # A prepared launch zeroes lifted arrays and resets dependency state
        # on the device, so it can be relaunched without new uploads.
        schedules = {
            ("csr", "pull"): dict(dependence_protocol="pull", n_warp=1),
            ("csr", "counter"): dict(dependence_protocol="counter", n_warp=1),
            ("csr", "block"): dict(n_warp=4),
            ("csc", "push"): dict(dependence_protocol="push", n_warp=1),
            ("csc", "block"): dict(n_warp=4),
        }
        solvers = {"csr": self.csr_solver, "csc": self.csc_solver}
        properties = {
            "csr": {"Ap": ["monotonically_increasing"], "Ai": []},
            "csc": {
                "Ap": ["monotonically_increasing"],
                "Ai": [],
                ("Ap", "Ai"): ["triangularity"],
            },
        }
        matrix_types = {"csr": csr_matrix, "csc": csc_matrix}
        for (storage, protocol), options in schedules.items():
            with self.subTest(storage=storage, protocol=protocol):
                with redirect_stdout(io.StringIO()):
                    solver = parallel(
                        block_map="i",
                        thread_map="j",
                        function_props=properties[storage],
                        execution_mode="sync_free",
                        threads_per_block=128,
                        thread_unroll=1,
                        **options,
                    )(solvers[storage].original_function)
                matrix = matrix_types[storage](self.lower_triangular)
                matrix.sort_indices()
                arguments = (
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
                prepared = solver.prepare_cuda_arguments(*arguments)
                for _ in range(3):
                    solver.cuda_launcher(*prepared.launcher_arguments)
                    cuda.synchronize()
                    np.testing.assert_allclose(
                        prepared.copy_to_host("x"),
                        reference,
                        rtol=1e-10,
                        atol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
