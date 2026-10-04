import ast
from pathlib import Path
import unittest

import numpy as np

from compiler.cuda_lowering import lower_to_numba_cuda
from compiler.dag_codegen import (
    compile_dag_builder,
    create_dag_plan,
    generate_dag_source,
)
from compiler.dag_preprocessing import preprocess_dag_output
from compiler.spf_def import DataDependence, Equality, GreaterThan
from compiler.spf_raising import Access, build_spf_from_ast, extract_node_data


GAUSS_SEIDEL_SOURCE = (
    Path(__file__).parents[1] / "demo" / "gauss_seidel.py"
)


def _load_function(function_name):
    module = ast.parse(GAUSS_SEIDEL_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    node_data = extract_node_data(
        build_spf_from_ast(function_module, block_iterator="i")
    )
    arguments = [argument.arg for argument in function.args.args]
    return function_module, node_data, arguments


def _make_dependence(
        node_data,
        from_statement_id,
        to_statement_id,
        premises,
):
    from_statement = node_data[from_statement_id]
    to_statement = node_data[to_statement_id]
    return DataDependence(
        from_statement.schedule,
        to_statement.schedule,
        "x",
        premises,
        "RAW",
        from_domain=from_statement.loop_domain + from_statement.domain,
        to_domain=to_statement.loop_domain + to_statement.domain,
        from_statement_id=from_statement_id,
        to_statement_id=to_statement_id,
    )


def _gauss_seidel_ordering_dependence(node_data):
    # Expected parallel-decorator result for simple_gauss_seidel_csr:
    # S2 -> S1, i1 = Ai[j2], i2 > i1.
    #
    # S2 is the completed write to x[i] at the end of the source row.
    # S1 is the read/update in the destination row's sparse inner loop.
    return _make_dependence(
        node_data,
        from_statement_id="S2",
        to_statement_id="S1",
        premises=[
            Equality("i1", Access("Ai", "j2")),
            GreaterThan("i2", "i1"),
        ],
    )


def _full_gauss_seidel_ordering_dependence(node_data):
    # Expected ordering for full_gauss_seidel_csr:
    # S4 -> S3, i1 = Ai[j2], i2 > i1.
    #
    # S4 is the completed write to x[i]. S3 is the off-diagonal
    # accumulation guarded by the else-domain not(Ai[j] = i).
    return _make_dependence(
        node_data,
        from_statement_id="S4",
        to_statement_id="S3",
        premises=[
            Equality("i1", Access("Ai", "j2")),
            GreaterThan("i2", "i1"),
        ],
    )


class GaussSeidelCodegenTests(unittest.TestCase):
    def test_manual_dependence_generates_expected_wavefront_data(self):
        _, node_data, arguments = _load_function("simple_gauss_seidel_csr")
        plan = create_dag_plan(
            [_gauss_seidel_ordering_dependence(node_data)],
            node_data,
            parallel_iterator="i",
            arguments=arguments,
        )
        dag_builder = compile_dag_builder(
            plan,
            execution_mode="wavefront",
        )

        # CSR rows contain strictly-lower entries only; D stores the diagonal.
        # Dependencies are 0 -> {1, 2}, 1 -> {2, 3}, and 2 -> {3}.
        ap = [0, 0, 1, 3, 5]
        ai = [0, 0, 1, 1, 2]
        dag, wavefronts = dag_builder(
            ap,
            ai,
            None,
            None,
            None,
            None,
            4,
            2,
        )

        self.assertEqual(dag, [[1, 2], [2, 3], [3], []])
        self.assertEqual(wavefronts, [[0], [1], [2], [3]])

        execution_data = preprocess_dag_output(
            dag,
            wavefronts,
            execution_mode="wavefront",
        )
        np.testing.assert_array_equal(
            execution_data.wavefronts,
            [0, 1, 2, 3],
        )
        np.testing.assert_array_equal(
            execution_data.wavefront_ptr,
            [0, 1, 2, 3, 4],
        )

        source = generate_dag_source(
            plan,
            execution_mode="wavefront",
        )
        self.assertNotIn("for sweep", source)
        self.assertNotIn("sweep1", source)
        self.assertNotIn("sweep2", source)
        self.assertIn("_dag_source_0 = Ai[j]", source)
        self.assertIn("_dag_destination_0 = i", source)
        self.assertIn("if i > Ai[j]:", source)

    def test_lcd_frontend_records_full_gauss_seidel_else_domains(self):
        _, node_data, _ = _load_function("full_gauss_seidel_csr")

        self.assertEqual(str(node_data["S3"].schedule), "(0, i, 2, j, 1)")
        for node in node_data.values():
            schedule = getattr(node, "schedule", None)
            if schedule is not None:
                self.assertNotIn("sweep", schedule.schedule)
        self.assertEqual(str(node_data["S2"].write_access), "diag")
        self.assertEqual(str(node_data["S2"].domain[0]), "Ai[j] = i")
        self.assertEqual(str(node_data["S3"].write_access), "sigma")
        self.assertEqual(str(node_data["S3"].domain[0]), "not (Ai[j] = i)")
        self.assertEqual(node_data["S3"].update_op, "+")

    def test_full_manual_dependence_generates_expected_wavefront_data(self):
        _, node_data, arguments = _load_function("full_gauss_seidel_csr")
        plan = create_dag_plan(
            [_full_gauss_seidel_ordering_dependence(node_data)],
            node_data,
            parallel_iterator="i",
            arguments=arguments,
        )
        dag_builder = compile_dag_builder(
            plan,
            execution_mode="wavefront",
        )

        # Lower-triangular rows with the diagonal stored in each row. The
        # diagonal entries are filtered by the LCD-generated else-domain.
        ap = [0, 1, 3, 6, 9]
        ai = [0, 0, 1, 0, 1, 2, 1, 2, 3]
        dag, wavefronts = dag_builder(
            ap,
            ai,
            None,
            None,
            None,
            4,
            2,
        )

        self.assertEqual(dag, [[1, 2], [2, 3], [3], []])
        self.assertEqual(wavefronts, [[0], [1], [2], [3]])

        source = generate_dag_source(
            plan,
            execution_mode="wavefront",
        )
        self.assertNotIn("for sweep", source)
        self.assertNotIn("sweep1", source)
        self.assertNotIn("sweep2", source)
        self.assertIn("_dag_source_0 = Ai[j]", source)
        self.assertIn("_dag_destination_0 = i", source)
        self.assertIn("if i > Ai[j] and (not Ai[j] == i):", source)

    def test_wavefront_cuda_lowering_keeps_inner_row_loop_serial(self):
        function_module, _, _ = _load_function("simple_gauss_seidel_csr")
        result = lower_to_numba_cuda(
            function_module,
            loop_iterator="i",
            execution_mode="wavefront",
        )

        self.assertEqual(result.execution_mode, "wavefront")
        self.assertEqual(
            result.kernel_name,
            "simple_gauss_seidel_csr_cuda_kernel",
        )
        self.assertEqual(
            result.kernel_arguments,
            [
                "Ap",
                "Ai",
                "Ax",
                "D",
                "b",
                "x",
                "n",
                "max_iter",
                "sweep",
                "wavefronts",
                "wave_start",
                "wave_end",
            ],
        )
        self.assertEqual(
            result.launcher_arguments,
            [
                "Ap",
                "Ai",
                "Ax",
                "D",
                "b",
                "x",
                "n",
                "max_iter",
                "wavefronts",
                "wavefront_ptr",
            ],
        )
        self.assertEqual(result.host_loop_iterators, ("sweep",))
        self.assertIn("for sweep in range(max_iter):", result.source)
        self.assertIn("_wavefront_location = cuda.grid(1)", result.source)
        self.assertIn(
            "i = wavefronts[wave_start + _wavefront_location]",
            result.source,
        )
        self.assertIn(
            "for j in range(Ap[i], Ap[i + 1]):",
            result.source,
        )
        self.assertIn("x[i] -= Ax[j] * x[Ai[j]]", result.source)
        self.assertIn("x[i] /= D[i]", result.source)
        self.assertNotIn("cuda.atomic", result.source)
        self.assertNotIn("cuda.threadIdx", result.source)
        compile(result.module_ast, "<gauss-seidel-wavefront-lowering>", "exec")

    def test_full_wavefront_cuda_lowering_preserves_if_else(self):
        function_module, _, _ = _load_function("full_gauss_seidel_csr")
        result = lower_to_numba_cuda(
            function_module,
            loop_iterator="i",
            execution_mode="wavefront",
        )

        self.assertEqual(
            result.kernel_name,
            "full_gauss_seidel_csr_cuda_kernel",
        )
        self.assertEqual(result.host_loop_iterators, ("sweep",))
        self.assertIn("for sweep in range(max_iter):", result.source)
        self.assertIn("sigma = 0.0", result.source)
        self.assertIn("diag = 1.0", result.source)
        self.assertIn("if Ai[j] == i:", result.source)
        self.assertIn("diag = Ax[j]", result.source)
        self.assertIn("else:", result.source)
        self.assertIn("sigma += Ax[j] * x[Ai[j]]", result.source)
        self.assertIn("x[i] = (b[i] - sigma) / diag", result.source)
        compile(
            result.module_ast,
            "<full-gauss-seidel-wavefront-lowering>",
            "exec",
        )


if __name__ == "__main__":
    unittest.main()
