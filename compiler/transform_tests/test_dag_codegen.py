import ast
from pathlib import Path
import unittest

import numpy as np

from compiler.dag_codegen import (
    DAGCodegenError,
    compile_dag_builder,
    create_dag_plan,
    generate_dag_source
)
from compiler.dag_preprocessing import (
    DAGPreprocessingError,
    preprocess_dag_output,
)
from compiler.spf_def import DataDependence, Equality, GreaterThan
from compiler.spf_raising import Access, build_spf_from_ast, extract_node_data


SPTRSV_SOURCE = Path(__file__).parents[1] / "demo" / "sptrsv.py"


def _load_function(function_name):
    module = ast.parse(SPTRSV_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_module = ast.Module(body=[function], type_ignores=[])
    node_data = extract_node_data(build_spf_from_ast(function_module))
    arguments = [argument.arg for argument in function.args.args]
    return node_data, arguments


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


def _csr_ordering_dependence(node_data):
    # Actual parallel-decorator result:
    # S3 -> S2, i1 = Ai[j2], i2 > i1.
    return _make_dependence(
        node_data,
        from_statement_id="S3",
        to_statement_id="S2",
        premises=[
            Equality("i1", Access("Ai", "j2")),
            GreaterThan("i2", "i1"),
        ],
    )


def _csc_ordering_dependence(node_data):
    # Actual parallel-decorator result:
    # S2 -> S1, Ai[j1] = i2, i2 > i1.
    return _make_dependence(
        node_data,
        from_statement_id="S2",
        to_statement_id="S1",
        premises=[
            Equality(Access("Ai", "j1"), "i2"),
            GreaterThan("i2", "i1"),
        ],
    )


def _csc_reduction_dependence(node_data):
    # Actual parallel-decorator reduction:
    # S2 -> S2, Ai[j1] = Ai[j2], i2 > i1.
    dependence = _make_dependence(
        node_data,
        from_statement_id="S2",
        to_statement_id="S2",
        premises=[
            Equality(Access("Ai", "j1"), Access("Ai", "j2")),
            GreaterThan("i2", "i1"),
        ],
    )
    dependence.kind = "reduction"
    return dependence


class SpTRSVDAGCodegenTests(unittest.TestCase):
    def test_csr_and_csc_generate_the_same_task_dag(self):
        csr_nodes, csr_arguments = _load_function("simple_sptrsv_csr")
        csr_plan = create_dag_plan(
            [_csr_ordering_dependence(csr_nodes)],
            csr_nodes,
            parallel_iterator="i",
            arguments=csr_arguments,
        )
        csr_dag_builder = compile_dag_builder(csr_plan)

        # Lower-triangular CSR with the diagonal stored last in each row.
        csr_ap = [0, 1, 3, 6, 9]
        csr_ai = [0, 0, 1, 0, 1, 2, 1, 2, 3]
        csr_dag, csr_dep_count = csr_dag_builder(
            csr_ap,
            csr_ai,
            None,
            None,
            4,
        )

        csc_nodes, csc_arguments = _load_function("simple_sptrsv_csc")
        csc_plan = create_dag_plan(
            [_csc_ordering_dependence(csc_nodes)],
            csc_nodes,
            parallel_iterator="i",
            arguments=csc_arguments,
        )
        csc_dag_builder = compile_dag_builder(csc_plan)

        # The same lower-triangular structure in CSC, with diagonal first.
        # The repeated row 2 entry checks that a duplicate edge is counted once.
        csc_ap = [0, 4, 7, 9, 10]
        csc_ai = [0, 1, 2, 2, 1, 2, 3, 2, 3, 3]
        csc_dag, csc_dep_count = csc_dag_builder(
            csc_ap,
            csc_ai,
            None,
            None,
            4,
        )

        expected = [[1, 2], [2, 3], [3], []]
        expected_dep_count = [0, 1, 2, 2]
        self.assertEqual(csr_dag, expected)
        self.assertEqual(csc_dag, expected)
        self.assertEqual(csr_dep_count, expected_dep_count)
        self.assertEqual(csc_dep_count, expected_dep_count)
        execution_data = preprocess_dag_output(
            csc_dag,
            csc_dep_count,
            execution_mode="sync_free",
        )
        np.testing.assert_array_equal(
            execution_data.dep_count,
            [0, 1, 2, 2],
        )
        np.testing.assert_array_equal(
            execution_data.dag_ptr,
            [0, 2, 4, 5, 5],
        )
        np.testing.assert_array_equal(
            execution_data.dag_index,
            [1, 2, 2, 3, 3],
        )
        for plan in (csr_plan, csc_plan):
            generated_source = generate_dag_source(plan)
            self.assertIn("dep_count[_dag_destination] += 1", generated_source)
            self.assertNotIn("0 <= _dag_source", generated_source)
            self.assertNotIn("0 <= _dag_destination", generated_source)
            self.assertNotIn(
                "_dag_source_0 != _dag_destination_0",
                generated_source,
            )

    def test_csc_reduction_dependence_is_rejected(self):
        node_data, arguments = _load_function("simple_sptrsv_csc")
        reduction = _csc_reduction_dependence(node_data)
        with self.assertRaisesRegex(
                DAGCodegenError,
                "Reduction dependences",
        ):
            create_dag_plan(
                [reduction],
                node_data,
                parallel_iterator="i",
                arguments=arguments,
            )

    def test_wavefront_mode_uses_generated_indegrees(self):
        node_data, arguments = _load_function("simple_sptrsv_csc")
        plan = create_dag_plan(
            [_csc_ordering_dependence(node_data)],
            node_data,
            parallel_iterator="i",
            arguments=arguments,
        )
        wavefront_builder = compile_dag_builder(
            plan,
            execution_mode="wavefront",
        )
        dag, wavefronts = wavefront_builder(
            [0, 3, 6, 8, 9],
            [0, 1, 2, 1, 2, 3, 2, 3, 3],
            None,
            None,
            4,
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
        self.assertIn("if count == 0", source)
        self.assertIn(
            "remaining_dep_count[successor] -= 1",
            source,
        )

    def test_incomplete_wavefront_schedule_is_rejected(self):
        with self.assertRaisesRegex(
                DAGPreprocessingError,
                "may contain a cycle",
        ):
            preprocess_dag_output(
                [[1], [0]],
                [],
                execution_mode="wavefront",
            )

if __name__ == "__main__":
    unittest.main()
