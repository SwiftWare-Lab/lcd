"""Pipeline evaluation for the CSR and CSC ILU0 implementations."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from compiler.cuda_lowering import lower_to_numba_cuda
from compiler.cuda_runtime import compile_cuda_module
from compiler.dag_codegen import (
    COUNTER_DEPENDENCE,
    PULL_DEPENDENCE,
    DirectDependence,
    compile_dag_builder,
    create_dag_plan,
    direct_dependence,
    generate_dag_source,
)
from compiler.decorator import parallel
from compiler.legality_check import finalize_legality_of_fusion
from compiler.spf_raising import build_spf_from_ast, extract_node_data
from compiler.demo.spilu0 import (
    FUNCTION_PROPS,
    compile_ilu0_csc,
    compile_ilu0_csr,
    ilu0_csr,
)


SPILU0_SOURCE = Path(__file__).parents[1] / "demo" / "spilu0.py"


def _load_function(function_name):
    module = ast.parse(SPILU0_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    return ast.Module(body=[function], type_ignores=[])


class SpILU0CodegenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {
            "csr": _load_function("ilu0_csr"),
            "csc": _load_function("ilu0_csc"),
        }
        cls.iterators = {"csr": "i", "csc": "k"}
        cls.legality = {}
        cls.node_data = {}
        for storage in ("csr", "csc"):
            iterator = cls.iterators[storage]
            module = cls.modules[storage]
            cls.legality[storage] = finalize_legality_of_fusion(
                (iterator, iterator),
                module,
                FUNCTION_PROPS,
                block_iterator=iterator,
            )
            cls.node_data[storage] = extract_node_data(
                build_spf_from_ast(module, block_iterator=iterator)
            )

    def test_frontend_extracts_break_aware_dependences(self):
        expected_counts = {"csr": 2, "csc": 3}
        for storage in ("csr", "csc"):
            with self.subTest(storage=storage):
                _, analysis = self.legality[storage]
                self.assertEqual(
                    len(analysis.ordering_dependences),
                    expected_counts[storage],
                )
                self.assertFalse(analysis.reduction_dependences)
                conditions = "\n".join(
                    str(dependence)
                    for dependence in analysis.ordering_dependences
                )
                self.assertIn("forall ", conditions)
                self.assertIn(">= A_indptr[", conditions)
                self.assertNotIn("l_ik", conditions)

        csr_dependences = self.legality["csr"][1].ordering_dependences
        self.assertNotIn(
            ("S2", "S3"),
            {
                (dependence.from_statement_id, dependence.to_statement_id)
                for dependence in csr_dependences
            },
        )

    def test_cuda_lowering_and_binding_succeed(self):
        for storage in ("csr", "csc"):
            with self.subTest(storage=storage):
                module = self.modules[storage]
                iterator = self.iterators[storage]
                result = lower_to_numba_cuda(
                    module,
                    loop_iterator=iterator,
                    execution_mode="wavefront",
                )
                compiled = compile_cuda_module(result)

                self.assertTrue(callable(compiled.kernel))
                self.assertTrue(callable(compiled.launcher))
                self.assertEqual(result.lifted_loop_parameters, ["n"])
                self.assertEqual(
                    sum(
                        isinstance(node, ast.Break)
                        for node in ast.walk(result.kernel_ast)
                    ),
                    3,
                )
                self.assertNotIn("l_ik =", result.source)

    def test_dag_planner_projects_relations_with_segment_ownership(self):
        arguments = ["A_data", "A_indices", "A_indptr", "n"]
        plans = {}
        sources = {}
        for storage in ("csr", "csc"):
            _, analysis = self.legality[storage]
            plans[storage] = create_dag_plan(
                analysis.ordering_dependences,
                self.node_data[storage],
                parallel_iterator=self.iterators[storage],
                arguments=arguments,
                function_props=FUNCTION_PROPS,
            )
            sources[storage] = generate_dag_source(plans[storage])

        csr_source = sources["csr"]
        self.assertEqual(len(plans["csr"].edge_rules), 1)
        self.assertIn("for i in range(0, n):", csr_source)
        self.assertIn(
            "for k_ptr in range(A_indptr[i], A_indptr[i + 1]):",
            csr_source,
        )
        self.assertIn("if A_indices[k_ptr] >= i:", csr_source)
        self.assertIn("_dag_source_0 = A_indices[k_ptr]", csr_source)
        self.assertIn("_dag_destination_0 = i", csr_source)
        self.assertNotIn("not A_indices[k_ptr] >= i", csr_source)
        self.assertNotIn("j_ptr", csr_source)
        self.assertNotIn("diag_ptr", csr_source)
        self.assertNotIn("kj_ptr", csr_source)
        self.assertNotIn("i1", csr_source)
        self.assertNotIn("i2", csr_source)

        csc_source = sources["csc"]
        self.assertEqual(len(plans["csc"].edge_rules), 2)
        self.assertIn("_dag_destination_0 = A_indices[i_ptr]", csc_source)
        self.assertIn("for j in range(k + 1, n):", csc_source)
        self.assertIn("_dag_destination_1 = j", csc_source)
        self.assertNotIn("ij_ptr", csc_source)
        self.assertNotIn("for dptr", csc_source)
        self.assertNotIn("kj_ptr", csc_source)
        self.assertNotIn("k1", csc_source)
        self.assertNotIn("k2", csc_source)

        for storage in ("csr", "csc"):
            with self.subTest(storage=storage):
                builder = compile_dag_builder(plans[storage])
                dag, dep_count = builder(
                    [1.0] * 9,
                    [0, 1, 2, 0, 1, 2, 0, 1, 2],
                    [0, 3, 6, 9],
                    3,
                )
                self.assertEqual(dag, [[1, 2], [2], []])
                self.assertEqual(dep_count, [0, 1, 2])

    def test_ownership_projection_requires_monotonic_property(self):
        _, analysis = self.legality["csr"]
        plan = create_dag_plan(
            analysis.ordering_dependences,
            self.node_data["csr"],
            parallel_iterator="i",
            arguments=["A_data", "A_indices", "A_indptr", "n"],
        )
        source = generate_dag_source(plan)
        self.assertIn("for i1 in range(0, n):", source)
        self.assertIn("for i2 in range(0, n):", source)

    def test_csr_pivot_loop_carries_a_pull_dependence(self):
        arguments = ["A_data", "A_indices", "A_indptr", "n"]
        _, analysis = self.legality["csr"]
        plan = create_dag_plan(
            analysis.ordering_dependences,
            self.node_data["csr"],
            parallel_iterator="i",
            arguments=arguments,
            function_props=FUNCTION_PROPS,
        )
        dependence = direct_dependence(plan, "j_ptr")

        self.assertEqual(dependence.kind, PULL_DEPENDENCE)
        self.assertEqual(dependence.iterator, "k_ptr")
        self.assertEqual(ast.unparse(dependence.endpoint), "A_indices[k_ptr]")

    def test_csr_pull_waits_for_each_pivot_row(self):
        with patch(
            "compiler.decorator.finalize_legality_of_fusion",
            return_value=self.legality["csr"],
        ):
            with redirect_stdout(io.StringIO()):
                compiled = parallel(
                    block_map="i",
                    thread_map="j_ptr",
                    function_props=FUNCTION_PROPS,
                    execution_mode="sync_free",
                    threads_per_block=128,
                    thread_unroll=1,
                    dependence_protocol="pull",
                )(ilu0_csr)
        source = compiled.cuda_kernel_source
        pivot_loop = source[source.index("for k_ptr in range("):]

        self.assertEqual(
            compiled.cuda_lowering.dependence_protocol,
            PULL_DEPENDENCE,
        )
        self.assertNotIn("dep_count", source)
        self.assertLess(
            pivot_loop.index("break"),
            pivot_loop.index(
                "while cuda.atomic.add(done, A_indices[k_ptr], 0) == 0:"
            ),
        )
        self.assertLess(
            pivot_loop.index("cuda.atomic.add(done, A_indices[k_ptr], 0)"),
            pivot_loop.index("_lcd_lookup_position_"),
        )
        self.assertIn("cuda.atomic.exch(done, i, 1)", source)

    def test_pull_falls_back_when_a_break_test_reads_task_output(self):
        module = ast.parse(
            """
def scaled(A_data, A_indices, A_indptr):
    n = len(A_indptr) - 1
    for i in range(n):
        for k_ptr in range(A_indptr[i], A_indptr[i + 1]):
            if A_data[k_ptr] == 0.0:
                break
            for j_ptr in range(k_ptr + 1, A_indptr[i + 1]):
                A_data[j_ptr] = A_data[j_ptr] * A_data[A_indptr[A_indices[k_ptr]]]
    return A_data
"""
        )
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            execution_mode="sync_free",
            thread_loop_iterator="j_ptr",
            n_warp=1,
            dependence_protocol=PULL_DEPENDENCE,
            direct_dependence=DirectDependence(
                kind=PULL_DEPENDENCE,
                endpoint=ast.parse("A_indices[k_ptr]", mode="eval").body,
                guard=None,
                iterator="k_ptr",
            ),
        )

        self.assertEqual(result.dependence_protocol, COUNTER_DEPENDENCE)
        self.assertNotIn("done", result.source)

    def test_real_decorator_builds_simplified_dag_inspector(self):
        compile_functions = {
            "csr": compile_ilu0_csr,
            "csc": compile_ilu0_csc,
        }
        for storage in ("csr", "csc"):
            with self.subTest(storage=storage):
                legality_result = self.legality[storage]
                with patch(
                    "compiler.decorator.finalize_legality_of_fusion",
                    return_value=legality_result,
                ):
                    with redirect_stdout(io.StringIO()):
                        compiled = (
                            compile_functions[storage](thread_unroll=1)
                            if storage == "csr"
                            else compile_functions[storage]()
                        )
                self.assertTrue(callable(compiled))
                self.assertEqual(len(compiled.structural_lookups), 1)
                lookup_name = compiled.structural_lookups[0].name
                self.assertIn(lookup_name, compiled.cuda_kernel_source)
                self.assertIn(lookup_name, compiled.cuda_launcher_source)
                if storage == "csr":
                    self.assertNotIn(
                        "for diag_ptr in range",
                        compiled.cuda_kernel_source,
                    )
                else:
                    self.assertNotIn(
                        "for dptr in range",
                        compiled.cuda_kernel_source,
                    )


if __name__ == "__main__":
    unittest.main()
