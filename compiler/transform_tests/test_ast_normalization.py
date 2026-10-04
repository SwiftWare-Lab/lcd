"""Tests for alias and array-size normalization."""

import ast
from pathlib import Path
import unittest

from compiler.ast_normalization import normalize_aliases
from compiler.cuda_lowering import lower_to_numba_cuda
from compiler.cuda_runtime import evaluate_host_prelude
from compiler.dag_codegen import (
    compile_dag_builder,
    create_dag_plan,
    generate_dag_source,
)
from compiler.SMT_Generator import (
    generate_loop_bound_s_expressions_v2,
)
from compiler.spf_raising import (
    LoopNode,
    build_spf_from_ast,
    extract_node_data,
)


SPILU0_SOURCE = (
    Path(__file__).parents[2] / "benchmark" / "spilu0.py"
)


def _load_spilu0_function(name):
    module = ast.parse(SPILU0_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    return ast.Module(body=[function], type_ignores=[])


class AliasNormalizationTests(unittest.TestCase):
    def test_csr_ilu0_aliases_are_expanded(self):
        normalized = normalize_aliases(
            _load_spilu0_function("ilu0_csr_numba")
        )
        source = ast.unparse(normalized)

        self.assertIn("n = len(A_indptr) - 1", source)
        self.assertNotIn("row_start =", source)
        self.assertNotIn("row_end =", source)
        self.assertNotIn("k = A_indices[k_ptr]", source)
        self.assertNotIn("l_ik = A_data[k_ptr]", source)
        self.assertNotIn("j = A_indices[j_ptr]", source)
        self.assertIn(
            "for i in range(n):",
            source,
        )
        self.assertIn(
            "for diag_ptr in range(A_indptr[A_indices[k_ptr]], "
            "A_indptr[A_indices[k_ptr] + 1]):",
            source,
        )
        self.assertIn(
            "A_data[j_ptr] = A_data[j_ptr] - "
            "A_data[k_ptr] * A_data[kj_ptr]",
            source,
        )

    def test_csc_ilu0_aliases_are_expanded(self):
        normalized = normalize_aliases(
            _load_spilu0_function("ilu0_csc_numba")
        )
        source = ast.unparse(normalized)

        self.assertIn("n = len(A_indptr) - 1", source)
        self.assertNotIn("col_start =", source)
        self.assertNotIn("col_end =", source)
        self.assertNotIn("i = A_indices[i_ptr]", source)
        self.assertNotIn("l_ik = A_data[i_ptr]", source)
        self.assertNotIn("j_start =", source)
        self.assertNotIn("j_end =", source)
        self.assertIn(
            "for k in range(n):",
            source,
        )
        self.assertIn(
            "A_data[ij_ptr] = A_data[ij_ptr] - "
            "A_data[i_ptr] * u_kj",
            source,
        )
        # This variable is conditionally overwritten, so it is not an alias.
        self.assertIn("u_kj = 0.0", source)

    def test_alias_is_kept_when_an_array_write_invalidates_it(self):
        module = ast.parse(
            """
def snapshot(A, p):
    value = A[p]
    A[p] = 0
    return value
"""
        )
        source = ast.unparse(normalize_aliases(module))

        self.assertIn("value = A[p]", source)
        self.assertIn("return value", source)


class ArraySizeLoweringTests(unittest.TestCase):
    SOURCE = """
def sized_loop(indptr, data):
    n = len(indptr) - 1
    for i in range(n):
        data[i] = 0.0
"""

    def test_spf_and_dag_keep_named_size_parameter(self):
        module = ast.parse(self.SOURCE)
        node_data = extract_node_data(
            build_spf_from_ast(module, block_iterator="i")
        )
        loop = next(
            node
            for node in node_data.values()
            if isinstance(node, LoopNode) and node.iter_var == "i"
        )
        constants, _, _ = loop.bound.get_sympy_expr("i")
        self.assertIn("n", {str(value) for value in constants})
        smt_lines = []
        generate_loop_bound_s_expressions_v2(
            {"i": loop.bound},
            smt_lines,
        )
        self.assertIn("(declare-const n Int)\n", smt_lines)

        plan = create_dag_plan(
            [],
            node_data,
            parallel_iterator="i",
            arguments=["indptr", "data", "n"],
        )
        source = generate_dag_source(plan)
        self.assertIn("def extract_dag(indptr, data, n):", source)
        self.assertIn("range(n)", source)
        dag, dep_count = compile_dag_builder(plan)([0, 1, 2], None, 2)
        self.assertEqual(dag, [[], []])
        self.assertEqual(dep_count, [0, 0])

    def test_cuda_launcher_uses_lifted_size_parameter(self):
        result = lower_to_numba_cuda(
            ast.parse(self.SOURCE),
            loop_iterator="i",
        )

        self.assertEqual(len(result.host_prelude), 1)
        self.assertEqual(result.lifted_variables, ["n"])
        self.assertEqual(result.lifted_loop_parameters, ["n"])
        self.assertIn(
            "_blocks_per_grid = n",
            result.source,
        )
        self.assertIn("n", result.launcher_arguments)
        self.assertEqual(
            evaluate_host_prelude(
                result,
                {"indptr": [0, 1, 2], "data": None},
            )["n"],
            2,
        )

    def test_explicit_size_parameter_needs_no_host_alias(self):
        result = lower_to_numba_cuda(
            ast.parse(
                """
def sized_loop(indptr, data, n):
    for i in range(n):
        data[i] = 0.0
"""
            ),
            loop_iterator="i",
        )

        self.assertFalse(result.host_prelude)
        self.assertFalse(result.lifted_variables)
        self.assertFalse(result.lifted_loop_parameters)
        self.assertIn("n", result.launcher_arguments)
        self.assertIn("_blocks_per_grid = n", result.source)


if __name__ == "__main__":
    unittest.main()
