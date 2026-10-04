"""Tests for lowering break-prefix premises directly to SMT-LIB."""

import ast
from pathlib import Path
import unittest

from z3 import Solver, sat

from compiler.ast_normalization import normalize_aliases
from compiler.SMT_Generator import generate_dep_constraints, premise_to_smt
from compiler.spf_def import AndPremise, ForAllPremise, NotEqual
from compiler.spf_raising import (
    AssignNode,
    LoopBound,
    build_spf_from_ast,
    extract_node_data,
)


SPILU0_SOURCE = (
    Path(__file__).parents[2] / "benchmark" / "spilu0.py"
)


def _csr_break_premise():
    module = ast.parse(SPILU0_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ilu0_csr_numba"
    )
    function.decorator_list = []
    function_module = normalize_aliases(
        ast.Module(body=[function], type_ignores=[])
    )
    node_data = extract_node_data(
        build_spf_from_ast(function_module, block_iterator="i")
    )
    return next(
        premise
        for node in node_data.values()
        if isinstance(node, AssignNode)
        for premise in node.domain
        if isinstance(premise, ForAllPremise)
        and premise.quantified_var == "k_ptr_prev"
    )


class BreakSMTTests(unittest.TestCase):
    def test_python_range_uses_a_closed_lower_and_open_upper_bound(self):
        bound = LoopBound("0", "n")
        lower, upper = bound.to_premises("i")

        self.assertEqual(str(lower), "i >= 0")
        self.assertEqual(str(upper), "i < n")
        _, _, expression = bound.get_sympy_expr("i")
        self.assertTrue(bool(expression.subs({"i": 0, "n": 1})))

    def test_emits_bounded_forall_with_implication(self):
        premise = _csr_break_premise()
        emitted = premise_to_smt(
            premise,
            {"A_indptr", "A_indices"},
        )

        self.assertIn("(forall ((k_ptr_prev Int))", emitted)
        self.assertIn("(=> (and", emitted)
        self.assertIn("(>= k_ptr_prev (A_indptr i))", emitted)
        self.assertIn("(> k_ptr k_ptr_prev)", emitted)
        self.assertIn(
            "(not (>= (A_indices k_ptr_prev) i))",
            emitted,
        )

        smt = [
            "(declare-const i Int)\n",
            "(declare-const k_ptr Int)\n",
            "(declare-fun A_indices (Int) Int)\n",
            "(declare-fun A_indptr (Int) Int)\n",
        ]
        generate_dep_constraints(
            premise,
            smt,
            {"A_indptr", "A_indices"},
        )
        solver = Solver()
        solver.from_string("".join(smt))
        self.assertEqual(solver.check(), sat)

    def test_declares_float_scalar_used_by_break_domain(self):
        condition = AndPremise([NotEqual("diag_val", "0.0")])
        smt = []
        generate_dep_constraints(condition, smt, set())

        self.assertIn("(declare-const diag_val Real)\n", smt)
        solver = Solver()
        solver.from_string("".join(smt))
        self.assertEqual(solver.check(), sat)


if __name__ == "__main__":
    unittest.main()
