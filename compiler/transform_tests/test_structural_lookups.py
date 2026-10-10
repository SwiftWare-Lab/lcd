"""Tests for generic inspector-hoisted sparse structural lookups."""

import ast
from pathlib import Path
import unittest

import numpy as np

from compiler.ast_normalization import normalize_aliases
from compiler.structural_lookups import (
    StructuralLookupError,
    create_structural_lookup_plan,
    materialize_structural_lookups,
    rewrite_structural_lookups,
)


SPILU0_SOURCE = Path(__file__).parents[1] / "demo" / "spilu0.py"
FUNCTION_PROPS = {
    "A_indptr": ["monotonically_increasing"],
    "A_indices": [],
    ("A_indptr", "A_indices"): [
        ("segmented_monotonically_increasing", "n"),
    ],
}


def _load_function(name):
    module = ast.parse(SPILU0_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    return normalize_aliases(ast.Module(body=[function], type_ignores=[]))


class StructuralLookupTests(unittest.TestCase):
    def test_detects_and_rewrites_current_csr_and_csc_diagonal_searches(self):
        for function_name in ("ilu0_csr", "ilu0_csc"):
            with self.subTest(function=function_name):
                module = _load_function(function_name)
                plan = create_structural_lookup_plan(
                    module,
                    FUNCTION_PROPS,
                )

                self.assertEqual(len(plan), 1)
                self.assertEqual(plan[0].pointer_array, "A_indptr")
                self.assertEqual(plan[0].index_array, "A_indices")
                source = ast.unparse(rewrite_structural_lookups(
                    module,
                    plan,
                    FUNCTION_PROPS,
                ))
                self.assertIn("_lcd_lookup_position_0[", source)
                self.assertIn("diag_val = A_data[", source)
                if function_name == "ilu0_csr":
                    self.assertNotIn("for diag_ptr in range", source)
                else:
                    self.assertNotIn("for dptr in range", source)

    def test_materializes_first_position_and_missing_sentinel(self):
        plan = create_structural_lookup_plan(
            _load_function("ilu0_csr"),
            FUNCTION_PROPS,
        )
        values = materialize_structural_lookups(plan, {
            "A_indptr": np.array([0, 2, 3, 5], dtype=np.int32),
            "A_indices": np.array([0, 2, 0, 1, 2], dtype=np.int32),
        })

        np.testing.assert_array_equal(
            values[plan[0].name],
            np.array([0, -1, 4], dtype=np.int32),
        )

    def test_rejects_inputs_that_violate_sorted_segment_property(self):
        plan = create_structural_lookup_plan(
            _load_function("ilu0_csr"),
            FUNCTION_PROPS,
        )

        with self.assertRaisesRegex(
                StructuralLookupError,
                "segmented_monotonically_increasing",
        ):
            materialize_structural_lookups(plan, {
                "A_indptr": np.array([0, 2], dtype=np.int32),
                "A_indices": np.array([1, 0], dtype=np.int32),
            })

    def test_does_not_hoist_lookup_with_independent_needle(self):
        module = ast.parse("""
def find_column(A_indptr, A_indices, targets, n):
    for row in range(n):
        value = -1
        for position in range(A_indptr[row], A_indptr[row + 1]):
            if A_indices[position] == targets[row]:
                value = position
                break
""")

        self.assertFalse(create_structural_lookup_plan(
            module,
            FUNCTION_PROPS,
        ))


if __name__ == "__main__":
    unittest.main()
