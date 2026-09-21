"""Print generated CUDA for the SpTRSV demos.

Run from the repository root with:

    python -m compiler.demo.show_cuda_lowering
"""

import ast
from pathlib import Path

from compiler.cuda_lowering import lower_to_numba_cuda
from compiler.spf_raising import build_spf_from_ast, extract_node_data


SPTRSV_SOURCE = Path(__file__).with_name("sptrsv.py")


def load_function(function_name):
    module = ast.parse(SPTRSV_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return ast.Module(body=[function], type_ignores=[])


def print_lowering(function_name, reductions=None):
    function_module = load_function(function_name)
    if reductions:
        node_data = extract_node_data(build_spf_from_ast(function_module))
        reductions = [node_data[statement_id] for statement_id in reductions]
    result = lower_to_numba_cuda(
        function_module,
        loop_iterator="i",
        reductions=reductions,
    )

    print("=" * 72)
    print(function_name)
    print("=" * 72)
    print(result.source)
    print()
    print("Kernel arguments:", ", ".join(result.kernel_arguments))
    print("Lifted host variables:", result.lifted_variables)
    print("Atomic statement IDs:", sorted(result.atomic_statement_ids))
    print("Removed host prelude:")
    for statement in result.host_prelude:
        print("  ", ast.unparse(statement))
    print("Removed host epilogue:")
    for statement in result.host_epilogue:
        print("  ", ast.unparse(statement))
    print()


def main():
    print_lowering("simple_sptrsv_csr")
    print_lowering("simple_sptrsv_csc", reductions=["S2"])


if __name__ == "__main__":
    main()

