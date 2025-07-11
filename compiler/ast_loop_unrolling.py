import ast
import copy

class LoopUnroller(ast.NodeTransformer):
    def __init__(self, factor):
        self.factor = factor

    def visit_While(self, node):
        self.generic_visit(node)

        if not self.is_simple(node):
            return node

        remainder_loop = copy.deepcopy(node)
        node.body *= self.factor

        loop_var = node.test.left  # Stores initial value at this point
        incr = node.body[-1].value
        # Test becomes i + factor*incr - 1 < final
        # Compute factor*incr - 1 outside of loop
        incr_step_name = "_$incr_step"
        incr_step_assign = ast.Assign(
            targets=[ast.Name(incr_step_name, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.BinOp(
                    left=ast.Constant(self.factor),
                    op=ast.Mult(),
                    right=incr
                ),
                op=ast.Sub(),
                right=ast.Constant(1)
            )
        )
        
        node.test.left = ast.BinOp(
            left=loop_var,
            op=ast.Add(),
            right=ast.Name(incr_step_name, ctx=ast.Load())
        )

        return [incr_step_assign, node, remainder_loop]

    def is_simple(self, loop):
        control_flow_nodes = (ast.If, ast.While, ast.For, ast.Try, ast.Break, ast.Continue, ast.Return)
        return all(not isinstance(stmt, control_flow_nodes) for stmt in loop.body)


src1 = """
import numba
import time

# @numba.njit
def test():
    a = [0] * 100
    i = 0
    while i < len(a):
        a[i] = i
        i += 1

start = time.perf_counter()
test()
end = time.perf_counter()
print(end - start)
"""

src2 = """
import time
def test():
    a = list(range(1000))
    s = 0
    i = 0
    while i < len(a):
        s += i
        i += 1
    print(s)

start = time.perf_counter()
test()
end = time.perf_counter()
print(end - start)
"""

tree = ast.parse(src1)
unrolled_tree = LoopUnroller(factor=100).visit(tree)
ast.fix_missing_locations(unrolled_tree)
exec(compile(tree, filename="<ast>", mode="exec"))

print(ast.unparse(unrolled_tree))
print()

exec(compile(tree, filename="<ast>", mode="exec"))
exec(compile(unrolled_tree, filename="<ast>", mode="exec"))
