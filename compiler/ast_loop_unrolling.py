import ast
import copy
import timeit

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
        incr_step_name = "__incr_step"
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
    

    def visit_For(self, node):
        self.generic_visit(node)

        if not self.is_simple(node):
            return node
        
        # Convert for.target & for.iter into a while.test
        # Case 1: for x in y => initialize _$i = 0, while _$i < len(y), x = y[_$i], _$i += 1
        if isinstance(node.iter, ast.Name):
            loop_var_name = "__i"
            load_loop_var = ast.Name(loop_var_name, ctx=ast.Load())
            store_loop_var = ast.Name(loop_var_name, ctx=ast.Store())

            init_stmt = ast.Assign(
                targets=[store_loop_var],
                value=ast.Constant(0)
            )
            while_test = ast.Compare(
                left=load_loop_var,
                ops=[ast.Lt()],
                comparators=[
                    ast.Call(
                        func=ast.Name('len', ctx=ast.Load()), 
                        args=[node.iter],
                        keywords=[]
                    )
                ]
            )
            assign_stmt = ast.Assign(
                targets=[node.target],
                value=ast.Subscript(
                    value=node.iter,
                    slice=load_loop_var,
                    ctx=ast.Load(),
                )
            )
            incr_stmt = ast.AugAssign(
                target=store_loop_var,
                op=ast.Add(),
                value=ast.Constant(1),
            )

            node.body.insert(0, assign_stmt)
        # Case 2: for i in range(...) => same as above but use range args for incr and final and use original loop var
        elif isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            match len(node.iter.args):
                case 1:
                    init = ast.Constant(0)
                    final = node.iter.args[0]
                    incr = ast.Constant(1)
                case 2:
                    init = node.iter.args[0]
                    final = node.iter.args[1]
                    incr = ast.Constant(1)
                case 3:
                    init = node.iter.args[0]
                    final = node.iter.args[1]
                    incr = node.iter.args[2]
            
            init_stmt = ast.Assign(
                targets=[node.target],
                value=init
            )
            while_test = ast.Compare(
                left=ast.Name(node.target.id, ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[final]
            )
            incr_stmt = ast.AugAssign(
                target=node.target,
                op=ast.Add(),
                value=incr,
            )
        # Otherwise do nothing
        else:
            return node

        node.body.append(incr_stmt)
        while_loop = ast.While(while_test, node.body, node.orelse)
        return [init_stmt, *self.visit_While(while_loop)]


    def is_simple(self, loop):
        control_flow_nodes = (ast.If, ast.While, ast.For, ast.Try, ast.Break, ast.Continue, ast.Return)
        return all(not isinstance(stmt, control_flow_nodes) for stmt in loop.body)


test_codes = ["""
def test():
    a = [0] * 100_000
    i = 0
    while i < len(a):
        a[i] = i
        i += 1

test()
""", 

"""
def test():
    a = list(range(100_000))
    s = 0
    i = 0
    while i < len(a):
        s += i
        i += 1

test()
""",

"""
def test():
    y = list(range(100_000))
    s = 0
    for x in range(len(y)):
        s += y[x]

test()
"""
]


def run_code(src, number=100):
    code = compile(src, filename="<ast>", mode="exec")
    return timeit.timeit(lambda: exec(code), number=number)


factors = [1, 2, 3, 4, 8, 9, 10, 200]
for i, code in enumerate(test_codes):
    print(f"Test {i + 1}:")
    print(code)
    tree = ast.parse(code)
    print(f"Original: {run_code(code)}")

    for factor in factors:
        tree = ast.parse(code)
        unrolled_tree = LoopUnroller(factor).visit(tree)
        ast.fix_missing_locations(unrolled_tree)
        unrolled_code = ast.unparse(unrolled_tree)
        print(f"Unrolled (factor={factor}): {run_code(unrolled_code)}")
    print()
