import ast
from copy import deepcopy
import inspect
from compiler.cfg import CFGBuilder, CFG

from benchmark.sptrsv import simple_sptrsv_csc


"""
Input: SpTRSV CSC {AST, CFG}
Output: 3 code blocks {flag initializer: FI, flag checker: FC, flag setter: FS}

# Generate R & W
AST' := AST
for every array access A in a loop in AST':
  i := loop variable
  j := expr inside subscript of A
  replace the statement containing A in AST' with a statement which adds j into R or W at set i

R, W := exec(AST') # ?

# Generate FI (generating W did most of the work already)
FI := {}
i := loop variable of outer loop in AST
add statement `i = tl.program_id(axis=0)` to FI
add early return if i < N to FI where N is matrix dimension

# Generate FC
FC := {}
add while loop to FC where the condition checks if W[i] is empty and the body is noop

# Generate FS
FS := {}
add statements to FS which load start and end indices of column i

let op be the first statement of the outer loop in AST

substitute array accesses in the rhs of op with tl.load(ptr + i) where ptr is a triton pointer that corresponds to the original array

substitute array access in the lhs of op with tl.store(ptr + i, rhs of op)

add op to FS

add statements to FS which load the data values below the diagonal entry of column i into d_ptr

let x_ptr be a triton ptr that corresponds to the array being written to in AST

add `partial_sum = d_ptr * x_ptr` to FS

add statement to FS which atomically adds partial sum to entries in x_ptr that correspond to the indices of the data values loaded in d_ptr

add statement to FS which removes i from the sets in W[j] for each j in the indices of the data values loaded in d_ptr
"""


class WritesTracker(ast.NodeTransformer):
    def __init__(self):
        # When a For loop is visited, this gets set
        self.outer_loop_var = None
        # Gets passed to ReadsTracker so it knows which subscripts are not reads
        self.write_nodes = set()  
        # * Assume only 1 array is written to
        self.array_written_to = None

    def generate_add_write_stmt(self, write_index):
        return ast.Expr(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name("writes", ctx=ast.Load()),
                        slice=self.outer_loop_var,
                    ),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[write_index],
                keywords=[],
            )
        )

    def visit_For(self, node):
        if not self.outer_loop_var:
            self.outer_loop_var = node.target
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        # * Assume 1 target (so no a, b = c, d)
        if self.outer_loop_var and isinstance(node.targets[0], ast.Subscript):
            index = node.targets[0].slice
            self.write_nodes.add(node.targets[0])
            self.array_written_to = node.targets[0].value
            return [node, self.generate_add_write_stmt(index)]

        return node

    def visit_AugAssign(self, node):
        # * Assume 1 target (so no a, b = c, d)
        if self.outer_loop_var and isinstance(node.target, ast.Subscript):
            index = node.target.slice
            self.write_nodes.add(node.target)
            self.array_written_to = node.target.value
            return [node, self.generate_add_write_stmt(index)]

        return node


class ReadsTracker(ast.NodeTransformer):
    def __init__(self, write_nodes, array_written_to):
        self.loop_var = None
        self.write_nodes = write_nodes
        self.array_written_to = array_written_to
        self.added_reads_from_stmt = []

    def generate_add_read_stmt(self, read_index):
        return ast.Expr(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name("reads", ctx=ast.Load()),
                        slice=self.loop_var,
                    ),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[read_index],
                keywords=[],
            )
        )

    def visit_For(self, node):
        self.loop_var = node.target
        self.generic_visit(node)
        node.body += self.added_reads_from_stmt
        self.added_reads_from_stmt = []
        return node

    def visit_Subscript(self, node):
        if not self.loop_var or node in self.write_nodes or node.value != self.array_written_to:
            return node
        index = node.slice
        self.added_reads_from_stmt.append(self.generate_add_read_stmt(index))
        return node
    
    def visit_Return(self, node):
        return ast.Return(
            ast.Tuple(
                elts=[ast.Name("reads", ctx=ast.Load()), 
                      ast.Name("writes", ctx=ast.Load())], 
                ctx=ast.Load()
            )
        )


class SptrsvParallelizer(ast.NodeTransformer):
    def __init__(self, cfg: CFG):
        self.cfg = cfg

    def visit_FunctionDef(self, node):
        """Takes sequential CSC SpTRSV function, outputs parallel Triton kernel"""

        output = [ast.Import([ast.alias("triton"), ast.alias("triton.language", "tl")])]

        # * Assume first arg is the sparse matrix and second arg is the rhs of the sptrsv equation
        matrix = node.args.args[0]
        rhs = node.args.args[1]

        N = ast.Name("__N", ctx=ast.Load())

        # Generate function that creates Read and Write sets
        lineno = node.lineno
        get_read_writes_body = [
            ast.Assign(
                targets=[ast.Name("reads", ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.List(
                        elts=[ast.List(elts=[], ctx=ast.Load())], ctx=ast.Load()
                    ),
                    op=ast.Mult(),
                    right=N,
                ),
                lineno=lineno + len(output),
            ),
            ast.Assign(
                targets=[ast.Name("writes", ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.List(
                        elts=[ast.List(elts=[], ctx=ast.Load())], ctx=ast.Load()
                    ),
                    op=ast.Mult(),
                    right=N,
                ),
                lineno=lineno + len(output) + 1,
            ),
        ]

        writes_tracker = WritesTracker()
        tree_with_writes: ast.FunctionDef = writes_tracker.visit(node)
        print(writes_tracker.write_nodes)

        # ast.fix_missing_locations(tree_with_writes)
        reads_tracker = ReadsTracker(writes_tracker.write_nodes, writes_tracker.array_written_to)
        tree_with_read_writes: ast.FunctionDef = reads_tracker.visit(tree_with_writes)

        get_read_writes_body += tree_with_read_writes.body

        # unrolled_code = ast.unparse(tree_with_writes)
        # print("\n".join(ast.unparse(node) for node in get_read_writes_body))
        # print(unrolled_code)

        # for block in iter(self.cfg):
        #     print("\n".join(ast.unparse(stmt) for stmt in block.statements))
        #     print()

        output.append(
            ast.FunctionDef(
                name="get_read_writes",
                args=ast.arguments(args=[matrix, N], posonlyargs=[], defaults=[], kwonlyargs=[]), 
                body=get_read_writes_body,
                decorator_list=[],
            )
        )

        # todo triton function definition
        # Generate FI (generating W did most of the work already)
        ## Get loop variable form original code
        ### todo: while loops
        ### If you don't use the loop target, this is pointless. Think why to use same name
        # outer_loop = next((b for b in node.body if isinstance(b, (ast.For))), None)
        # if outer_loop is None:
        #     print("todo")
        #     return
        # ## This loop var corresponds to the program id in the parallel output
        # output.append(
        #     ast.Assign(
        #         targets=[ast.Name("i")],
        #         value=ast.Call(
        #             func=ast.Attribute(value=ast.Name("tl"), attr="program_id"),
        #             args=[],
        #             keywords=[ast.keyword("axis", ast.Constant(0))],
        #         ),
        #     )
        # )
        # ## Early return if invalid column
        # output.append(
        #     ast.If(
        #         test=ast.Compare(
        #             left=ast.Name("i"), ops=[ast.GtE()], comparators=[ast.Name("N")]
        #         ),
        #         body=[ast.Return()],
        #         orelse=None,
        #     )
        # )

        # Flag check

        # Operation

        # Flag set

        return output


code = inspect.getsource(simple_sptrsv_csc)
print(f"Original code: \n{code}")
cfg = CFGBuilder().build_from_src("CSC SpTRSV", code)
parallelizer = SptrsvParallelizer(cfg)
test = parallelizer.visit(ast.parse(code))
ast.fix_missing_locations(test)
print(f"Converted code: \n{ast.unparse(test)}")

# code = """
# i = tl.program_body(axis=0)
# """
# for x in ast.parse(code).body:
#     c: ast.Call = x.value
#     print(c.keywords[0].value)
