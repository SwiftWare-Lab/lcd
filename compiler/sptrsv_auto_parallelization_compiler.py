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
        self.write_array_id = None

    def _generate_add_write_stmt(self, write_index):
        return ast.Expr(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name("__writes", ctx=ast.Load()),
                        slice=self.outer_loop_var,
                    ),
                    attr="add",
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
            self.write_array_id = node.targets[0].value.id
            return [node, self._generate_add_write_stmt(index)]

        return node

    def visit_AugAssign(self, node):
        # * Assume 1 target (so no a, b = c, d)
        if self.outer_loop_var and isinstance(node.target, ast.Subscript):
            index = node.target.slice
            self.write_nodes.add(node.target)
            self.write_array_id = node.target.value.id
            return [node, self._generate_add_write_stmt(index)]

        return node


class ReadsTracker(ast.NodeTransformer):
    class _SubscriptVisitor(ast.NodeTransformer):
        def __init__(self, outer_ref):
            self.outer_ref = outer_ref
            self.added_reads_from_stmt = []

        def visit_Subscript(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == self.outer_ref.write_array_id
            ):
                index = node.slice
                self.added_reads_from_stmt.append(self._generate_add_read_stmt(index))
            return node

        def flush_stmts(self):
            stmts = self.added_reads_from_stmt
            self.added_reads_from_stmt = []
            return stmts

        def _generate_add_read_stmt(self, read_index):
            return ast.Expr(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Subscript(
                            value=ast.Name("__reads", ctx=ast.Load()),
                            slice=self.outer_ref.outer_loop_var,
                        ),
                        attr="add",
                        ctx=ast.Load(),
                    ),
                    args=[read_index],
                    keywords=[],
                )
            )

    def __init__(self, write_array_id):
        self.write_array_id = write_array_id
        self.outer_loop_var = None
        self.__internal_tracker = self._SubscriptVisitor(self)

    def visit_For(self, node):
        if not self.outer_loop_var:
            self.outer_loop_var = node.target
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if self.outer_loop_var:
            self.__internal_tracker.visit(node.value)
        return [node, *self.__internal_tracker.flush_stmts()]

    def visit_AugAssign(self, node):
        if self.outer_loop_var:
            self.__internal_tracker.visit(node)
        return [node, *self.__internal_tracker.flush_stmts()]

    # Sneak this transformation in here, it needs to happen eventually
    def visit_Return(self, node):
        return ast.Return(
            ast.Tuple(
                elts=[
                    ast.Name("__reads", ctx=ast.Load()),
                    ast.Name("__writes", ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            )
        )


def generate_read_write_tracker_body(node: ast.FunctionDef) -> list[ast.stmt]:
    writes_tracker = WritesTracker()
    writes_tracker.visit(node)
    reads_tracker = ReadsTracker(writes_tracker.write_array_id)
    reads_tracker.visit(node)
    return node.body


def generate_tensor_assign(varname, data, dtype=None):
    if dtype:
        keywords = [
            ast.keyword(
                arg="dtype",
                value=dtype,
            ),
            ast.keyword(
                arg="device",
                value=ast.Name("__DEVICE", ctx=ast.Load()),
            ),
        ]
    else:
        keywords = [
            ast.keyword(
                arg="device",
                value=ast.Name("__DEVICE", ctx=ast.Load()),
            )
        ]

    return ast.Assign(
        targets=[ast.Name(varname, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name("torch", ctx=ast.Load()),
                attr="as_tensor",
                ctx=ast.Load(),
            ),
            args=[data],
            keywords=keywords,
        ),
    )


class SptrsvParallelizer(ast.NodeTransformer):
    def __init__(self, cfg: CFG):
        self.cfg = cfg

    def visit_FunctionDef(self, node):
        """Takes sequential CSC SpTRSV function, outputs parallel Triton kernel"""
        # todo: while loops
        output = [
            ast.Import(
                [
                    ast.alias("triton"),
                    ast.alias("triton.language", "tl"),
                    ast.alias("torch"),
                ]
            )
        ]

        # * Assume first arg is the sparse matrix and second arg is the rhs of the sptrsv equation
        matrix = node.args.args[0]
        rhs = node.args.args[1]

        N = ast.Name("__N", ctx=ast.Load())

        # Function that creates Read and Write sets
        lineno = node.lineno
        get_read_writes_body = [
            ast.Assign(
                targets=[ast.Name("__reads", ctx=ast.Store())],
                value=ast.ListComp(
                    elt=ast.Call(
                        func=ast.Name("set", ctx=ast.Load()), args=[], keywords=[]
                    ),
                    generators=[
                        ast.comprehension(
                            target=ast.Name("_", ctx=ast.Store()),
                            iter=ast.Call(
                                func=ast.Name("range", ctx=ast.Load()),
                                args=[N],
                                keywords=[],
                            ),
                            ifs=[],
                            is_async=0,
                        )
                    ],
                ),
                lineno=lineno + len(output),
            ),
            ast.Assign(
                targets=[ast.Name("__writes", ctx=ast.Store())],
                value=ast.ListComp(
                    elt=ast.Call(
                        func=ast.Name("set", ctx=ast.Load()), args=[], keywords=[]
                    ),
                    generators=[
                        ast.comprehension(
                            target=ast.Name("_", ctx=ast.Store()),
                            iter=ast.Call(
                                func=ast.Name("range", ctx=ast.Load()),
                                args=[N],
                                keywords=[],
                            ),
                            ifs=[],
                            is_async=0,
                        )
                    ],
                ),
                lineno=lineno + len(output) + 1,
            ),
        ]
        get_read_writes_body += generate_read_write_tracker_body(deepcopy(node))

        output.append(
            ast.FunctionDef(
                name="__get_read_writes",
                args=ast.arguments(
                    args=[*node.args.args, N],
                    posonlyargs=[],
                    defaults=[],
                    kwonlyargs=[],
                ),
                body=get_read_writes_body,
                decorator_list=[],
            )
        )

        # Generate `deps_count`
        # Todo figure out where args come from
        output.append(
            ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name("__reads", ctx=ast.Load()),
                            ast.Name("__writes", ctx=ast.Load()),
                        ],
                        ctx=ast.Load(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name("__get_read_writes", ctx=ast.Load()),
                    args=[*node.args.args, N],
                    keywords=[],
                ),
            )
        )

        torch_alias = ast.Name("torch", ctx=ast.Load())
        device = ast.Name("__DEVICE", ctx=ast.Load())
        output.append(
            ast.Assign(
                targets=[ast.Name("__DEVICE", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=torch_alias,
                        attr="device",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant("cuda:0")],
                    keywords=[],
                ),
            )
        )

        output.append(
            ast.Assign(
                targets=[ast.Name("__deps_count", ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
        )

        output.append(
            ast.For(
                target=ast.Name(id="i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[N],
                    keywords=[],
                ),
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="__deps_count", ctx=ast.Load()),
                                attr="append",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Call(
                                    func=ast.Name(id="sum", ctx=ast.Load()),
                                    args=[
                                        ast.GeneratorExp(
                                            elt=ast.Constant(value=1),
                                            generators=[
                                                ast.comprehension(
                                                    target=ast.Tuple(
                                                        elts=[
                                                            ast.Name(
                                                                id="j", ctx=ast.Store()
                                                            ),
                                                            ast.Name(
                                                                id="x", ctx=ast.Store()
                                                            ),
                                                        ],
                                                        ctx=ast.Store(),
                                                    ),
                                                    iter=ast.Call(
                                                        func=ast.Name(
                                                            id="enumerate",
                                                            ctx=ast.Load(),
                                                        ),
                                                        args=[
                                                            ast.Name(
                                                                id="writes",
                                                                ctx=ast.Load(),
                                                            )
                                                        ],
                                                        keywords=[],
                                                    ),
                                                    ifs=[
                                                        ast.Compare(
                                                            left=ast.Name(
                                                                id="j", ctx=ast.Load()
                                                            ),
                                                            ops=[ast.NotEq()],
                                                            comparators=[
                                                                ast.Name(
                                                                    id="i",
                                                                    ctx=ast.Load(),
                                                                )
                                                            ],
                                                        ),
                                                        ast.Compare(
                                                            left=ast.Name(
                                                                id="i", ctx=ast.Load()
                                                            ),
                                                            ops=[ast.In()],
                                                            comparators=[
                                                                ast.Name(
                                                                    id="x",
                                                                    ctx=ast.Load(),
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                    is_async=0,
                                                )
                                            ],
                                        )
                                    ],
                                    keywords=[],
                                )
                            ],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            )
        )

        # Set up tensors to pass to launcher
        output.append(
            generate_tensor_assign(
                varname="__deps_count",
                data=ast.Name("__deps_count", ctx=ast.Load()),
                dtype=ast.Attribute(value=torch_alias, attr="int32", ctx=ast.Load()),
            )
        )

        output.append(
            generate_tensor_assign(
                varname="__data",
                data=ast.Attribute(value=matrix, attr="data", ctx=ast.Load()),
            )
        )
        output.append(
            generate_tensor_assign(
                varname="__rhs",
                data=rhs,
            )
        )
        output.append(
            generate_tensor_assign(
                varname="__indices",
                data=ast.Attribute(value=matrix, attr="indices", ctx=ast.Load()),
                dtype=ast.Attribute(value=torch_alias, attr="int32", ctx=ast.Load()),
            )
        )
        output.append(
            generate_tensor_assign(
                varname="__indptr",
                data=ast.Attribute(value=matrix, attr="indptr", ctx=ast.Load()),
                dtype=ast.Attribute(value=torch_alias, attr="int32", ctx=ast.Load()),
            )
        )

        # Generate kernel launcher
        launcher_body = []

        ## Get loop variable form original code
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
