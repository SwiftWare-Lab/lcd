import ast
from copy import deepcopy
import inspect
from compiler.cfg import CFGBuilder, CFG

from benchmark.sptrsv import simple_sptrsv_csc


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


def generate_tensor_assign(varname, data, torch_alias, dtype):
    return ast.Assign(
        targets=[ast.Name(varname, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=torch_alias,
                attr="as_tensor",
                ctx=ast.Load(),
            ),
            args=[data],
            keywords=[
                ast.keyword(
                    arg="dtype",
                    value=dtype,
                ),
                ast.keyword(
                    arg="device",
                    value=ast.Name("__DEVICE", ctx=ast.Load()),
                ),
            ],
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
        output.append(
            ast.FunctionDef(
                name="__get_read_writes",
                args=ast.arguments(
                    args=[*node.args.args, N],
                    posonlyargs=[],
                    defaults=[],
                    kwonlyargs=[],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name("__reads", ctx=ast.Store())],
                        value=ast.ListComp(
                            elt=ast.Call(
                                func=ast.Name("set", ctx=ast.Load()),
                                args=[],
                                keywords=[],
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
                                func=ast.Name("set", ctx=ast.Load()),
                                args=[],
                                keywords=[],
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
                    *generate_read_write_tracker_body(deepcopy(node)),
                ],
                decorator_list=[],
            )
        )

        # Generate `deps_count` and other needed variables
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
        real_type = ast.Attribute(value=torch_alias, attr="float64", ctx=ast.Load())
        int_type = ast.Attribute(value=torch_alias, attr="int32", ctx=ast.Load())

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

        # Generate code to compute blocksize (max nnz in a column round up to nearest power of 2)
        output.append(
            ast.Assign(
                targets=[ast.Name(id="__max_col_nnz", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="max", ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Attribute(
                                value=matrix,
                                attr="getnnz",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[
                                ast.keyword(arg="axis", value=ast.Constant(value=0))
                            ],
                        )
                    ],
                    keywords=[],
                ),
            )
        )
        output.append(
            ast.Assign(
                targets=[ast.Name(id="__BLOCKSIZE", ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Constant(value=1),
                    op=ast.LShift(),
                    right=ast.Call(
                        func=ast.Attribute(
                            value=ast.BinOp(
                                left=ast.Name(id="__max_col_nnz", ctx=ast.Load()),
                                op=ast.Sub(),
                                right=ast.Constant(value=1),
                            ),
                            attr="bit_length",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                ),
            )
        )

        # Set up tensors to pass to GPU
        output.append(
            generate_tensor_assign(
                "__deps_count",
                ast.Name("__deps_count", ctx=ast.Load()),
                torch_alias,
                int_type,
            )
        )

        output.append(
            generate_tensor_assign(
                "__data",
                ast.Attribute(value=matrix, attr="data", ctx=ast.Load()),
                torch_alias,
                real_type,
            )
        )
        output.append(
            generate_tensor_assign(
                "__rhs",
                rhs,
                torch_alias,
                real_type,
            )
        )
        output.append(
            generate_tensor_assign(
                "__indices",
                ast.Attribute(value=matrix, attr="indices", ctx=ast.Load()),
                torch_alias,
                int_type,
            )
        )
        output.append(
            generate_tensor_assign(
                "__indptr",
                ast.Attribute(value=matrix, attr="indptr", ctx=ast.Load()),
                torch_alias,
                int_type,
            )
        )

        # Generate kernel
        output.append(
            ast.FunctionDef(
                name="__sptrsv_sync_free_csc",
                args=ast.arguments(
                    args=[
                        ast.arg(arg="data_ptr"),
                        ast.arg(arg="indices_ptr"),
                        ast.arg(arg="indptr_ptr"),
                        ast.arg(arg="rhs_vec"),
                        ast.arg(arg="x_ptr"),
                        ast.arg(arg="N"),
                        ast.arg(
                            arg="BLOCKSIZE",
                            annotation=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="constexpr",
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.arg(arg="deps_count"),
                    ],
                    posonlyargs=[],
                    defaults=[],
                    kwonlyargs=[],
                ),
                decorator_list=[
                    ast.Attribute(
                        value=ast.Name(id="triton", ctx=ast.Load()),
                        attr="jit",
                        ctx=ast.Load(),
                    )
                ],
                body=[
                    # Flag checker
                    ast.Assign(
                        targets=[ast.Name(id="col", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="program_id",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[
                                ast.keyword(arg="axis", value=ast.Constant(value=0))
                            ],
                        ),
                    ),
                    ast.Assign(
                        targets=[ast.Name(id="is_valid_col", ctx=ast.Store())],
                        value=ast.Compare(
                            left=ast.Name(id="col", ctx=ast.Load()),
                            ops=[ast.Lt()],
                            comparators=[ast.Name(id="M", ctx=ast.Load())],
                        ),
                    ),
                    ast.If(
                        test=ast.UnaryOp(
                            ast.Not(), ast.Name(id="is_valid_col", ctx=ast.Load())
                        ),
                        body=[ast.Return()],
                        orelse=[],
                    ),
                    # Busy-wait
                    # while tl.atomic_add(deps_count + col, 0) != 1: pass
                    ast.While(
                        test=ast.Compare(
                            left=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="tl", ctx=ast.Load()),
                                    attr="atomic_add",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.BinOp(
                                        left=ast.Name(id="deps_count", ctx=ast.Load()),
                                        op=ast.Add(),
                                        right=ast.Name(id="col", ctx=ast.Load()),
                                    ),
                                    ast.Constant(value=0),
                                ],
                                keywords=[],
                            ),
                            ops=[ast.NotEq()],
                            comparators=[ast.Constant(value=1)],
                        ),
                        body=[ast.Pass()],
                        orelse=[],
                    ),
                    # Operation
                    # start = tl.load(indptr_ptr + col, mask=is_valid_col, other=0)
                    ast.Assign(
                        targets=[ast.Name(id="start", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="load",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id="indptr_ptr", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="col", ctx=ast.Load()),
                                )
                            ],
                            keywords=[
                                ast.keyword(
                                    arg="mask",
                                    value=ast.Name(id="is_valid_col", ctx=ast.Load()),
                                ),
                                ast.keyword(arg="other", value=ast.Constant(value=0)),
                            ],
                        ),
                    ),
                    # end = tl.load(indptr_ptr + col + 1, mask=is_valid_col, other=0)
                    ast.Assign(
                        targets=[ast.Name(id="end", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="load",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.BinOp(
                                        left=ast.Name(id="indptr_ptr", ctx=ast.Load()),
                                        op=ast.Add(),
                                        right=ast.Name(id="col", ctx=ast.Load()),
                                    ),
                                    op=ast.Add(),
                                    right=ast.Constant(value=1),
                                )
                            ],
                            keywords=[
                                ast.keyword(
                                    arg="mask",
                                    value=ast.Name(id="is_valid_col", ctx=ast.Load()),
                                ),
                                ast.keyword(arg="other", value=ast.Constant(value=0)),
                            ],
                        ),
                    ),
                    # x_col_sol = (tl.load(rhs_vec + col) - tl.load(x_ptr + col)) / tl.load(data_ptr + start)
                    ast.Assign(
                        targets=[ast.Name(id="x_col_sol", ctx=ast.Store())],
                        value=ast.BinOp(
                            left=ast.BinOp(
                                left=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="tl", ctx=ast.Load()),
                                        attr="load",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.BinOp(
                                            left=ast.Name(id="rhs_vec", ctx=ast.Load()),
                                            op=ast.Add(),
                                            right=ast.Name(id="col", ctx=ast.Load()),
                                        )
                                    ],
                                    keywords=[],
                                ),
                                op=ast.Sub(),
                                right=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="tl", ctx=ast.Load()),
                                        attr="load",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.BinOp(
                                            left=ast.Name(id="x_ptr", ctx=ast.Load()),
                                            op=ast.Add(),
                                            right=ast.Name(id="col", ctx=ast.Load()),
                                        )
                                    ],
                                    keywords=[],
                                ),
                            ),
                            op=ast.Div(),
                            right=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="tl", ctx=ast.Load()),
                                    attr="load",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.BinOp(
                                        left=ast.Name(id="data_ptr", ctx=ast.Load()),
                                        op=ast.Add(),
                                        right=ast.Name(id="start", ctx=ast.Load()),
                                    )
                                ],
                                keywords=[],
                            ),
                        ),
                    ),
                    # tl.store(x_ptr + col, x_col_sol)
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="store",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id="x_ptr", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="col", ctx=ast.Load()),
                                ),
                                ast.Name(id="x_col_sol", ctx=ast.Load()),
                            ],
                            keywords=[],
                        )
                    ),
                    # thread_idx = tl.arange(0, BLOCKSIZE)
                    ast.Assign(
                        targets=[ast.Name(id="thread_idx", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="arange",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Constant(value=0),
                                ast.Name(id="BLOCKSIZE", ctx=ast.Load()),
                            ],
                            keywords=[],
                        ),
                    ),
                    # block_start = start + 1 + thread_idx
                    ast.Assign(
                        targets=[ast.Name(id="block_start", ctx=ast.Store())],
                        value=ast.BinOp(
                            left=ast.BinOp(
                                left=ast.Name(id="start", ctx=ast.Load()),
                                op=ast.Add(),
                                right=ast.Constant(value=1),
                            ),
                            op=ast.Add(),
                            right=ast.Name(id="thread_idx", ctx=ast.Load()),
                        ),
                    ),
                    # mask = block_start < end
                    ast.Assign(
                        targets=[ast.Name(id="mask", ctx=ast.Store())],
                        value=ast.Compare(
                            left=ast.Name(id="block_start", ctx=ast.Load()),
                            ops=[ast.Lt()],
                            comparators=[ast.Name(id="end", ctx=ast.Load())],
                        ),
                    ),
                    # data_col = tl.load(data_ptr + block_start, mask=mask, other=0)
                    ast.Assign(
                        targets=[ast.Name(id="data_col", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="load",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id="data_ptr", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="block_start", ctx=ast.Load()),
                                )
                            ],
                            keywords=[
                                ast.keyword(
                                    arg="mask",
                                    value=ast.Name(id="mask", ctx=ast.Load()),
                                ),
                                ast.keyword(arg="other", value=ast.Constant(value=0)),
                            ],
                        ),
                    ),
                    # partial_sum = data_col * x_col_sol
                    ast.Assign(
                        targets=[ast.Name(id="partial_sum", ctx=ast.Store())],
                        value=ast.BinOp(
                            left=ast.Name(id="data_col", ctx=ast.Load()),
                            op=ast.Mult(),
                            right=ast.Name(id="x_col_sol", ctx=ast.Load()),
                        ),
                    ),
                    # indices_col = tl.load(indices_ptr + block_start, mask=mask, other=0)
                    ast.Assign(
                        targets=[ast.Name(id="indices_col", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="load",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id="indices_ptr", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="block_start", ctx=ast.Load()),
                                )
                            ],
                            keywords=[
                                ast.keyword(
                                    arg="mask",
                                    value=ast.Name(id="mask", ctx=ast.Load()),
                                ),
                                ast.keyword(arg="other", value=ast.Constant(value=0)),
                            ],
                        ),
                    ),
                    # valid_indices = mask
                    ast.Assign(
                        targets=[ast.Name(id="valid_indices", ctx=ast.Store())],
                        value=ast.Name(id="mask", ctx=ast.Load()),
                    ),
                    # tl.atomic_add(x_ptr + indices_col, partial_sum, mask=valid_indices)
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="atomic_add",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id="x_ptr", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="indices_col", ctx=ast.Load()),
                                ),
                                ast.Name(id="partial_sum", ctx=ast.Load()),
                            ],
                            keywords=[
                                ast.keyword(
                                    arg="mask",
                                    value=ast.Name(id="valid_indices", ctx=ast.Load()),
                                )
                            ],
                        )
                    ),
                    # tl.atomic_add(deps_count + indices_col, -1, mask=valid_indices)
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="atomic_add",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id="deps_count", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="indices_col", ctx=ast.Load()),
                                ),
                                ast.UnaryOp(
                                    op=ast.USub(), operand=ast.Constant(value=1)
                                ),
                            ],
                            keywords=[
                                ast.keyword(
                                    arg="mask",
                                    value=ast.Name(id="valid_indices", ctx=ast.Load()),
                                )
                            ],
                        )
                    ),
                ],
                returns=None,
            )
        )

        # Generate kernel launcher
        output.append(
            ast.FunctionDef(
                name="__launch_sptrsv_sync_free_csc",
                args=ast.arguments(
                    args=[
                        ast.Name("data", ast.Load()),
                        ast.Name("indices", ast.Load()),
                        ast.Name("indptr", ast.Load()),
                        ast.Name("rhs", ast.Load()),
                        N,
                        ast.Name(id="deps_count", ctx=ast.Load()),
                        ast.Name(id="blocksize", ctx=ast.Load()),
                    ],
                    posonlyargs=[],
                    defaults=[],
                    kwonlyargs=[],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name("x", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=torch_alias, attr="zeros", ctx=ast.Load()
                            ),
                            args=[N],
                            keywords=[
                                ast.keyword(arg="dtype", value=real_type),
                                ast.keyword(arg="device", value=device),
                            ],
                        ),
                    ),
                    ast.Assign(
                        targets=[ast.Name("grid", ctx=ast.Store())],
                        value=ast.Tuple(elts=[N], ctx=ast.Load()),
                    ),
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Subscript(
                                value=ast.Name(
                                    id="__sptrsv_sync_free_csc", ctx=ast.Load()
                                ),
                                slice=ast.Name(id="grid", ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Name(id="data", ctx=ast.Load()),
                                ast.Name(id="indices", ctx=ast.Load()),
                                ast.Name(id="indptr", ctx=ast.Load()),
                                ast.Name(id="rhs", ctx=ast.Load()),
                                ast.Name(id="x", ctx=ast.Load()),
                                N,
                                ast.Name(id="deps_count", ctx=ast.Load()),
                                ast.Name(id="blocksize", ctx=ast.Load()),
                            ],
                            keywords=[],
                        )
                    ),
                    ast.Return(ast.Name("x", ctx=ast.Load())),
                ],
                decorator_list=[],
            )
        )

        return output


code = inspect.getsource(simple_sptrsv_csc)
print(f"Original code: \n{code}")
cfg = CFGBuilder().build_from_src("CSC SpTRSV", code)
parallelizer = SptrsvParallelizer(cfg)
test = parallelizer.visit(ast.parse(code))
ast.fix_missing_locations(test)
print(f"Converted code: \n{ast.unparse(test)}")
