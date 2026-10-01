"""Lower one annotated Python loop to a Numba-CUDA kernel.

This stage extracts the selected loop body, records the removed host
statements, and emits both a CUDA kernel and its host-side launcher.

In sync-free mode, ``dag_ptr`` and ``dag_index`` are a CSR representation of
the DAG's successor lists.  A completed task decrements the dependency count
of each successor.

In wavefront mode, ``wavefronts`` stores all wavefront nodes in one compact
array. ``wave_start`` and ``wave_end`` select the current wavefront.
"""

import ast
from copy import deepcopy
from dataclasses import dataclass
import inspect
import textwrap


class CUDALoweringError(Exception):
    """Base error for unsupported CUDA lowering."""


class UnsupportedAtomicOperation(CUDALoweringError):
    """Raised when a reduction has no direct Numba-CUDA atomic lowering."""


KNOWN_LOWERING_CHALLENGES = (
    "Thread mapping within blocks",
    "DAG schedule-to-block mapping",
    "Host/device allocation and transfers",
    "Grid bounds checks and synchronization",
    "Atomic operation and dtype support",
    "Stable statement IDs across transformations",
    "Busy-wait forward progress and global-memory publication ordering",
)

CUDA_EXECUTION_MODES = frozenset({"plain", "sync_free", "wavefront"})
SYNC_FREE_ARGUMENTS = ("dep_count", "dag_ptr", "dag_index")
WAVEFRONT_ARGUMENTS = ("wavefronts", "wave_start", "wave_end")


@dataclass
class CUDALoweringResult:
    module_ast: ast.Module
    kernel_ast: ast.FunctionDef
    launcher_ast: ast.FunctionDef
    kernel_name: str
    launcher_name: str
    kernel_arguments: list[str]
    launcher_arguments: list[str]
    lifted_variables: list[str]
    host_prelude: list[ast.stmt]
    host_epilogue: list[ast.stmt]
    atomic_statement_ids: set[str]
    execution_mode: str
    auxiliary_arguments: list[str]
    challenges: tuple[str, ...] = KNOWN_LOWERING_CHALLENGES

    @property
    def source(self):
        return ast.unparse(self.module_ast)


@dataclass(frozen=True)
class _ReductionSite:
    statement_id: str
    operator: str
    location: tuple


class _LoadedNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


class _StoredNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names = []

    def _add_target(self, target):
        if isinstance(target, ast.Name) and target.id not in self.names:
            self.names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._add_target(element)

    def visit_Assign(self, node):
        for target in node.targets:
            self._add_target(target)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node):
        self._add_target(node.target)
        if node.value is not None:
            self.generic_visit(node.value)


class _AtomicReductionLowerer(ast.NodeTransformer):
    ATOMIC_OPERATIONS = {
        "+": "add",
        "-": "sub",
        "&": "and_",
        "|": "or_",
        "^": "xor",
    }

    def __init__(self, reductions):
        self.reductions = reductions
        self.lowered_statement_ids = set()

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        reduction = self.reductions.get(_source_location(node))
        if reduction is None:
            return node

        statement_id = reduction.statement_id
        operator = reduction.operator
        atomic_operation = self.ATOMIC_OPERATIONS.get(operator)
        if atomic_operation is None:
            raise UnsupportedAtomicOperation(
                f"Reduction {statement_id} uses {operator!r}, which has no "
                "direct atomic lowering in this stage."
            )

        array, index = self._atomic_target(node.target)
        atomic_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="cuda", ctx=ast.Load()),
                        attr="atomic",
                        ctx=ast.Load(),
                    ),
                    attr=atomic_operation,
                    ctx=ast.Load(),
                ),
                args=[array, index, node.value],
                keywords=[],
            )
        )
        self.lowered_statement_ids.add(statement_id)
        return ast.copy_location(atomic_call, node)

    @staticmethod
    def _atomic_target(target):
        if not isinstance(target, ast.Subscript):
            raise CUDALoweringError(
                "Numba-CUDA atomics require an indexed array target."
            )

        indices = []
        current = target
        while isinstance(current, ast.Subscript):
            indices.append(deepcopy(current.slice))
            current = current.value
        indices.reverse()

        if len(indices) == 1:
            index = indices[0]
        else:
            index = ast.Tuple(elts=indices, ctx=ast.Load())
        return deepcopy(current), index


class CUDALowerer:
    def __init__(
            self,
            loop_iterator,
            reductions=None,
            kernel_name=None,
            execution_mode="plain",
            sync_free_ready_count=0,
            wavefront_threads_per_block=128,
    ):
        self.loop_iterator = loop_iterator
        self.reductions = _normalize_reductions(reductions)
        self.kernel_name = kernel_name
        self.execution_mode = _normalize_execution_mode(execution_mode)
        self.sync_free_ready_count = sync_free_ready_count
        self.wavefront_threads_per_block = wavefront_threads_per_block
        if not isinstance(sync_free_ready_count, int):
            raise CUDALoweringError(
                "The sync-free readiness count must be an integer."
            )
        if (
                not isinstance(wavefront_threads_per_block, int)
                or wavefront_threads_per_block <= 0
        ):
            raise CUDALoweringError(
                "The wavefront thread-block size must be a positive integer."
            )

    def lower(self, code):
        module = _to_module(code)
        functions = [
            node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
            raise CUDALoweringError(
                "CUDA lowering expects exactly one synchronous function."
            )

        function = deepcopy(functions[0])
        loop_index, target_loop = self._target_loop(function)
        host_prelude = deepcopy(function.body[:loop_index])
        host_epilogue = deepcopy(function.body[loop_index + 1:])

        lifted_variables = self._lifted_variables(function, target_loop, host_prelude)
        kernel_arguments = [
            argument.arg
            for argument in function.args.args
        ]
        kernel_arguments.extend(
            variable
            for variable in lifted_variables
            if variable not in kernel_arguments
        )
        auxiliary_arguments = []
        if self.execution_mode == "sync_free":
            auxiliary_arguments = list(SYNC_FREE_ARGUMENTS)
        elif self.execution_mode == "wavefront":
            auxiliary_arguments = list(WAVEFRONT_ARGUMENTS)
        if auxiliary_arguments:
            kernel_arguments.extend(
                name
                for name in auxiliary_arguments
                if name not in kernel_arguments
            )

        atomic_lowerer = _AtomicReductionLowerer(self.reductions)
        lowered_loop_body = [
            atomic_lowerer.visit(deepcopy(statement))
            for statement in target_loop.body
        ]
        if self.execution_mode == "wavefront":
            kernel_body = self._wavefront_body(
                target_loop,
                lowered_loop_body,
            )
        else:
            kernel_body = [self._block_iterator_assignment()]
            if self.execution_mode == "sync_free":
                kernel_body.append(self._sync_free_wait())
                # Acquire predecessor outputs after observing the ready count.
                kernel_body.append(self._device_fence())
            kernel_body.extend(lowered_loop_body)
        if self.execution_mode == "sync_free":
            kernel_body.extend(self._sync_free_release())
        missing_reductions = (
            {
                reduction.statement_id
                for reduction in self.reductions.values()
            }
            - atomic_lowerer.lowered_statement_ids
        )
        if missing_reductions:
            raise CUDALoweringError(
                "Reduction statements were not found in the selected loop: "
                + ", ".join(sorted(missing_reductions))
            )

        kernel_name = self.kernel_name or function.name + "_cuda_kernel"
        kernel = ast.FunctionDef(
            name=kernel_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in kernel_arguments],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=kernel_body,
            decorator_list=[
                ast.Attribute(
                    value=ast.Name(id="cuda", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load(),
                )
            ],
            returns=None,
            type_comment=None,
        )
        launcher_name = function.name + "_cuda_launch"
        launcher, launcher_arguments = self._launcher(
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
        )
        cuda_import = ast.ImportFrom(
            module="numba",
            names=[ast.alias(name="cuda")],
            level=0,
        )
        lowered_module = ast.fix_missing_locations(
            ast.Module(
                body=[cuda_import, kernel, launcher],
                type_ignores=[],
            )
        )
        return CUDALoweringResult(
            module_ast=lowered_module,
            kernel_ast=kernel,
            launcher_ast=launcher,
            kernel_name=kernel_name,
            launcher_name=launcher_name,
            kernel_arguments=kernel_arguments,
            launcher_arguments=launcher_arguments,
            lifted_variables=lifted_variables,
            host_prelude=host_prelude,
            host_epilogue=host_epilogue,
            atomic_statement_ids=atomic_lowerer.lowered_statement_ids,
            execution_mode=self.execution_mode,
            auxiliary_arguments=auxiliary_arguments,
        )

    def _launcher(
            self,
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
    ):
        if self.execution_mode == "wavefront":
            return self._wavefront_launcher(
                kernel_name,
                kernel_arguments,
                launcher_name,
            )
        return self._single_launch_launcher(
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
        )

    def _single_launch_launcher(
            self,
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
    ):
        launcher_arguments = list(kernel_arguments)
        if self.execution_mode == "sync_free":
            block_count = ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id="dep_count", ctx=ast.Load())],
                keywords=[],
            )
        else:
            block_count = self._block_task_count(target_loop)

        launcher = ast.FunctionDef(
            name=launcher_name,
            args=_function_arguments(launcher_arguments),
            body=[
                ast.Assign(
                    targets=[
                        ast.Name(id="_blocks_per_grid", ctx=ast.Store())
                    ],
                    value=block_count,
                ),
                ast.Assign(
                    targets=[
                        ast.Name(id="_threads_per_block", ctx=ast.Store())
                    ],
                    # One block is one task in the current lowering.
                    value=ast.Constant(value=1),
                ),
                _kernel_launch(
                    kernel_name,
                    ast.Name(id="_blocks_per_grid", ctx=ast.Load()),
                    ast.Name(id="_threads_per_block", ctx=ast.Load()),
                    kernel_arguments,
                ),
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return launcher, launcher_arguments

    def _wavefront_launcher(
            self,
            kernel_name,
            kernel_arguments,
            launcher_name,
    ):
        base_arguments = [
            argument
            for argument in kernel_arguments
            if argument not in {"wave_start", "wave_end"}
        ]
        launcher_arguments = base_arguments + ["wavefront_ptr"]
        kernel_base_arguments = [
            argument
            for argument in kernel_arguments
            if argument not in {"wave_start", "wave_end"}
        ]
        launch_arguments = kernel_base_arguments + [
            "_wave_start",
            "_wave_end",
        ]

        wavefront_loop = ast.For(
            target=ast.Name(id="_wavefront", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[
                                ast.Name(
                                    id="wavefront_ptr",
                                    ctx=ast.Load(),
                                )
                            ],
                            keywords=[],
                        ),
                        op=ast.Sub(),
                        right=ast.Constant(value=1),
                    )
                ],
                keywords=[],
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Name(id="_wave_start", ctx=ast.Store())
                    ],
                    value=ast.Subscript(
                        value=ast.Name(
                            id="wavefront_ptr",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Name(
                            id="_wavefront",
                            ctx=ast.Load(),
                        ),
                        ctx=ast.Load(),
                    ),
                ),
                ast.Assign(
                    targets=[
                        ast.Name(id="_wave_end", ctx=ast.Store())
                    ],
                    value=ast.Subscript(
                        value=ast.Name(
                            id="wavefront_ptr",
                            ctx=ast.Load(),
                        ),
                        slice=ast.BinOp(
                            left=ast.Name(
                                id="_wavefront",
                                ctx=ast.Load(),
                            ),
                            op=ast.Add(),
                            right=ast.Constant(value=1),
                        ),
                        ctx=ast.Load(),
                    ),
                ),
                ast.Assign(
                    targets=[
                        ast.Name(id="_wave_size", ctx=ast.Store())
                    ],
                    value=ast.BinOp(
                        left=ast.Name(id="_wave_end", ctx=ast.Load()),
                        op=ast.Sub(),
                        right=ast.Name(id="_wave_start", ctx=ast.Load()),
                    ),
                ),
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id="_wave_size", ctx=ast.Load()),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=0)],
                    ),
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Name(
                                    id="_blocks_per_grid",
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Call(
                                func=ast.Name(id="int", ctx=ast.Load()),
                                args=[
                                    ast.BinOp(
                                        left=ast.BinOp(
                                            left=ast.Name(
                                                id="_wave_size",
                                                ctx=ast.Load(),
                                            ),
                                            op=ast.Add(),
                                            right=ast.BinOp(
                                                left=ast.Name(
                                                    id="_threads_per_block",
                                                    ctx=ast.Load(),
                                                ),
                                                op=ast.Sub(),
                                                right=ast.Constant(value=1),
                                            ),
                                        ),
                                        op=ast.FloorDiv(),
                                        right=ast.Name(
                                            id="_threads_per_block",
                                            ctx=ast.Load(),
                                        ),
                                    ),
                                ],
                                keywords=[],
                            ),
                        ),
                        _kernel_launch(
                            kernel_name,
                            ast.Name(
                                id="_blocks_per_grid",
                                ctx=ast.Load(),
                            ),
                            ast.Name(
                                id="_threads_per_block",
                                ctx=ast.Load(),
                            ),
                            launch_arguments,
                        ),
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(
                                        id="cuda",
                                        ctx=ast.Load(),
                                    ),
                                    attr="synchronize",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            )
                        ),
                    ],
                    orelse=[],
                ),
            ],
            orelse=[],
        )
        launcher = ast.FunctionDef(
            name=launcher_name,
            args=_function_arguments(launcher_arguments),
            body=[
                ast.Assign(
                    targets=[
                        ast.Name(id="_threads_per_block", ctx=ast.Store())
                    ],
                    value=ast.Constant(
                        value=self.wavefront_threads_per_block
                    ),
                ),
                wavefront_loop,
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return launcher, launcher_arguments

    def _block_task_count(self, target_loop):
        loop_range = target_loop.iter
        if not (
                isinstance(loop_range, ast.Call)
                and isinstance(loop_range.func, ast.Name)
                and loop_range.func.id == "range"
                and not loop_range.keywords
        ):
            raise CUDALoweringError(
                "CUDA launch generation requires a range loop."
            )
        if len(loop_range.args) == 1:
            return deepcopy(loop_range.args[0])
        if (
                len(loop_range.args) == 2
                and isinstance(loop_range.args[0], ast.Constant)
                and loop_range.args[0].value == 0
        ):
            return deepcopy(loop_range.args[1])
        raise CUDALoweringError(
            "Block-mapped CUDA launch generation currently requires "
            "range(stop) or range(0, stop)."
        )

    def _block_iterator_assignment(self):
        return ast.Assign(
            targets=[
                ast.Name(id=self.loop_iterator, ctx=ast.Store())
            ],
            value=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="cuda", ctx=ast.Load()),
                    attr="blockIdx",
                    ctx=ast.Load(),
                ),
                attr="x",
                ctx=ast.Load(),
            ),
        )

    def _wavefront_body(self, target_loop, lowered_loop_body):
        wavefront_location = ast.Name(
            id="_wavefront_location",
            ctx=ast.Load(),
        )
        wavefront_index = ast.BinOp(
            left=ast.Name(id="wave_start", ctx=ast.Load()),
            op=ast.Add(),
            right=wavefront_location,
        )
        return [
            ast.Assign(
                targets=[
                    ast.Name(
                        id="_wavefront_location",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="cuda", ctx=ast.Load()),
                        attr="grid",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=1)],
                    keywords=[],
                ),
            ),
            ast.If(
                test=ast.Compare(
                    left=deepcopy(wavefront_index),
                    ops=[ast.Lt()],
                    comparators=[
                        ast.Name(id="wave_end", ctx=ast.Load())
                    ],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Name(
                                id=self.loop_iterator,
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Subscript(
                            value=ast.Name(
                                id="wavefronts",
                                ctx=ast.Load(),
                            ),
                            slice=deepcopy(wavefront_index),
                            ctx=ast.Load(),
                        ),
                    ),
                    ast.If(
                        test=self._loop_bounds_guard(target_loop),
                        body=lowered_loop_body,
                        orelse=[],
                    ),
                ],
                orelse=[],
            ),
        ]

    def _loop_bounds_guard(self, target_loop):
        loop_range = target_loop.iter
        if not (
                isinstance(loop_range, ast.Call)
                and isinstance(loop_range.func, ast.Name)
                and loop_range.func.id == "range"
                and not loop_range.keywords
                and len(loop_range.args) in {1, 2}
        ):
            raise CUDALoweringError(
                "Wavefront lowering currently requires range(stop) or "
                "range(start, stop) on the selected loop."
            )

        iterator = ast.Name(id=self.loop_iterator, ctx=ast.Load())
        if len(loop_range.args) == 1:
            return ast.Compare(
                left=iterator,
                ops=[ast.Lt()],
                comparators=[deepcopy(loop_range.args[0])],
            )
        return ast.Compare(
            left=deepcopy(loop_range.args[0]),
            ops=[ast.LtE(), ast.Lt()],
            comparators=[iterator, deepcopy(loop_range.args[1])],
        )

    def _sync_free_wait(self):
        return ast.While(
            test=ast.Compare(
                left=_cuda_atomic_add(
                    "dep_count",
                    ast.Name(id=self.loop_iterator, ctx=ast.Load()),
                    ast.Constant(value=0),
                ),
                ops=[ast.NotEq()],
                comparators=[
                    ast.Constant(value=self.sync_free_ready_count)
                ],
            ),
            body=[ast.Pass()],
            orelse=[],
        )

    @staticmethod
    def _device_fence():
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="cuda", ctx=ast.Load()),
                    attr="threadfence",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )

    def _sync_free_release(self):
        # Do not reuse the task iterator as the adjacency-list iterator: the
        # task ID is still needed to select its successor-list bounds.
        task = ast.Name(id=self.loop_iterator, ctx=ast.Load())
        return [
            # Publish the row output before successors see a ready count.
            self._device_fence(),
            ast.Assign(
                targets=[ast.Name(id="_dag_start", ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id="dag_ptr", ctx=ast.Load()),
                    slice=task,
                    ctx=ast.Load(),
                ),
            ),
            ast.Assign(
                targets=[ast.Name(id="_dag_end", ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id="dag_ptr", ctx=ast.Load()),
                    slice=ast.BinOp(
                        left=task,
                        op=ast.Add(),
                        right=ast.Constant(value=1),
                    ),
                    ctx=ast.Load(),
                ),
            ),
            ast.For(
                target=ast.Name(id="_dag_position", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[
                        ast.Name(id="_dag_start", ctx=ast.Load()),
                        ast.Name(id="_dag_end", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Name(id="_dag_successor", ctx=ast.Store())
                        ],
                        value=ast.Subscript(
                            value=ast.Name(id="dag_index", ctx=ast.Load()),
                            slice=ast.Name(
                                id="_dag_position",
                                ctx=ast.Load(),
                            ),
                            ctx=ast.Load(),
                        ),
                    ),
                    ast.Expr(
                        value=_cuda_atomic_add(
                            "dep_count",
                            ast.Name(id="_dag_successor", ctx=ast.Load()),
                            ast.Constant(value=-1),
                        )
                    ),
                ],
                orelse=[],
            ),
        ]

    def _target_loop(self, function):
        matches = [
            (index, statement)
            for index, statement in enumerate(function.body)
            if (
                isinstance(statement, ast.For)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == self.loop_iterator
            )
        ]
        if len(matches) != 1:
            raise CUDALoweringError(
                "The selected iterator must identify exactly one top-level "
                f"for loop; found {len(matches)} for {self.loop_iterator!r}."
            )
        return matches[0]

    def _lifted_variables(self, function, target_loop, host_prelude):
        loaded = _LoadedNameCollector()
        for statement in target_loop.body:
            loaded.visit(statement)

        stored = _StoredNameCollector()
        for statement in host_prelude:
            stored.visit(statement)

        argument_names = {
            argument.arg for argument in function.args.args
        }
        return [
            name
            for name in stored.names
            if name in loaded.names
            and name not in argument_names
            and name != self.loop_iterator
        ]


def _normalize_reductions(reductions):
    if reductions is None:
        return {}
    if hasattr(reductions, "reductions"):
        reductions = reductions.reductions
    if isinstance(reductions, dict):
        reductions = reductions.values()

    normalized = {}
    for reduction in reductions:
        statement = getattr(reduction, "statement", reduction)
        statement_id = getattr(
            reduction,
            "statement_id",
            getattr(statement, "id", None),
        )
        operator = getattr(
            reduction,
            "operator",
            getattr(statement, "update_op", None),
        )
        source_statement = getattr(statement, "source_statement", None)
        if (
                statement_id is None
                or operator is None
                or source_statement is None
        ):
            raise CUDALoweringError(
                "Reduction metadata must come from dependence analysis and "
                "contain a source statement, statement ID, and operator."
            )
        location = _source_location(source_statement)
        normalized[location] = _ReductionSite(
            statement_id=statement_id,
            operator=operator,
            location=location,
        )
    return normalized


def _normalize_execution_mode(execution_mode):
    execution_mode = execution_mode.replace("-", "_")
    if execution_mode not in CUDA_EXECUTION_MODES:
        raise CUDALoweringError(
            f"Unknown CUDA execution mode {execution_mode!r}; expected one "
            f"of {sorted(CUDA_EXECUTION_MODES)}."
        )
    return execution_mode


def _cuda_atomic_add(array_name, index, value):
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="cuda", ctx=ast.Load()),
                attr="atomic",
                ctx=ast.Load(),
            ),
            attr="add",
            ctx=ast.Load(),
        ),
        args=[
            ast.Name(id=array_name, ctx=ast.Load()),
            index,
            value,
        ],
        keywords=[],
    )


def _function_arguments(argument_names):
    return ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name) for name in argument_names],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )


def _kernel_launch(
        kernel_name,
        blocks_per_grid,
        threads_per_block,
        argument_names,
):
    return ast.Expr(
        value=ast.Call(
            func=ast.Subscript(
                value=ast.Name(id=kernel_name, ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[blocks_per_grid, threads_per_block],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            ),
            args=[
                ast.Name(id=name, ctx=ast.Load())
                for name in argument_names
            ],
            keywords=[],
        )
    )


def _source_location(node):
    if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
        raise CUDALoweringError(
            "A reduction source statement has no source location."
        )
    return (
        node.lineno,
        node.col_offset,
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
    )


def _to_module(code):
    if isinstance(code, ast.Module):
        return deepcopy(code)
    if isinstance(code, ast.FunctionDef):
        return ast.Module(body=[deepcopy(code)], type_ignores=[])
    if isinstance(code, str):
        return ast.parse(textwrap.dedent(code))
    if inspect.isfunction(code):
        return ast.parse(textwrap.dedent(inspect.getsource(code)))
    raise CUDALoweringError(
        f"Unsupported lowering input {type(code).__name__}."
    )


def lower_to_numba_cuda(
        code,
        loop_iterator,
        reductions=None,
        kernel_name=None,
        execution_mode="plain",
        sync_free_ready_count=0,
        wavefront_threads_per_block=128,
):
    return CUDALowerer(
        loop_iterator=loop_iterator,
        reductions=reductions,
        kernel_name=kernel_name,
        execution_mode=execution_mode,
        sync_free_ready_count=sync_free_ready_count,
        wavefront_threads_per_block=wavefront_threads_per_block,
    ).lower(code)
