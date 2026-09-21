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

from compiler.ast_normalization import normalize_aliases


class CUDALoweringError(Exception):
    """Base error for unsupported CUDA lowering."""


class UnsupportedAtomicOperation(CUDALoweringError):
    """Raised when a reduction has no direct Numba-CUDA atomic lowering."""


KNOWN_LOWERING_CHALLENGES = (
    "Block reductions for thread-mapped loops",
    "DAG schedule-to-block mapping",
    "Host/device allocation and transfers",
    "Grid bounds checks and synchronization",
    "Atomic operation and dtype support",
    "Stable statement IDs across transformations",
    "Busy-wait forward progress and global-memory publication ordering",
)

CUDA_EXECUTION_MODES = frozenset({"plain", "sync_free", "wavefront"})
SYNC_FREE_ARGUMENTS = ("dep_count", "dag_ptr", "dag_index")
SYNC_FREE_RESET_ARGUMENT = "initial_dep_count"
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
    lifted_loop_parameters: list[str]
    host_prelude: list[ast.stmt]
    host_epilogue: list[ast.stmt]
    atomic_statement_ids: set[str]
    execution_mode: str
    auxiliary_arguments: list[str]
    thread_loop_iterator: str | None = None
    thread_unroll_factor: int = 1
    host_loop_iterators: tuple[str, ...] = ()
    challenges: tuple[str, ...] = KNOWN_LOWERING_CHALLENGES

    @property
    def source(self):
        return ast.unparse(self.module_ast)


@dataclass(frozen=True)
class _ReductionSite:
    statement_id: str
    operator: str
    location: tuple


@dataclass(frozen=True)
class _TargetLoop:
    top_level_index: int
    loop: ast.For
    enclosing_loops: tuple[ast.For, ...]


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
            thread_loop_iterator=None,
            thread_unroll_factor=1,
    ):
        self.loop_iterator = loop_iterator
        self.thread_loop_iterator = thread_loop_iterator
        self.reductions = _normalize_reductions(reductions)
        self.kernel_name = kernel_name
        self.execution_mode = _normalize_execution_mode(execution_mode)
        self.sync_free_ready_count = sync_free_ready_count
        self.wavefront_threads_per_block = wavefront_threads_per_block
        self.thread_unroll_factor = thread_unroll_factor
        self.enclosing_block_loops = []
        self.host_loop_iterators = []
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
        if (
                not isinstance(thread_unroll_factor, int)
                or thread_unroll_factor <= 0
        ):
            raise CUDALoweringError(
                "The thread unroll factor must be a positive integer."
            )
        if thread_unroll_factor != 1 and thread_loop_iterator is None:
            raise CUDALoweringError(
                "A thread unroll factor requires a thread-mapped loop."
            )

    def lower(self, code):
        module = normalize_aliases(_to_module(code))
        functions = [
            node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
            raise CUDALoweringError(
                "CUDA lowering expects exactly one synchronous function."
            )

        function = deepcopy(functions[0])
        target_info = self._target_loop(function)
        target_loop = target_info.loop
        self.enclosing_block_loops = [
            deepcopy(loop)
            for loop in target_info.enclosing_loops
        ]
        self.host_loop_iterators = [
            loop.target.id
            for loop in self.enclosing_block_loops
            if isinstance(loop.target, ast.Name)
        ]
        host_prelude = deepcopy(function.body[:target_info.top_level_index])
        host_epilogue = deepcopy(
            function.body[target_info.top_level_index + 1:]
        )

        lifted_variables = self._lifted_variables(function, target_loop, host_prelude)
        lifted_loop_parameters = self._lifted_loop_parameters(
            function,
            target_loop,
            host_prelude,
        )
        kernel_arguments = [
            argument.arg
            for argument in function.args.args
        ]
        kernel_arguments.extend(
            variable
            for variable in lifted_variables
            if variable not in kernel_arguments
        )
        kernel_arguments.extend(
            iterator
            for iterator in self.host_loop_iterators
            if iterator not in kernel_arguments
        )
        auxiliary_arguments = []
        if self.execution_mode == "sync_free":
            auxiliary_arguments = list(SYNC_FREE_ARGUMENTS)
            if self._needs_sync_free_dep_count_reset():
                auxiliary_arguments.insert(1, SYNC_FREE_RESET_ARGUMENT)
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
        if self.thread_loop_iterator is not None:
            lowered_loop_body = self._thread_mapped_body(
                lowered_loop_body,
            )

        if self.execution_mode == "wavefront":
            kernel_body = self._wavefront_body(
                target_loop,
                lowered_loop_body,
            )
        else:
            kernel_body = [self._block_iterator_assignment()]
            if self.thread_loop_iterator is not None:
                kernel_body.append(self._thread_index_assignment())
            if self.execution_mode == "sync_free":
                if self.thread_loop_iterator is None:
                    kernel_body.append(self._sync_free_wait())
                else:
                    kernel_body.append(
                        self._thread_zero_statement(
                            self._sync_free_wait(),
                        )
                    )
                    kernel_body.append(_cuda_syncthreads())
            kernel_body.extend(lowered_loop_body)
        if self.execution_mode == "sync_free":
            if self.thread_loop_iterator is None:
                kernel_body.extend(self._sync_free_release())
            else:
                kernel_body.append(
                    self._thread_zero_statement(
                        *self._sync_free_release(),
                    )
                )
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
        reset_kernel_name = None
        reset_kernel = None
        if self._needs_sync_free_dep_count_reset():
            reset_kernel_name = self._sync_free_reset_kernel_name(
                kernel_name,
            )
            reset_kernel = self._sync_free_reset_kernel(reset_kernel_name)
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
            reset_kernel_name,
        )
        cuda_import = ast.ImportFrom(
            module="numba",
            names=[ast.alias(name="cuda")],
            level=0,
        )
        module_body = [cuda_import]
        if reset_kernel is not None:
            module_body.append(reset_kernel)
        module_body.extend([kernel, launcher])
        lowered_module = ast.fix_missing_locations(
            ast.Module(
                body=module_body,
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
            lifted_loop_parameters=lifted_loop_parameters,
            host_prelude=host_prelude,
            host_epilogue=host_epilogue,
            atomic_statement_ids=atomic_lowerer.lowered_statement_ids,
            execution_mode=self.execution_mode,
            auxiliary_arguments=auxiliary_arguments,
            thread_loop_iterator=self.thread_loop_iterator,
            thread_unroll_factor=self.thread_unroll_factor,
            host_loop_iterators=tuple(self.host_loop_iterators),
        )

    def _launcher(
            self,
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
            reset_kernel_name=None,
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
            reset_kernel_name,
        )

    def _single_launch_launcher(
            self,
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
            reset_kernel_name,
    ):
        launcher_arguments = self._external_launcher_arguments(
            kernel_arguments,
        )
        if self.execution_mode == "sync_free":
            block_count = ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id="dep_count", ctx=ast.Load())],
                keywords=[],
            )
        else:
            block_count = self._block_task_count(target_loop)

        body = []
        if reset_kernel_name is not None:
            body.extend(
                self._sync_free_reset_launch(reset_kernel_name)
            )
        body.extend([
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
                value=ast.Constant(
                    value=(
                        self.wavefront_threads_per_block
                        if self.thread_loop_iterator is not None
                        else 1
                    )
                ),
            ),
            _kernel_launch(
                kernel_name,
                ast.Name(id="_blocks_per_grid", ctx=ast.Load()),
                ast.Name(id="_threads_per_block", ctx=ast.Load()),
                kernel_arguments,
            ),
        ])
        if reset_kernel_name is not None:
            body.append(_cuda_synchronize())

        launcher = ast.FunctionDef(
            name=launcher_name,
            args=_function_arguments(launcher_arguments),
            body=self._wrap_in_host_loops(body),
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return launcher, launcher_arguments

    def _needs_sync_free_dep_count_reset(self):
        return bool(
            self.enclosing_block_loops
            and self.execution_mode == "sync_free"
        )

    @staticmethod
    def _sync_free_reset_kernel_name(kernel_name):
        suffix = "_cuda_kernel"
        if kernel_name.endswith(suffix):
            return (
                kernel_name[:-len(suffix)]
                + "_dep_count_reset_cuda_kernel"
            )
        return kernel_name + "_dep_count_reset"

    @staticmethod
    def _sync_free_reset_kernel(reset_kernel_name):
        return ast.parse(
            f"""
@cuda.jit
def {reset_kernel_name}(dep_count, initial_dep_count):
    _dep_count_index = cuda.grid(1)
    if _dep_count_index < len(dep_count):
        dep_count[_dep_count_index] = initial_dep_count[_dep_count_index]
"""
        ).body[0]

    def _sync_free_reset_launch(self, reset_kernel_name):
        return [
            ast.Assign(
                targets=[
                    ast.Name(
                        id="_reset_threads_per_block",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(value=self.wavefront_threads_per_block),
            ),
            ast.Assign(
                targets=[
                    ast.Name(
                        id="_reset_blocks_per_grid",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.BinOp(
                    left=ast.BinOp(
                        left=ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[
                                ast.Name(
                                    id="dep_count",
                                    ctx=ast.Load(),
                                )
                            ],
                            keywords=[],
                        ),
                        op=ast.Add(),
                        right=ast.BinOp(
                            left=ast.Name(
                                id="_reset_threads_per_block",
                                ctx=ast.Load(),
                            ),
                            op=ast.Sub(),
                            right=ast.Constant(value=1),
                        ),
                    ),
                    op=ast.FloorDiv(),
                    right=ast.Name(
                        id="_reset_threads_per_block",
                        ctx=ast.Load(),
                    ),
                ),
            ),
            _kernel_launch(
                reset_kernel_name,
                ast.Name(
                    id="_reset_blocks_per_grid",
                    ctx=ast.Load(),
                ),
                ast.Name(
                    id="_reset_threads_per_block",
                    ctx=ast.Load(),
                ),
                ["dep_count", SYNC_FREE_RESET_ARGUMENT],
            ),
        ]

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
        launcher_arguments = (
            self._external_launcher_arguments(base_arguments)
            + ["wavefront_ptr"]
        )
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
                            value=self._wavefront_block_count(),
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
        body = [
            ast.Assign(
                targets=[
                    ast.Name(id="_threads_per_block", ctx=ast.Store())
                ],
                value=ast.Constant(
                    value=self.wavefront_threads_per_block
                ),
            ),
            wavefront_loop,
        ]
        launcher = ast.FunctionDef(
            name=launcher_name,
            args=_function_arguments(launcher_arguments),
            body=self._wrap_in_host_loops(body),
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return launcher, launcher_arguments

    def _external_launcher_arguments(self, kernel_arguments):
        host_iterators = set(self.host_loop_iterators)
        return [
            argument
            for argument in kernel_arguments
            if argument not in host_iterators
        ]

    def _wrap_in_host_loops(self, body):
        nested_body = body
        for loop in reversed(self.enclosing_block_loops):
            nested_body = [
                ast.For(
                    target=deepcopy(loop.target),
                    iter=deepcopy(loop.iter),
                    body=nested_body,
                    orelse=[],
                )
            ]
        return nested_body

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

    def _thread_index_assignment(self):
        return ast.Assign(
            targets=[ast.Name(id="_thread_id", ctx=ast.Store())],
            value=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="cuda", ctx=ast.Load()),
                    attr="threadIdx",
                    ctx=ast.Load(),
                ),
                attr="x",
                ctx=ast.Load(),
            ),
        )

    def _thread_zero_statement(self, *statements):
        return ast.If(
            test=ast.Compare(
                left=ast.Name(id="_thread_id", ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)],
            ),
            body=list(statements),
            orelse=[],
        )

    def _wavefront_block_count(self):
        if self.thread_loop_iterator is not None:
            return ast.Name(id="_wave_size", ctx=ast.Load())
        return ast.BinOp(
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
                value=self._wavefront_location_value(),
            ),
            *(
                [self._thread_index_assignment()]
                if self.thread_loop_iterator is not None
                else []
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

    def _wavefront_location_value(self):
        if self.thread_loop_iterator is None:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="cuda", ctx=ast.Load()),
                    attr="grid",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=1)],
                keywords=[],
            )
        return ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="cuda", ctx=ast.Load()),
                attr="blockIdx",
                ctx=ast.Load(),
            ),
            attr="x",
            ctx=ast.Load(),
        )

    def _thread_mapped_body(self, lowered_loop_body):
        matches = [
            (index, statement)
            for index, statement in enumerate(lowered_loop_body)
            if (
                isinstance(statement, ast.For)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == self.thread_loop_iterator
            )
        ]
        if len(matches) != 1:
            raise CUDALoweringError(
                "The selected thread iterator must identify exactly one "
                f"direct inner for loop; found {len(matches)} for "
                f"{self.thread_loop_iterator!r}."
            )

        loop_index, thread_loop = matches[0]
        prelude = lowered_loop_body[:loop_index]
        epilogue = lowered_loop_body[loop_index + 1:]

        body = []
        if prelude:
            body.append(self._thread_zero_statement(*prelude))
        body.append(_cuda_syncthreads())
        body.append(self._thread_stride_loop(thread_loop))
        body.append(_cuda_syncthreads())
        if epilogue:
            body.append(self._thread_zero_statement(*epilogue))
        return body

    def _thread_stride_loop(self, thread_loop):
        if thread_loop.orelse:
            raise CUDALoweringError(
                "Thread-mapped loops with for/else are not supported."
            )
        start, end = self._thread_loop_range(thread_loop)
        if self.thread_unroll_factor == 1:
            return ast.For(
                target=deepcopy(thread_loop.target),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[
                        _add_expr(
                            deepcopy(start),
                            ast.Name(id="_thread_id", ctx=ast.Load()),
                        ),
                        deepcopy(end),
                        _cuda_block_dim(),
                    ],
                    keywords=[],
                ),
                body=deepcopy(thread_loop.body),
                orelse=[],
            )

        iterator_name = thread_loop.target.id
        base_name = "_" + iterator_name + "_base"
        unrolled_body = [
            ast.Assign(
                targets=[ast.Name(id=iterator_name, ctx=ast.Store())],
                value=ast.Name(id=base_name, ctx=ast.Load()),
            ),
            *deepcopy(thread_loop.body),
        ]
        for offset in range(1, self.thread_unroll_factor):
            candidate = _add_expr(
                ast.Name(id=base_name, ctx=ast.Load()),
                _multiply_expr(_cuda_block_dim(), ast.Constant(value=offset)),
            )
            unrolled_body.extend([
                ast.Assign(
                    targets=[
                        ast.Name(id=iterator_name, ctx=ast.Store())
                    ],
                    value=candidate,
                ),
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=iterator_name, ctx=ast.Load()),
                        ops=[ast.Lt()],
                        comparators=[deepcopy(end)],
                    ),
                    body=deepcopy(thread_loop.body),
                    orelse=[],
                ),
            ])

        return ast.For(
            target=ast.Name(id=base_name, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    _add_expr(
                        deepcopy(start),
                        ast.Name(id="_thread_id", ctx=ast.Load()),
                    ),
                    deepcopy(end),
                    _multiply_expr(
                        _cuda_block_dim(),
                        ast.Constant(value=self.thread_unroll_factor),
                    ),
                ],
                keywords=[],
            ),
            body=unrolled_body,
            orelse=[],
        )

    def _thread_loop_range(self, thread_loop):
        loop_range = thread_loop.iter
        if not (
                isinstance(loop_range, ast.Call)
                and isinstance(loop_range.func, ast.Name)
                and loop_range.func.id == "range"
                and not loop_range.keywords
                and len(loop_range.args) in {1, 2}
        ):
            raise CUDALoweringError(
                "Thread-mapped lowering currently requires range(stop) or "
                "range(start, stop)."
            )
        if len(loop_range.args) == 1:
            return ast.Constant(value=0), deepcopy(loop_range.args[0])
        return deepcopy(loop_range.args[0]), deepcopy(loop_range.args[1])

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

    def _sync_free_release(self):
        # Do not reuse the task iterator as the adjacency-list iterator: the
        # task ID is still needed to select its successor-list bounds.
        task = ast.Name(id=self.loop_iterator, ctx=ast.Load())
        return [
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
        matches = []

        def visit_statements(statements, top_level_index, enclosing_loops):
            for index, statement in enumerate(statements):
                if not isinstance(statement, ast.For):
                    continue
                current_top_level_index = (
                    index
                    if top_level_index is None
                    else top_level_index
                )
                if (
                        isinstance(statement.target, ast.Name)
                        and statement.target.id == self.loop_iterator
                ):
                    matches.append(
                        _TargetLoop(
                            current_top_level_index,
                            statement,
                            tuple(enclosing_loops),
                        )
                    )
                visit_statements(
                    statement.body,
                    current_top_level_index,
                    enclosing_loops + (statement,),
                )

        visit_statements(function.body, None, ())
        if len(matches) != 1:
            raise CUDALoweringError(
                "The selected iterator must identify exactly one "
                f"for loop; found {len(matches)} for {self.loop_iterator!r}."
            )
        target = matches[0]
        self._validate_enclosing_block_loops(target)
        return target

    @staticmethod
    def _validate_enclosing_block_loops(target):
        expected_child = target.loop
        for loop in reversed(target.enclosing_loops):
            if not isinstance(loop.target, ast.Name):
                raise CUDALoweringError(
                    "Enclosing block-mapped loops must use simple iterator "
                    "variables."
                )
            body = [
                statement
                for statement in loop.body
                if not isinstance(statement, ast.Pass)
            ]
            if len(body) != 1 or body[0] is not expected_child:
                raise CUDALoweringError(
                    "Nested block mapping currently requires every "
                    "enclosing loop body to contain only the next nested "
                    "loop."
                )
            expected_child = loop

    def _lifted_variables(self, function, target_loop, host_prelude):
        loaded = _LoadedNameCollector()
        for enclosing_loop in self.enclosing_block_loops:
            loaded.visit(enclosing_loop.iter)
        loaded.visit(target_loop.iter)
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

    def _lifted_loop_parameters(
            self,
            function,
            target_loop,
            host_prelude,
    ):
        """Return host-defined values used to parameterize mapped loops."""
        loaded = _LoadedNameCollector()
        for enclosing_loop in self.enclosing_block_loops:
            loaded.visit(enclosing_loop.iter)
        loaded.visit(target_loop.iter)

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


def _cuda_syncthreads():
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="cuda", ctx=ast.Load()),
                attr="syncthreads",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    )


def _cuda_synchronize():
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="cuda", ctx=ast.Load()),
                attr="synchronize",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    )


def _cuda_block_dim():
    return ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="cuda", ctx=ast.Load()),
            attr="blockDim",
            ctx=ast.Load(),
        ),
        attr="x",
        ctx=ast.Load(),
    )


def _add_expr(left, right):
    if isinstance(left, ast.Constant) and left.value == 0:
        return right
    if isinstance(right, ast.Constant) and right.value == 0:
        return left
    return ast.BinOp(left=left, op=ast.Add(), right=right)


def _multiply_expr(left, right):
    if isinstance(right, ast.Constant) and right.value == 1:
        return left
    if isinstance(left, ast.Constant) and left.value == 1:
        return right
    return ast.BinOp(left=left, op=ast.Mult(), right=right)


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
        thread_loop_iterator=None,
        thread_unroll_factor=1,
):
    return CUDALowerer(
        loop_iterator=loop_iterator,
        reductions=reductions,
        kernel_name=kernel_name,
        execution_mode=execution_mode,
        sync_free_ready_count=sync_free_ready_count,
        wavefront_threads_per_block=wavefront_threads_per_block,
        thread_loop_iterator=thread_loop_iterator,
        thread_unroll_factor=thread_unroll_factor,
    ).lower(code)
