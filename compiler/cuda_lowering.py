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
from compiler.dag_codegen import (
    COUNTER_DEPENDENCE,
    DEPENDENCE_PROTOCOLS,
    PULL_DEPENDENCE,
    PUSH_DEPENDENCE,
)


class CUDALoweringError(Exception):
    """Base error for unsupported CUDA lowering."""


class UnsupportedAtomicOperation(CUDALoweringError):
    """Raised when a reduction has no direct Numba-CUDA atomic lowering."""


KNOWN_LOWERING_CHALLENGES = (
    "DAG schedule-to-block mapping",
    "Host/device allocation and transfers",
    "Grid bounds checks and synchronization",
    "Atomic operation and dtype support",
    "Stable statement IDs across transformations",
    "Busy-wait forward progress",
)

CUDA_EXECUTION_MODES = frozenset({"plain", "sync_free", "wavefront"})
SYNC_FREE_ARGUMENTS = ("dep_count", "dag_ptr", "dag_index")
SYNC_FREE_RESET_ARGUMENT = "initial_dep_count"
WAVEFRONT_ARGUMENTS = ("wavefronts", "wave_start", "wave_end")
WARP_SYNC_FREE_ARGUMENTS = (
    "dep_count",
    "dag_ptr",
    "dag_index",
    "task_nodes",
    "task_ptr",
    "task_counter",
)
WARP_PUSH_ARGUMENTS = ("dep_count", "task_nodes", "task_ptr", "task_counter")
WARP_PULL_ARGUMENTS = ("done", "task_nodes", "task_ptr", "task_counter")
WARP_WAVEFRONT_ARGUMENTS = ("task_nodes", "task_ptr", "band_start", "band_end")
WARP_SIZE = 32
FULL_WARP_MASK = 0xFFFFFFFF
RESIDENT_WARPS_PER_SM = 32
_PURE_CALLS = frozenset({"abs", "float", "int", "len", "max", "min", "range"})
NODE_LOAD_PREFIX = "_node_load_"


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
    n_warp: int | None = None
    warp_mapped: bool = False
    dependence_protocol: str = COUNTER_DEPENDENCE
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


THREAD_REDUCTION_DTYPE = "_thread_reduction_float64"


@dataclass(frozen=True)
class _ThreadReduction:
    target: ast.expr
    operator: ast.operator
    partial_name: str
    buffer_name: str


class _ThreadReductionPrivatizer(ast.NodeTransformer):
    REDUCTION_OPERATORS = (ast.Add, ast.Sub)

    def __init__(self, loop_local_names, live_out_names, first_index):
        self.loop_local_names = loop_local_names
        self.live_out_names = live_out_names
        self.next_index = first_index
        self.reductions = []
        self.parallel_safe = True
        self.loop_depth = 0

    def visit_AugAssign(self, node):
        if not self._is_thread_invariant(node.target):
            return self.generic_visit(node)
        if not isinstance(node.op, self.REDUCTION_OPERATORS):
            self.parallel_safe = False
            return node
        reduction = _ThreadReduction(
            target=deepcopy(node.target),
            operator=node.op,
            partial_name=f"_thread_partial_{self.next_index}",
            buffer_name=f"_thread_reduction_buffer_{self.next_index}",
        )
        self.next_index += 1
        self.reductions.append(reduction)
        return ast.copy_location(
            ast.AugAssign(
                target=ast.Name(id=reduction.partial_name, ctx=ast.Store()),
                op=ast.Add(),
                value=self.visit(node.value),
            ),
            node,
        )

    def visit_Assign(self, node):
        if any(self._is_thread_invariant(target) for target in node.targets):
            self.parallel_safe = False
            return node
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if self._is_thread_invariant(node.target):
            self.parallel_safe = False
            return node
        return self.generic_visit(node)

    def visit_For(self, node):
        if self._is_thread_invariant(node.target):
            self.parallel_safe = False
            return node
        return self._visit_nested_loop(node)

    def visit_While(self, node):
        return self._visit_nested_loop(node)

    def visit_Break(self, node):
        if self.loop_depth == 0:
            self.parallel_safe = False
        return node

    def visit_Return(self, node):
        self.parallel_safe = False
        return node

    def _visit_nested_loop(self, node):
        self.loop_depth += 1
        node = self.generic_visit(node)
        self.loop_depth -= 1
        return node

    def _is_thread_invariant(self, target):
        if isinstance(target, ast.Name):
            return target.id in self.live_out_names
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(
                self._is_thread_invariant(element)
                for element in target.elts
            )
        if isinstance(target, ast.Subscript):
            names = {
                node.id
                for node in ast.walk(target)
                if isinstance(node, ast.Name)
            }
            return not names & self.loop_local_names
        return False


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
            n_warp=None,
            dependence_protocol=COUNTER_DEPENDENCE,
            direct_dependence=None,
    ):
        self.loop_iterator = loop_iterator
        self.thread_loop_iterator = thread_loop_iterator
        self.n_warp = n_warp
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
        if n_warp is not None and (not isinstance(n_warp, int) or n_warp <= 0):
            raise CUDALoweringError("n_warp must be a positive integer.")
        self.warp_mapped = (
            n_warp == 1
            and thread_loop_iterator is not None
            and self.execution_mode in {"sync_free", "wavefront"}
            and wavefront_threads_per_block % WARP_SIZE == 0
        )
        if dependence_protocol not in DEPENDENCE_PROTOCOLS:
            raise CUDALoweringError(
                f"Unknown dependence protocol {dependence_protocol!r}."
            )
        self.direct_dependence = direct_dependence
        self.dependence_protocol = (
            dependence_protocol
            if (
                direct_dependence is not None
                and direct_dependence.kind == dependence_protocol
                and self.warp_mapped
                and self.execution_mode == "sync_free"
                and sync_free_ready_count == 0
            )
            else COUNTER_DEPENDENCE
        )

    def lower(self, code):
        self.thread_reduction_count = 0
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
        atomic_lowerer = _AtomicReductionLowerer(self.reductions)
        lowered_loop_body = [
            atomic_lowerer.visit(deepcopy(statement))
            for statement in target_loop.body
        ]
        if self.thread_loop_iterator is not None:
            lowered_loop_body = self._thread_mapped_body(
                lowered_loop_body,
            )

        auxiliary_arguments = []
        if self.execution_mode == "sync_free":
            if not self.warp_mapped:
                auxiliary_arguments = list(SYNC_FREE_ARGUMENTS)
            elif self.dependence_protocol == PUSH_DEPENDENCE:
                auxiliary_arguments = list(WARP_PUSH_ARGUMENTS)
            elif self.dependence_protocol == PULL_DEPENDENCE:
                auxiliary_arguments = list(WARP_PULL_ARGUMENTS)
            else:
                auxiliary_arguments = list(WARP_SYNC_FREE_ARGUMENTS)
            if (
                    self._needs_sync_free_dep_count_reset()
                    and self.dependence_protocol != PULL_DEPENDENCE
            ):
                auxiliary_arguments.insert(1, SYNC_FREE_RESET_ARGUMENT)
        elif self.execution_mode == "wavefront":
            auxiliary_arguments = list(
                WARP_WAVEFRONT_ARGUMENTS
                if self.warp_mapped
                else WAVEFRONT_ARGUMENTS
            )
        if auxiliary_arguments:
            kernel_arguments.extend(
                name
                for name in auxiliary_arguments
                if name not in kernel_arguments
            )

        if self.warp_mapped:
            kernel_body = self._warp_kernel_body(
                target_loop,
                lowered_loop_body,
            )
        elif self.execution_mode == "wavefront":
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
                    kernel_body.extend(self._sync_free_wait())
                else:
                    kernel_body.append(
                        self._thread_zero_statement(
                            *self._sync_free_wait(),
                        )
                    )
                    kernel_body.append(_cuda_syncthreads())
            kernel_body.extend(lowered_loop_body)
        if self.execution_mode == "sync_free" and not self.warp_mapped:
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
        numba_names = [ast.alias(name="cuda")]
        if self.thread_reduction_count:
            numba_names.append(
                ast.alias(name="float64", asname=THREAD_REDUCTION_DTYPE)
            )
        cuda_import = ast.ImportFrom(
            module="numba",
            names=numba_names,
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
            n_warp=self.n_warp,
            warp_mapped=self.warp_mapped,
            dependence_protocol=self.dependence_protocol,
        )

    def _launcher(
            self,
            kernel_name,
            kernel_arguments,
            target_loop,
            launcher_name,
            reset_kernel_name=None,
    ):
        if self.warp_mapped and self.execution_mode == "wavefront":
            return self._warp_wavefront_launcher(
                kernel_name,
                kernel_arguments,
                launcher_name,
            )
        if self.warp_mapped:
            return self._warp_sync_free_launcher(
                kernel_name,
                kernel_arguments,
                launcher_name,
                reset_kernel_name,
            )
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

    def _sync_free_reset_kernel(self, reset_kernel_name):
        if self.dependence_protocol == PULL_DEPENDENCE:
            return ast.parse(
                f"""
@cuda.jit
def {reset_kernel_name}(done, task_counter):
    _dep_count_index = cuda.grid(1)
    if _dep_count_index < len(done):
        done[_dep_count_index] = 0
    if _dep_count_index == 0:
        task_counter[0] = 0
"""
            ).body[0]
        if self.warp_mapped:
            return ast.parse(
                f"""
@cuda.jit
def {reset_kernel_name}(dep_count, initial_dep_count, task_counter):
    _dep_count_index = cuda.grid(1)
    if _dep_count_index < len(dep_count):
        dep_count[_dep_count_index] = initial_dep_count[_dep_count_index]
    if _dep_count_index == 0:
        task_counter[0] = 0
"""
            ).body[0]
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
        if self.dependence_protocol == PULL_DEPENDENCE:
            reset_arguments = ["done", "task_counter"]
        else:
            reset_arguments = ["dep_count", SYNC_FREE_RESET_ARGUMENT] + (
                ["task_counter"] if self.warp_mapped else []
            )
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
                                    id=reset_arguments[0],
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
                reset_arguments,
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

    @property
    def _warps_per_block(self):
        return self.wavefront_threads_per_block // WARP_SIZE

    def _warp_wavefront_launcher(
            self,
            kernel_name,
            kernel_arguments,
            launcher_name,
    ):
        band_arguments = {"band_start", "band_end"}
        base_arguments = [
            argument
            for argument in kernel_arguments
            if argument not in band_arguments
        ]
        launcher_arguments = (
            self._external_launcher_arguments(base_arguments)
            + ["band_ptr"]
        )
        body = ast.parse(
            f"""
_threads_per_block = {self.wavefront_threads_per_block}
_warps_per_block = {self._warps_per_block}
for _band in range(len(band_ptr) - 1):
    _band_start = band_ptr[_band]
    _band_end = band_ptr[_band + 1]
    _band_size = _band_end - _band_start
    if _band_size > 0:
        _blocks_per_grid = (
            (_band_size + _warps_per_block - 1) // _warps_per_block
        )
"""
        ).body
        body[-1].body[-1].body.append(
            _kernel_launch(
                kernel_name,
                ast.Name(id="_blocks_per_grid", ctx=ast.Load()),
                ast.Name(id="_threads_per_block", ctx=ast.Load()),
                base_arguments + ["_band_start", "_band_end"],
            )
        )
        launcher = ast.FunctionDef(
            name=launcher_name,
            args=_function_arguments(launcher_arguments),
            body=self._wrap_in_host_loops(body),
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return launcher, launcher_arguments

    def _warp_sync_free_launcher(
            self,
            kernel_name,
            kernel_arguments,
            launcher_name,
            reset_kernel_name,
    ):
        launcher_arguments = self._external_launcher_arguments(
            kernel_arguments,
        )
        resident_blocks_per_sm = max(
            1,
            RESIDENT_WARPS_PER_SM // self._warps_per_block,
        )
        body = []
        if reset_kernel_name is not None:
            body.extend(self._sync_free_reset_launch(reset_kernel_name))
        body.extend(ast.parse(
            f"""
_threads_per_block = {self.wavefront_threads_per_block}
_warps_per_block = {self._warps_per_block}
_task_count = len(task_ptr) - 1
_blocks_per_grid = min(
    (_task_count + _warps_per_block - 1) // _warps_per_block,
    cuda.get_current_device().MULTIPROCESSOR_COUNT
    * {resident_blocks_per_sm},
)
if _blocks_per_grid > 0:
    pass
"""
        ).body)
        body[-1].body = [
            _kernel_launch(
                kernel_name,
                ast.Name(id="_blocks_per_grid", ctx=ast.Load()),
                ast.Name(id="_threads_per_block", ctx=ast.Load()),
                kernel_arguments,
            )
        ]
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

    def _warp_kernel_body(self, target_loop, lowered_loop_body):
        node_body = ast.If(
            test=self._loop_bounds_guard(target_loop),
            body=lowered_loop_body,
            orelse=[],
        )
        task_loop = ast.parse(
            f"""
for _task_position in range(task_ptr[_task], task_ptr[_task + 1]):
    {self.loop_iterator} = task_nodes[_task_position]
"""
        ).body[0]
        if self.execution_mode == "wavefront":
            task_loop.body.extend([node_body, self._thread_sync()])
            body = ast.parse(
                f"""
_task = (
    band_start
    + cuda.blockIdx.x * {self._warps_per_block}
    + cuda.threadIdx.x // {WARP_SIZE}
)
_thread_id = cuda.threadIdx.x % {WARP_SIZE}
if _task < band_end:
    pass
"""
            ).body
            body[-1].body = [task_loop]
            return body

        task_loop.body.extend(self._warp_node_statements(node_body))
        body = ast.parse(
            f"""
_thread_id = cuda.threadIdx.x % {WARP_SIZE}
_task_count = len(task_ptr) - 1
while True:
    _task = 0
    if _thread_id == 0:
        _task = cuda.atomic.add(task_counter, 0, 1)
    _task = cuda.shfl_sync({FULL_WARP_MASK:#x}, _task, 0)
    if _task >= _task_count:
        return
"""
        ).body
        body[-1].body.append(task_loop)
        return body

    def _warp_node_statements(self, node_body):
        task = self.loop_iterator
        if self.dependence_protocol == PULL_DEPENDENCE:
            return [node_body] + ast.parse(
                f"""
cuda.threadfence()
cuda.syncwarp({FULL_WARP_MASK:#x})
if _thread_id == 0:
    cuda.atomic.exch(done, {task}, 1)
"""
            ).body
        wait = ast.parse(
            f"""
while cuda.atomic.add(dep_count, {task}, 0) != {self.sync_free_ready_count}:
    pass
cuda.threadfence()
"""
        ).body
        node_body = deepcopy(node_body)
        statements = node_body.body
        position = 0
        while position < len(statements) and _is_node_load(
                statements[position]
        ):
            position += 1
        if position < len(statements) and self._is_thread_zero(
                statements[position]
        ):
            statements[position].body = wait + statements[position].body
        else:
            statements.insert(position, self._thread_zero_statement(*wait))
        if position + 1 < len(statements) and not self._is_thread_sync(
                statements[position + 1]
        ):
            statements.insert(position + 1, self._thread_sync())
        if self.dependence_protocol == PUSH_DEPENDENCE:
            if self._is_thread_sync(statements[-1]):
                return [node_body]
            return [node_body, self._thread_sync()]
        return [node_body] + ast.parse(
            f"""
cuda.threadfence()
cuda.syncwarp({FULL_WARP_MASK:#x})
for _dag_position in range(
    dag_ptr[{task}] + _thread_id,
    dag_ptr[{task} + 1],
    {WARP_SIZE},
):
    cuda.atomic.add(dep_count, dag_index[_dag_position], -1)
"""
        ).body

    def _thread_sync(self):
        if self.warp_mapped:
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="cuda", ctx=ast.Load()),
                        attr="syncwarp",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=FULL_WARP_MASK)],
                    keywords=[],
                )
            )
        return _cuda_syncthreads()

    def _is_thread_sync(self, statement):
        return ast.dump(statement) == ast.dump(self._thread_sync())

    def _is_thread_zero(self, statement):
        return (
            isinstance(statement, ast.If)
            and not statement.orelse
            and ast.dump(statement.test)
            == ast.dump(self._thread_zero_statement().test)
        )

    def _thread_stride(self):
        if self.warp_mapped:
            return ast.Constant(value=WARP_SIZE)
        return _cuda_block_dim()

    def _warp_reduction(self, reduction):
        partial = reduction.partial_name
        mask = f"{FULL_WARP_MASK:#x}"
        offsets = []
        offset = WARP_SIZE // 2
        while offset:
            offsets.append(offset)
            offset //= 2
        return ast.parse(
            "\n".join(
                f"{partial} += cuda.shfl_down_sync({mask}, {partial}, {offset})"
                for offset in offsets
            )
        ).body

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
        if sum(
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == self.thread_loop_iterator
                for statement in lowered_loop_body
                for node in ast.walk(statement)
        ) != 1:
            self.dependence_protocol = COUNTER_DEPENDENCE
        if self.dependence_protocol == PUSH_DEPENDENCE:
            reduction_count = self.thread_reduction_count
            privatized = self._privatize_thread_reductions(
                thread_loop,
                prelude,
                epilogue,
            )
            self.thread_reduction_count = reduction_count
            if (privatized is not None and privatized[1]) or any(
                    isinstance(node, (ast.Break, ast.Continue, ast.Return))
                    for node in ast.walk(thread_loop)
            ):
                self.dependence_protocol = COUNTER_DEPENDENCE
        if self.dependence_protocol != COUNTER_DEPENDENCE:
            thread_loop = self._with_dependence_hooks(thread_loop)
        privatized = self._privatize_thread_reductions(
            thread_loop,
            prelude,
            epilogue,
        )
        if privatized is None:
            return [
                self._thread_zero_statement(
                    *prelude,
                    thread_loop,
                    *epilogue,
                )
            ]
        thread_loop, reductions = privatized
        node_loads = []
        if self.warp_mapped:
            thread_loop = _cache_read_only_loads(
                thread_loop,
                _written_arrays([*prelude, thread_loop, *epilogue]),
            )
            prelude, thread_loop, epilogue, node_loads = (
                self._prefetch_node_loads(prelude, thread_loop, epilogue)
            )

        body = list(node_loads)
        if prelude:
            body.append(self._thread_zero_statement(*prelude))
        body.append(self._thread_sync())
        body.extend(
            ast.Assign(
                targets=[
                    ast.Name(id=reduction.partial_name, ctx=ast.Store())
                ],
                value=ast.Constant(value=0.0),
            )
            for reduction in reductions
        )
        body.append(self._thread_stride_loop(thread_loop))
        body.append(self._thread_sync())
        for reduction in reductions:
            if self.warp_mapped:
                body.extend(self._warp_reduction(reduction))
            else:
                body.extend(self._block_reduction(reduction))
        epilogue = [
            ast.AugAssign(
                target=deepcopy(reduction.target),
                op=type(reduction.operator)(),
                value=self._reduction_result(reduction),
            )
            for reduction in reductions
        ] + epilogue
        if epilogue:
            body.append(self._thread_zero_statement(*epilogue))
        return body

    def _prefetch_node_loads(self, prelude, thread_loop, epilogue):
        node_statements = [*prelude, thread_loop, *epilogue]
        written = _written_arrays(node_statements)
        mutable = _stored_names(node_statements)
        sources = []
        for statement in prelude:
            if _exits_early(statement):
                break
            sources.extend(_statement_evaluated_subscripts(statement))
        else:
            sources.extend(_evaluated_subscripts(thread_loop.iter))
        names = {}
        loads = []
        for subscript in sources:
            key = ast.dump(subscript)
            if (
                    key in names
                    or not isinstance(subscript.value, ast.Name)
                    or subscript.value.id in written
                    or subscript.value.id in mutable
                    or not _is_simple_index(subscript.slice, mutable)
            ):
                continue
            names[key] = f"{NODE_LOAD_PREFIX}{len(names)}"
            loads.append(
                ast.Assign(
                    targets=[ast.Name(id=names[key], ctx=ast.Store())],
                    value=deepcopy(subscript),
                )
            )
        if not loads:
            return prelude, thread_loop, epilogue, []
        replacer = _LoadReplacer(names)
        return (
            [replacer.visit(deepcopy(statement)) for statement in prelude],
            replacer.visit(deepcopy(thread_loop)),
            [replacer.visit(deepcopy(statement)) for statement in epilogue],
            loads,
        )

    def _with_dependence_hooks(self, thread_loop):
        endpoint = ast.unparse(self.direct_dependence.endpoint)
        if self.dependence_protocol == PULL_DEPENDENCE:
            hook = ast.parse(
                f"""
while cuda.atomic.add(done, {endpoint}, 0) == 0:
    pass
cuda.threadfence()
"""
            ).body
        else:
            hook = ast.parse(
                f"""
cuda.threadfence()
cuda.atomic.add(dep_count, {endpoint}, -1)
"""
            ).body
        if self.direct_dependence.guard is not None:
            hook = [ast.If(
                test=deepcopy(self.direct_dependence.guard),
                body=hook,
                orelse=[],
            )]
        hooked_loop = deepcopy(thread_loop)
        if self.dependence_protocol == PULL_DEPENDENCE:
            hooked_loop.body = hook + hooked_loop.body
        else:
            hooked_loop.body = hooked_loop.body + hook
        return hooked_loop

    def _privatize_thread_reductions(self, thread_loop, prelude, epilogue):
        loop_local_names = {self.thread_loop_iterator}
        loop_local_names.update(
            node.id
            for node in ast.walk(thread_loop)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        live_out_names = {
            node.id
            for statement in prelude + epilogue
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        prelude_locals = {
            node.id
            for statement in prelude
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        loop_loads = {
            node.id
            for node in ast.walk(thread_loop)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if prelude_locals & loop_loads:
            return None

        privatizer = _ThreadReductionPrivatizer(
            loop_local_names,
            live_out_names,
            self.thread_reduction_count,
        )
        privatized_loop = deepcopy(thread_loop)
        privatized_loop.body = [
            privatizer.visit(statement)
            for statement in privatized_loop.body
        ]
        if not privatizer.parallel_safe:
            return None
        loaded_expressions = {
            ast.unparse(node)
            for node in ast.walk(privatized_loop)
            if isinstance(node, (ast.Name, ast.Subscript))
            and isinstance(node.ctx, ast.Load)
        }
        if any(
                ast.unparse(reduction.target) in loaded_expressions
                for reduction in privatizer.reductions
        ):
            return None
        self.thread_reduction_count = privatizer.next_index
        return privatized_loop, privatizer.reductions

    def _reduction_result(self, reduction):
        if self.warp_mapped:
            return ast.Name(id=reduction.partial_name, ctx=ast.Load())
        return ast.Subscript(
            value=ast.Name(id=reduction.buffer_name, ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )

    def _block_reduction(self, reduction):
        block_size = self.wavefront_threads_per_block
        buffer = reduction.buffer_name
        return ast.parse(
            f"""
{buffer} = cuda.shared.array({block_size}, {THREAD_REDUCTION_DTYPE})
{buffer}[_thread_id] = {reduction.partial_name}
cuda.syncthreads()
_thread_reduction_stride = 1
while _thread_reduction_stride < {block_size}:
    if (
        _thread_id % (2 * _thread_reduction_stride) == 0
        and _thread_id + _thread_reduction_stride < {block_size}
    ):
        {buffer}[_thread_id] += {buffer}[_thread_id + _thread_reduction_stride]
    cuda.syncthreads()
    _thread_reduction_stride *= 2
"""
        ).body

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
                        self._thread_stride(),
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
                _multiply_expr(self._thread_stride(), ast.Constant(value=offset)),
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
                        self._thread_stride(),
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
        return [
            ast.While(
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
            ),
            _cuda_threadfence(),
        ]

    def _sync_free_release(self):
        # Do not reuse the task iterator as the adjacency-list iterator: the
        # task ID is still needed to select its successor-list bounds.
        task = ast.Name(id=self.loop_iterator, ctx=ast.Load())
        return [
            _cuda_threadfence(),
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


def _cuda_threadfence():
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


def _stored_names(statements):
    return {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _written_arrays(statements):
    written = set()
    for statement in statements:
        for node in ast.walk(statement):
            if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
            ):
                written.add(node.value.id)
            elif isinstance(node, ast.Call) and not (
                    isinstance(node.func, ast.Name)
                    and node.func.id in _PURE_CALLS
            ):
                written.update(
                    argument.id
                    for argument in node.args
                    if isinstance(argument, ast.Name)
                )
    return written


def _is_simple_index(index, mutable_names):
    for node in ast.walk(index):
        if isinstance(node, ast.Name):
            if node.id in mutable_names:
                return False
        elif not isinstance(
                node,
                (
                    ast.Constant,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.operator,
                    ast.unaryop,
                    ast.expr_context,
                ),
        ):
            return False
    return True


def _evaluated_subscripts(node):
    if isinstance(
            node,
            (
                ast.BoolOp,
                ast.IfExp,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
    ):
        return []
    found = []
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        found.append(node)
    for child in ast.iter_child_nodes(node):
        found.extend(_evaluated_subscripts(child))
    return found


def _statement_evaluated_subscripts(statement):
    if isinstance(statement, (ast.If, ast.While)):
        return _evaluated_subscripts(statement.test)
    if isinstance(statement, (ast.Assign, ast.AugAssign, ast.Expr)):
        return _evaluated_subscripts(statement)
    return []


def _exits_early(statement):
    return any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return))
        for node in ast.walk(statement)
    )


def _is_node_load(statement):
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.startswith(NODE_LOAD_PREFIX)
    )


def _cache_read_only_loads(thread_loop, written_arrays):
    rebound = _stored_names(thread_loop.body)
    names = {}
    body = []
    exited = False
    for position, statement in enumerate(thread_loop.body):
        for node in [] if exited else _statement_evaluated_subscripts(statement):
            key = ast.dump(node)
            if (
                    key in names
                    or not isinstance(node.value, ast.Name)
                    or node.value.id in written_arrays
                    or node.value.id in rebound
                    or not _is_simple_index(node.slice, rebound)
                    or sum(
                        ast.dump(other) == key
                        for later in thread_loop.body[position:]
                        for other in ast.walk(later)
                    ) < 2
            ):
                continue
            names[key] = f"_thread_load_{len(names)}"
            body.append(
                ast.Assign(
                    targets=[ast.Name(id=names[key], ctx=ast.Store())],
                    value=deepcopy(node),
                )
            )
        body.append(_LoadReplacer(names).visit(deepcopy(statement)))
        exited = exited or _exits_early(statement)
    cached_loop = deepcopy(thread_loop)
    cached_loop.body = body
    return cached_loop


class _LoadReplacer(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Subscript(self, node):
        name = self.names.get(ast.dump(node))
        if name is not None:
            return ast.Name(id=name, ctx=ast.Load())
        return self.generic_visit(node)


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
        n_warp=None,
        dependence_protocol=COUNTER_DEPENDENCE,
        direct_dependence=None,
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
        n_warp=n_warp,
        dependence_protocol=dependence_protocol,
        direct_dependence=direct_dependence,
    ).lower(code)
