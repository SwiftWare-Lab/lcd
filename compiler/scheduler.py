import ast
import hashlib
import itertools
import statistics
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from compiler.dag_codegen import (
    COUNTER_DEPENDENCE,
    PULL_DEPENDENCE,
    PUSH_DEPENDENCE,
)

def executer(Ap, Ai, Ax, **kwargs):
    assert 'schedule' in kwargs
    schedule = kwargs['schedule']
    for s in schedule:
        for first, second in s:
            for i1 in first:
                pass
            for j1 in second:
                pass


def chunks_part(lst, n):
    for i in range(0, len(lst), n):
        yield [], lst[i:i + n]


def schedule_bipartite(DAG: list[list], num_threads):
    init_block_size = 32
    l1_nodes_num = len(DAG) #TODO: I considered same number of computations for first level and second level for now. Need a fix for future.
    num_partitions = max(num_threads * 2, l1_nodes_num // init_block_size)
    new_block_size = l1_nodes_num // num_partitions
    residue = l1_nodes_num % num_partitions
    schedule = [([],[]) for _ in range(num_partitions)]
    unfused_elements = []
    for p in range(num_partitions):
        if p >= residue:
            tile_range = range(p*new_block_size + residue, (p+1)*new_block_size + residue)
        else:
            tile_range = range(p*new_block_size + p, (p+1)*new_block_size + p + 1)
        schedule[p][0].extend(tile_range)
        for l2n in tile_range:
            if len(DAG[l2n]) > 0 and DAG[l2n][0] >= tile_range[0] and DAG[l2n][-1] <= tile_range[-1]:
                schedule[p][1].append(l2n)
            else:
                unfused_elements.append(l2n)
    num_new_partitions = num_threads * 2
    new_partitions = list(chunks_part(unfused_elements, num_new_partitions))
    schedule = [schedule, new_partitions]
    return schedule


WAVEFRONT = "wavefront"
SYNC_FREE = "sync_free"
AUTO = "auto"
EXECUTION_MODES = (WAVEFRONT, SYNC_FREE)
BLOCK_SIZES = (32, 64, 128, 256)
UNROLL_FACTORS = (1, 2, 4, 8)
DEFAULT_THREADS_PER_BLOCK = 128
N_WARP_CANDIDATES = (1,)
WARP_SIZE = 32
PULL_POLLING_FACTOR = 2.0
ARITHMETIC_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)


class SchedulerError(Exception):
    pass


@dataclass(frozen=True)
class ArchConstants:
    launch_latency_s: float
    atomic_cost_s: float
    memory_bandwidth_bytes_per_s: float
    lane_flops_per_s: float


FALLBACK_ARCH_CONSTANTS = ArchConstants(
    launch_latency_s=1.0e-5,
    atomic_cost_s=2.0e-9,
    memory_bandwidth_bytes_per_s=1.0e12,
    lane_flops_per_s=1.5e9,
)


@dataclass(frozen=True)
class WorkloadFeatures:
    rows: int
    edges: int
    levels: int
    nonzeros: int
    flops: float
    memory_bytes: int
    thread_work_items: int = 0
    thread_loop_invocations: int = 0

    @property
    def average_in_degree(self):
        if self.rows == 0:
            return 0.0
        return self.edges / self.rows

    @property
    def average_thread_trip_count(self):
        if self.thread_loop_invocations == 0:
            return 0.0
        return self.thread_work_items / self.thread_loop_invocations


@dataclass(frozen=True)
class CostEstimate:
    launch_s: float
    sync_s: float
    memory_s: float
    compute_s: float

    @property
    def total_s(self):
        return self.launch_s + self.sync_s + self.memory_s + self.compute_s


@dataclass(frozen=True)
class Schedule:
    execution_mode: str
    threads_per_block: int
    thread_unroll: int
    n_warp: int = 1
    dependence_protocol: str = COUNTER_DEPENDENCE
    cost: CostEstimate | None = None

    @property
    def lowering_key(self):
        return (
            self.execution_mode,
            self.threads_per_block,
            self.thread_unroll,
            self.n_warp,
            self.dependence_protocol,
        )


@dataclass(frozen=True)
class SchedulePlan:
    schedule: Schedule
    execution_data: object
    features: WorkloadFeatures | None = None


def normalize_execution_mode(execution_mode):
    if execution_mode is None or execution_mode == AUTO:
        return None
    if execution_mode.replace("-", "_") == SYNC_FREE:
        return SYNC_FREE
    return WAVEFRONT


@dataclass(frozen=True)
class ScheduleSpace:
    execution_modes: tuple
    block_sizes: tuple
    unroll_factors: tuple
    n_warps: tuple = N_WARP_CANDIDATES
    thread_mapped: bool = False
    dependence_protocols: tuple = (COUNTER_DEPENDENCE,)

    @classmethod
    def from_options(
            cls,
            execution_mode=None,
            threads_per_block=None,
            thread_unroll=None,
            thread_mapped=False,
            n_warp=None,
            direct_protocol=None,
            dependence_protocol=None,
    ):
        mode = normalize_execution_mode(execution_mode)
        if n_warp is not None and (not isinstance(n_warp, int) or n_warp <= 0):
            raise SchedulerError("n_warp must be a positive integer.")
        block_task = n_warp is not None and n_warp > 1
        if (
                block_task
                and threads_per_block is not None
                and threads_per_block != n_warp * WARP_SIZE
        ):
            raise SchedulerError(
                f"n_warp={n_warp} maps one task to {n_warp * WARP_SIZE} "
                f"threads, but threads_per_block={threads_per_block}."
            )
        if (
                n_warp == 1
                and thread_mapped
                and threads_per_block is not None
                and threads_per_block % WARP_SIZE
        ):
            raise SchedulerError(
                "n_warp=1 requires threads_per_block to be a multiple of "
                f"{WARP_SIZE}, but threads_per_block={threads_per_block}."
            )
        if threads_per_block is not None:
            block_sizes = (threads_per_block,)
        elif block_task:
            block_sizes = (n_warp * WARP_SIZE,)
        elif thread_mapped:
            block_sizes = BLOCK_SIZES
        else:
            block_sizes = (DEFAULT_THREADS_PER_BLOCK,)
        if thread_unroll is not None:
            unroll_factors = (thread_unroll,)
        elif thread_mapped:
            unroll_factors = UNROLL_FACTORS
        else:
            unroll_factors = (1,)
        execution_modes = EXECUTION_MODES if mode is None else (mode,)
        n_warps = N_WARP_CANDIDATES if n_warp is None else (n_warp,)
        available_protocols = (COUNTER_DEPENDENCE,) + (
            (direct_protocol,)
            if thread_mapped and direct_protocol is not None
            else ()
        )
        searched = len(execution_modes) * len(block_sizes) * len(
            unroll_factors
        ) * len(n_warps) > 1
        if dependence_protocol is not None:
            if dependence_protocol not in available_protocols:
                raise SchedulerError(
                    f"Dependence protocol {dependence_protocol!r} is not "
                    "supported by this loop; available protocols: "
                    + ", ".join(available_protocols)
                    + "."
                )
            dependence_protocols = (dependence_protocol,)
        elif PUSH_DEPENDENCE in available_protocols:
            dependence_protocols = (PUSH_DEPENDENCE,)
        elif PULL_DEPENDENCE in available_protocols and searched:
            dependence_protocols = (COUNTER_DEPENDENCE, PULL_DEPENDENCE)
        else:
            dependence_protocols = (COUNTER_DEPENDENCE,)
        return cls(
            execution_modes=execution_modes,
            block_sizes=block_sizes,
            unroll_factors=unroll_factors,
            n_warps=n_warps,
            thread_mapped=thread_mapped,
            dependence_protocols=dependence_protocols,
        )

    def configurations(self):
        configurations = []
        for execution_mode, threads_per_block, thread_unroll, n_warp in (
                itertools.product(
                    self.execution_modes,
                    self.block_sizes,
                    self.unroll_factors,
                    self.n_warps,
                )
        ):
            protocols = (
                self.dependence_protocols
                if (
                    execution_mode == SYNC_FREE
                    and self.thread_mapped
                    and _warp_task(threads_per_block, n_warp)
                )
                else (COUNTER_DEPENDENCE,)
            )
            configurations.extend(
                (
                    execution_mode,
                    threads_per_block,
                    thread_unroll,
                    n_warp,
                    protocol,
                )
                for protocol in protocols
            )
        return configurations

    @property
    def is_fixed(self):
        return len(self.configurations()) == 1

    def fixed_schedule(self):
        if not self.is_fixed:
            raise SchedulerError(
                "The schedule space has more than one configuration."
            )
        return Schedule(*self.configurations()[0])


class AnalyticScheduler:
    def __init__(self, arch_constants=None):
        self._arch_constants = arch_constants

    @property
    def arch_constants(self):
        if self._arch_constants is None:
            self._arch_constants = measure_arch_constants()
        return self._arch_constants

    def estimate(
            self,
            features,
            execution_mode,
            threads_per_block,
            thread_unroll,
            n_warp,
            thread_mapped,
            dependence_protocol=COUNTER_DEPENDENCE,
    ):
        arch = self.arch_constants
        average_in_degree = features.average_in_degree
        if execution_mode == WAVEFRONT:
            launch_s = features.levels * arch.launch_latency_s
            sync_s = 0.0
        elif execution_mode == SYNC_FREE:
            launch_s = arch.launch_latency_s
            if dependence_protocol == PUSH_DEPENDENCE:
                sync_s = arch.atomic_cost_s * features.rows / n_warp
            elif dependence_protocol == PULL_DEPENDENCE:
                sync_s = arch.atomic_cost_s * average_in_degree * (
                    1.0 + PULL_POLLING_FACTOR * features.levels
                )
            else:
                sync_s = arch.atomic_cost_s * (
                    average_in_degree + features.rows / n_warp
                )
        else:
            raise SchedulerError(
                f"Unknown execution mode {execution_mode!r}."
            )
        lanes = (
            task_threads(threads_per_block, n_warp) * thread_unroll
            if thread_mapped
            else 1
        )
        thread_parallelism = (
            features.average_thread_trip_count
            if thread_mapped and features.thread_loop_invocations
            else average_in_degree
        )
        active_lanes = min(lanes, max(thread_parallelism, 1.0))
        return CostEstimate(
            launch_s=launch_s,
            sync_s=sync_s,
            memory_s=features.memory_bytes / arch.memory_bandwidth_bytes_per_s,
            compute_s=features.flops / (active_lanes * arch.lane_flops_per_s),
        )

    def candidates(self, features, space):
        return [
            Schedule(
                execution_mode=execution_mode,
                threads_per_block=threads_per_block,
                thread_unroll=thread_unroll,
                n_warp=n_warp,
                dependence_protocol=dependence_protocol,
                cost=self.estimate(
                    features,
                    execution_mode,
                    threads_per_block,
                    thread_unroll,
                    n_warp,
                    space.thread_mapped,
                    dependence_protocol,
                ),
            )
            for (
                execution_mode,
                threads_per_block,
                thread_unroll,
                n_warp,
                dependence_protocol,
            ) in space.configurations()
        ]

    def schedule(self, features, space):
        return min(self.candidates(features, space), key=_schedule_rank)


def task_threads(threads_per_block, n_warp):
    if threads_per_block % WARP_SIZE:
        return threads_per_block
    return min(threads_per_block, WARP_SIZE * n_warp)


def _warp_task(threads_per_block, n_warp):
    return n_warp == 1 and threads_per_block % WARP_SIZE == 0


def _schedule_rank(schedule):
    return (
        schedule.cost.total_s,
        task_threads(schedule.threads_per_block, schedule.n_warp)
        * schedule.thread_unroll,
        abs(
            schedule.threads_per_block.bit_length()
            - DEFAULT_THREADS_PER_BLOCK.bit_length()
        ),
        schedule.threads_per_block,
        schedule.n_warp,
        EXECUTION_MODES.index(schedule.execution_mode),
        schedule.dependence_protocol,
    )


def extract_workload_features(
        dag_successors,
        wavefront_list,
        argument_values,
        function_props=None,
        flops_per_iteration=1,
        thread_trip_counts=None,
):
    rows = len(dag_successors)
    edges = sum(len(successors) for successors in dag_successors)
    nonzeros = _nonzeros_from_pointer_arrays(argument_values, function_props)
    if nonzeros is None:
        nonzeros = rows + edges
    trip_counts = (
        None
        if thread_trip_counts is None
        else tuple(max(int(count), 0) for count in thread_trip_counts)
    )
    thread_work_items = 0 if trip_counts is None else sum(trip_counts)
    thread_loop_invocations = 0 if trip_counts is None else len(trip_counts)
    work_items = nonzeros if trip_counts is None else thread_work_items
    return WorkloadFeatures(
        rows=rows,
        edges=edges,
        levels=len(wavefront_list),
        nonzeros=nonzeros,
        flops=float(flops_per_iteration * work_items),
        memory_bytes=_array_bytes(argument_values),
        thread_work_items=thread_work_items,
        thread_loop_invocations=thread_loop_invocations,
    )


def compile_thread_trip_counter(
        function_ast,
        thread_loop_iterator,
        function_globals=None,
):
    """Compile a side-effect-isolated probe for a thread-mapped loop's ranges.

    The probe follows the original outer control flow, including ``break`` and
    ``continue``, but replaces each execution of the mapped loop with the
    length of its ``range``. Its callers pass copies of mutable arguments so
    scalar work before the mapped loop cannot modify user inputs.
    """
    if thread_loop_iterator is None:
        return None
    module = deepcopy(function_ast)
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    if len(functions) != 1:
        raise SchedulerError(
            "Thread-loop trip counting requires exactly one function."
        )
    function = functions[0]
    transformer = _ThreadTripCounter(thread_loop_iterator)
    function.body = [
        transformer.visit(statement) for statement in function.body
    ]
    if transformer.matches == 0:
        raise SchedulerError(
            f"Could not find thread-mapped loop {thread_loop_iterator!r}."
        )
    if transformer.matches > 1:
        raise SchedulerError(
            "Thread-loop trip counting requires a unique loop iterator; "
            f"found {transformer.matches} loops named "
            f"{thread_loop_iterator!r}."
        )
    function.name = "_lcd_thread_trip_counter"
    function.decorator_list = []
    function.body.insert(
        0,
        ast.Assign(
            targets=[ast.Name(id="_lcd_trip_counts", ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        ),
    )
    function.body.append(_trip_count_return())
    module.body = [function]
    namespace = dict(function_globals or {})
    exec(
        compile(
            ast.fix_missing_locations(module),
            filename="<lcd-thread-trip-counter>",
            mode="exec",
        ),
        namespace,
    )
    return namespace[function.name]


def measure_thread_trip_counts(counter, argument_values, argument_names):
    """Execute a trip-count probe without exposing mutable user arguments."""
    if counter is None:
        return None
    arguments = [
        _trip_probe_copy(argument_values[name])
        for name in argument_names
    ]
    return tuple(counter(*arguments))


class _ThreadTripCounter(ast.NodeTransformer):
    def __init__(self, iterator):
        self.iterator = iterator
        self.matches = 0

    def visit_For(self, node):
        if not (
                isinstance(node.target, ast.Name)
                and node.target.id == self.iterator
        ):
            return self.generic_visit(node)
        self.matches += 1
        if node.orelse:
            raise SchedulerError(
                "Thread-mapped loops with for/else cannot be trip-counted."
            )
        if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and not node.iter.keywords
                and 1 <= len(node.iter.args) <= 3
        ):
            raise SchedulerError(
                "Thread-loop trip counting requires a range-based loop."
            )
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lcd_trip_counts", ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[deepcopy(node.iter)],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        )

    def visit_Return(self, node):
        return _trip_count_return()


def _trip_count_return():
    return ast.Return(
        value=ast.Name(id="_lcd_trip_counts", ctx=ast.Load())
    )


def _trip_probe_copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (list, dict, set)):
        return deepcopy(value)
    return value


def pointer_array_names(function_props):
    return [
        name
        for name, properties in (function_props or {}).items()
        if isinstance(name, str) and "monotonically_increasing" in properties
    ]


def _nonzeros_from_pointer_arrays(argument_values, function_props):
    counts = []
    for name in pointer_array_names(function_props):
        value = argument_values.get(name)
        if value is None:
            continue
        pointers = np.asarray(value)
        if pointers.ndim == 1 and pointers.size >= 2:
            counts.append(int(pointers[-1]) - int(pointers[0]))
    return max(counts) if counts else None


def _array_bytes(argument_values):
    arrays = {
        id(value): value
        for value in argument_values.values()
        if isinstance(value, np.ndarray)
    }
    return int(sum(array.nbytes for array in arrays.values()))


def count_inner_loop_flops(function_ast, block_iterator, thread_loop_iterator=None):
    block_loop = _find_loop(function_ast, block_iterator)
    if block_loop is None:
        return 1
    inner_loop = None
    if thread_loop_iterator is not None:
        inner_loop = _find_loop(block_loop, thread_loop_iterator)
    if inner_loop is None:
        inner_loop = _deepest_loop(block_loop)
    counter = _ArithmeticCounter()
    for statement in inner_loop.body:
        counter.visit(statement)
    return max(counter.count, 1)


def _find_loop(tree, iterator):
    for node in ast.walk(tree):
        if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == iterator
        ):
            return node
    return None


def _deepest_loop(loop):
    deepest_depth, deepest_loop = 0, loop
    pending = deque([(loop, 0)])
    while pending:
        node, depth = pending.popleft()
        for child in ast.iter_child_nodes(node):
            child_depth = depth + isinstance(child, ast.For)
            if isinstance(child, ast.For) and child_depth > deepest_depth:
                deepest_depth, deepest_loop = child_depth, child
            pending.append((child, child_depth))
    return deepest_loop


class _ArithmeticCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0

    def visit_BinOp(self, node):
        if isinstance(node.op, ARITHMETIC_OPERATORS):
            self.count += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ARITHMETIC_OPERATORS):
            self.count += 1
        self.visit(node.value)

    def visit_Subscript(self, node):
        self.visit(node.value)

    def visit_For(self, node):
        for statement in node.body:
            self.visit(statement)


def inspection_fingerprint(values, content_names):
    content = tuple(
        (name, _value_digest(values.get(name)))
        for name in sorted(set(content_names))
    )
    shapes = tuple(
        (name, value.dtype.str, value.shape)
        for name, value in sorted(values.items())
        if isinstance(value, np.ndarray)
    )
    return content, shapes


def _value_digest(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return object()
        contiguous = np.ascontiguousarray(value)
        return (
            contiguous.dtype.str,
            contiguous.shape,
            hashlib.blake2b(
                contiguous.reshape(-1).view(np.uint8),
                digest_size=16,
            ).digest(),
        )
    if isinstance(value, (list, tuple)):
        try:
            return _value_digest(np.asarray(value))
        except (TypeError, ValueError):
            return object()
    if value is None or np.isscalar(value):
        return type(value).__name__, value
    return object()


def measure_arch_constants():
    try:
        from numba import config, cuda
        if config.ENABLE_CUDASIM or not cuda.is_available():
            return FALLBACK_ARCH_CONSTANTS
        device_id = cuda.get_current_device().id
    except Exception:
        return FALLBACK_ARCH_CONSTANTS
    return _measure_device_constants(device_id)


@lru_cache(maxsize=None)
def _measure_device_constants(device_id):
    from numba import cuda
    device = cuda.get_current_device()
    memory_bandwidth = (
        2.0 * device.MEMORY_CLOCK_RATE * 1.0e3
        * device.GLOBAL_MEMORY_BUS_WIDTH / 8.0
    )
    return ArchConstants(
        launch_latency_s=_measure_launch_latency(cuda),
        atomic_cost_s=_measure_atomic_cost(cuda),
        memory_bandwidth_bytes_per_s=memory_bandwidth,
        lane_flops_per_s=device.CLOCK_RATE * 1.0e3,
    )


def _measure_launch_latency(cuda, repeats=64):
    @cuda.jit
    def empty_kernel():
        return

    empty_kernel[1, 1]()
    cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        empty_kernel[1, 1]()
        cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _measure_atomic_cost(cuda, nodes=1 << 16, repeats=5, stride=7919):
    @cuda.jit
    def release_kernel(counters, stride):
        if cuda.threadIdx.x == 0:
            node = cuda.blockIdx.x
            cuda.atomic.sub(counters, (node * stride) % counters.shape[0], 1)

    counters = cuda.to_device(np.zeros(nodes, dtype=np.int32))
    release_kernel[nodes, 32](counters, stride)
    cuda.synchronize()
    start, stop = cuda.event(), cuda.event()
    samples = []
    for _ in range(repeats):
        start.record()
        release_kernel[nodes, 32](counters, stride)
        stop.record()
        stop.synchronize()
        samples.append(cuda.event_elapsed_time(start, stop) * 1.0e-3 / nodes)
    return statistics.median(samples)

