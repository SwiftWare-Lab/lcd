import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from compiler.cuda_lowering import (
    CUDALoweringError,
    UnsupportedAtomicOperation,
    lower_to_numba_cuda,
)
from compiler.cuda_runtime import (
    compile_cuda_module,
    execute_cuda_call,
    prepare_cuda_arguments,
)
from compiler.dag_preprocessing import (
    SyncFreeExecutionData,
    WavefrontExecutionData,
)
from compiler.spf_raising import build_spf_from_ast, extract_node_data


SPTRSV_SOURCE = Path(__file__).parents[1] / "demo" / "sptrsv.py"
SPILU0_SOURCE = Path(__file__).parents[1] / "demo" / "spilu0.py"
SPILU0_FUNCTION_PROPS = {
    "A_indptr": ["monotonically_increasing"],
    "A_indices": [],
    ("A_indptr", "A_indices"): [
        ("segmented_monotonically_increasing", "n"),
    ],
}


def _load_function(function_name, source=SPTRSV_SOURCE):
    module = ast.parse(source.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return ast.Module(body=[function], type_ignores=[])


def _reduction_statements(module, *statement_ids):
    node_data = extract_node_data(build_spf_from_ast(module))
    return [node_data[statement_id] for statement_id in statement_ids]


class _FakeDeviceArray:
    def __init__(self, host_value):
        self.host_value = np.asarray(host_value).copy()

    @property
    def __cuda_array_interface__(self):
        return {}

    def copy_to_host(self):
        return self.host_value.copy()


class _FakeCUDA:
    def __init__(self):
        self.copied_values = []
        self.synchronize_count = 0

    def to_device(self, value):
        device_value = _FakeDeviceArray(value)
        self.copied_values.append(device_value)
        return device_value

    def synchronize(self):
        self.synchronize_count += 1


class _FakeCUDACompiler:
    @staticmethod
    def jit(function):
        return function


class CUDALoweringTests(unittest.TestCase):
    def test_lowers_csr_outer_loop_to_one_cuda_block(self):
        result = lower_to_numba_cuda(
            _load_function("simple_sptrsv_csr"),
            loop_iterator="i",
        )

        self.assertEqual(result.kernel_name, "simple_sptrsv_csr_cuda_kernel")
        self.assertEqual(
            result.kernel_arguments,
            ["Ap", "Ai", "Ax", "b", "n", "x"],
        )
        self.assertEqual(result.lifted_variables, ["x"])
        self.assertEqual(len(result.host_prelude), 1)
        self.assertEqual(len(result.host_epilogue), 1)
        self.assertIn("i = cuda.blockIdx.x", result.source)
        self.assertIn(
            "for j in range(Ap[i], Ap[i + 1] - 1):",
            result.source,
        )
        self.assertNotIn("x = np.zeros(n)", result.source)
        self.assertNotIn("return x", result.source)
        self.assertNotIn("cuda.atomic", result.source)
        compile(result.module_ast, "<csr-cuda-lowering>", "exec")

    def test_lowers_csc_reduction_to_atomic_add(self):
        module = _load_function("simple_sptrsv_csc")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            reductions=_reduction_statements(module, "S2"),
        )

        self.assertEqual(result.atomic_statement_ids, {"S2"})
        self.assertIn(
            "cuda.atomic.add(x, Ai[j], Ax[j] * x[i])",
            result.source,
        )
        self.assertNotIn("x[Ai[j]] +=", result.source)
        compile(result.module_ast, "<csc-cuda-lowering>", "exec")

    def test_adds_sync_free_wait_and_successor_updates(self):
        module = _load_function("simple_sptrsv_csc")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            reductions=_reduction_statements(module, "S2"),
            execution_mode="sync_free",
        )

        self.assertEqual(result.execution_mode, "sync_free")
        self.assertEqual(
            result.auxiliary_arguments,
            ["dep_count", "dag_ptr", "dag_index"],
        )
        self.assertEqual(
            result.kernel_arguments[-3:],
            ["dep_count", "dag_ptr", "dag_index"],
        )
        self.assertIn(
            "while cuda.atomic.add(dep_count, i, 0) != 0:",
            result.source,
        )
        self.assertIn("_dag_start = dag_ptr[i]", result.source)
        self.assertIn("_dag_end = dag_ptr[i + 1]", result.source)
        self.assertIn(
            "for _dag_position in range(_dag_start, _dag_end):",
            result.source,
        )
        self.assertIn(
            "cuda.atomic.add(dep_count, _dag_successor, -1)",
            result.source,
        )
        self.assertIn(
            "cuda.atomic.add(x, Ai[j], Ax[j] * x[i])",
            result.source,
        )
        self.assertEqual(
            result.launcher_name,
            "simple_sptrsv_csc_cuda_launch",
        )
        self.assertIn(
            "_blocks_per_grid = len(dep_count)",
            result.source,
        )
        self.assertIn("_threads_per_block = 1", result.source)
        self.assertIn(
            "simple_sptrsv_csc_cuda_kernel"
            "[_blocks_per_grid, _threads_per_block]"
            "(Ap, Ai, Ax, b, n, x, dep_count, dag_ptr, dag_index)",
            result.source,
        )
        compile(result.module_ast, "<csc-sync-free-lowering>", "exec")

    def test_sync_free_readiness_count_can_include_self_edge(self):
        result = lower_to_numba_cuda(
            _load_function("simple_sptrsv_csr"),
            loop_iterator="i",
            execution_mode="sync_free",
            sync_free_ready_count=1,
        )

        self.assertIn(
            "while cuda.atomic.add(dep_count, i, 0) != 1:",
            result.source,
        )

    def test_maps_wavefront_location_to_original_iterator(self):
        module = _load_function("simple_sptrsv_csc")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            reductions=_reduction_statements(module, "S2"),
            execution_mode="wavefront",
        )

        self.assertEqual(result.execution_mode, "wavefront")
        self.assertEqual(
            result.auxiliary_arguments,
            ["wavefronts", "wave_start", "wave_end"],
        )
        self.assertEqual(
            result.kernel_arguments[-3:],
            ["wavefronts", "wave_start", "wave_end"],
        )
        self.assertIn(
            "_wavefront_location = cuda.grid(1)",
            result.source,
        )
        self.assertIn(
            "if wave_start + _wavefront_location < wave_end:",
            result.source,
        )
        self.assertIn(
            "i = wavefronts[wave_start + _wavefront_location]",
            result.source,
        )
        self.assertIn("if i < n:", result.source)
        self.assertIn(
            "cuda.atomic.add(x, Ai[j], Ax[j] * x[i])",
            result.source,
        )
        self.assertNotIn("dep_count", result.source)
        self.assertNotIn("dag_ptr", result.source)
        self.assertEqual(
            result.launcher_arguments[-2:],
            ["wavefronts", "wavefront_ptr"],
        )
        self.assertIn(
            "for _wavefront in range(len(wavefront_ptr) - 1):",
            result.source,
        )
        self.assertIn(
            "_blocks_per_grid = (_wave_size + "
            "(_threads_per_block - 1)) // _threads_per_block",
            result.source,
        )
        self.assertIn("cuda.synchronize()", result.source)
        compile(result.module_ast, "<csc-wavefront-lowering>", "exec")

    def test_thread_maps_wavefront_inner_loop_without_unrolling_for_factor_one(self):
        module = _load_function("simple_sptrsv_csc")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            reductions=_reduction_statements(module, "S2"),
            execution_mode="wavefront",
            thread_loop_iterator="j",
            thread_unroll_factor=1,
        )

        self.assertEqual(result.thread_loop_iterator, "j")
        self.assertEqual(result.thread_unroll_factor, 1)
        self.assertIn(
            "_wavefront_location = cuda.blockIdx.x",
            result.source,
        )
        self.assertNotIn("_wavefront_location = cuda.grid(1)", result.source)
        self.assertIn("_thread_id = cuda.threadIdx.x", result.source)
        self.assertIn("if _thread_id == 0:", result.source)
        self.assertIn("cuda.syncthreads()", result.source)
        self.assertIn(
            "for j in range(Ap[i] + 1 + _thread_id, "
            "Ap[i + 1], cuda.blockDim.x):",
            result.source,
        )
        self.assertIn("_blocks_per_grid = _wave_size", result.source)
        self.assertNotIn("_j_base", result.source)
        self.assertNotIn("cuda.blockDim.x * 1", result.source)
        compile(result.module_ast, "<csc-thread-map-u1-lowering>", "exec")

    def test_unrolls_thread_mapped_loop_for_factor_greater_than_one(self):
        module = _load_function("simple_sptrsv_csc")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            reductions=_reduction_statements(module, "S2"),
            execution_mode="wavefront",
            thread_loop_iterator="j",
            thread_unroll_factor=4,
        )

        self.assertEqual(result.thread_unroll_factor, 4)
        self.assertIn(
            "for _j_base in range(Ap[i] + 1 + _thread_id, "
            "Ap[i + 1], cuda.blockDim.x * 4):",
            result.source,
        )
        self.assertIn("j = _j_base", result.source)
        self.assertIn("j = _j_base + cuda.blockDim.x", result.source)
        self.assertIn("j = _j_base + cuda.blockDim.x * 2", result.source)
        self.assertIn("j = _j_base + cuda.blockDim.x * 3", result.source)
        self.assertEqual(
            result.source.count("cuda.atomic.add(x, Ai[j], Ax[j] * x[i])"),
            4,
        )
        compile(result.module_ast, "<csc-thread-map-u4-lowering>", "exec")

    def test_thread_maps_loop_nested_below_sequential_pivot_loop(self):
        result = lower_to_numba_cuda(
            _load_function("ilu0_csr", SPILU0_SOURCE),
            loop_iterator="i",
            execution_mode="wavefront",
            thread_loop_iterator="j_ptr",
            wavefront_threads_per_block=128,
            function_props=SPILU0_FUNCTION_PROPS,
        )

        self.assertIn(
            "for k_ptr in range(A_indptr[i], A_indptr[i + 1]):",
            result.source,
        )
        self.assertIn(
            "for j_ptr in range(k_ptr + 1 + _thread_id, "
            "A_indptr[i + 1], cuda.blockDim.x):",
            result.source,
        )
        self.assertIn("if _thread_id == 0:", result.source)
        self.assertIn(
            "A_data[k_ptr] = A_data[k_ptr] / diag_val",
            result.source,
        )
        self.assertIn(
            "_lcd_shared_scalar_0 = cuda.shared.array(1, ",
            result.source,
        )
        self.assertIn(
            "_lcd_shared_scalar_0[0] = A_data[k_ptr]",
            result.source,
        )
        self.assertIn(
            "_lcd_shared_scalar_0[0] * A_data[kj_ptr]",
            result.source,
        )
        self.assertNotIn(
            "for kj_ptr in range(A_indptr[A_indices[k_ptr]]",
            result.source,
        )
        self.assertIn(
            "while _lcd_binary_search_0_left < "
            "_lcd_binary_search_0_right:",
            result.source,
        )
        self.assertIn(
            "A_indices[_lcd_binary_search_0_middle] < "
            "_lcd_binary_search_0_target",
            result.source,
        )
        self.assertIn("cuda.syncthreads()", result.source)
        self.assertNotIn(
            "cuda.syncthreads()\n                    cuda.syncthreads()",
            result.source,
        )
        self.assertIn("_blocks_per_grid = _wave_size", result.source)
        self.assertNotIn("_threads_per_block = 1\n", result.source)
        compile(result.module_ast, "<nested-thread-map-lowering>", "exec")

    def test_keeps_linear_search_without_segmented_sorted_property(self):
        result = lower_to_numba_cuda(
            _load_function("ilu0_csr", SPILU0_SOURCE),
            loop_iterator="i",
            execution_mode="wavefront",
            thread_loop_iterator="j_ptr",
            wavefront_threads_per_block=128,
        )

        self.assertIn(
            "for kj_ptr in range(A_indptr[A_indices[k_ptr]], "
            "A_indptr[A_indices[k_ptr] + 1]):",
            result.source,
        )
        self.assertNotIn("_lcd_binary_search_0_left", result.source)

    def test_unrolls_nested_binary_search_update(self):
        result = lower_to_numba_cuda(
            _load_function("ilu0_csr", SPILU0_SOURCE),
            loop_iterator="i",
            execution_mode="wavefront",
            thread_loop_iterator="j_ptr",
            thread_unroll_factor=4,
            wavefront_threads_per_block=128,
            function_props=SPILU0_FUNCTION_PROPS,
        )

        self.assertIn("cuda.blockDim.x * 4", result.source)
        self.assertEqual(
            result.source.count(
                "while _lcd_binary_search_0_left < "
                "_lcd_binary_search_0_right:"
            ),
            4,
        )
        compile(result.module_ast, "<nested-thread-map-u4>", "exec")

    def test_maps_nested_loop_to_blocks_inside_host_loop(self):
        module = ast.parse("""
def nested_wavefront(A, n, steps):
    for sweep in range(steps):
        for i in range(n):
            A[i] = A[i] + sweep
    return A
""")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            execution_mode="wavefront",
        )

        self.assertEqual(result.host_loop_iterators, ("sweep",))
        self.assertEqual(
            result.kernel_arguments,
            [
                "A",
                "n",
                "steps",
                "sweep",
                "wavefronts",
                "wave_start",
                "wave_end",
            ],
        )
        self.assertEqual(
            result.launcher_arguments,
            ["A", "n", "steps", "wavefronts", "wavefront_ptr"],
        )
        self.assertIn("for sweep in range(steps):", result.source)
        self.assertIn(
            "nested_wavefront_cuda_kernel[_blocks_per_grid, "
            "_threads_per_block](A, n, steps, sweep, wavefronts, "
            "_wave_start, _wave_end)",
            result.source,
        )
        self.assertIn("A[i] = A[i] + sweep", result.source)
        compile(result.module_ast, "<nested-wavefront-lowering>", "exec")

    def test_normalizes_nested_block_loop_alias_sibling(self):
        module = ast.parse("""
def nested_with_sibling(A, n, steps):
    for sweep in range(steps):
        before = sweep
        for i in range(n):
            A[i] = A[i] + before
    return A
""")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            execution_mode="wavefront",
        )

        self.assertNotIn("before = sweep", result.source)
        self.assertIn("A[i] = A[i] + sweep", result.source)

    def test_rejects_nested_block_loop_with_unhandled_host_siblings(self):
        module = ast.parse("""
def nested_with_sibling(A, n, steps):
    for sweep in range(steps):
        before = transform(sweep)
        for i in range(n):
            A[i] = A[i] + before
    return A
""")
        with self.assertRaisesRegex(
                CUDALoweringError,
                "enclosing loop body",
        ):
            lower_to_numba_cuda(
                module,
                loop_iterator="i",
                execution_mode="wavefront",
            )

    def test_resets_dep_count_for_nested_sync_free_host_loop(self):
        module = ast.parse("""
def nested_sync_free(A, n, steps):
    for sweep in range(steps):
        for i in range(n):
            A[i] = A[i] + sweep
    return A
""")
        result = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            execution_mode="sync_free",
        )

        self.assertEqual(result.host_loop_iterators, ("sweep",))
        self.assertEqual(
            result.auxiliary_arguments,
            ["dep_count", "initial_dep_count", "dag_ptr", "dag_index"],
        )
        self.assertEqual(
            result.kernel_arguments,
            [
                "A",
                "n",
                "steps",
                "sweep",
                "dep_count",
                "initial_dep_count",
                "dag_ptr",
                "dag_index",
            ],
        )
        self.assertEqual(
            result.launcher_arguments,
            [
                "A",
                "n",
                "steps",
                "dep_count",
                "initial_dep_count",
                "dag_ptr",
                "dag_index",
            ],
        )
        self.assertIn(
            "def nested_sync_free_dep_count_reset_cuda_kernel"
            "(dep_count, initial_dep_count):",
            result.source,
        )
        self.assertIn(
            "dep_count[_dep_count_index] = "
            "initial_dep_count[_dep_count_index]",
            result.source,
        )
        self.assertIn("for sweep in range(steps):", result.source)
        self.assertIn(
            "nested_sync_free_dep_count_reset_cuda_kernel"
            "[_reset_blocks_per_grid, _reset_threads_per_block]"
            "(dep_count, initial_dep_count)",
            result.source,
        )
        self.assertIn(
            "nested_sync_free_cuda_kernel[_blocks_per_grid, "
            "_threads_per_block](A, n, steps, sweep, dep_count, "
            "initial_dep_count, dag_ptr, dag_index)",
            result.source,
        )
        self.assertIn("cuda.synchronize()", result.source)
        compile(result.module_ast, "<nested-sync-free-lowering>", "exec")

    def test_prepares_sync_free_device_arrays_and_lifted_output(self):
        lowering = lower_to_numba_cuda(
            _load_function("simple_sptrsv_csr"),
            loop_iterator="i",
            execution_mode="sync_free",
        )
        fake_cuda = _FakeCUDA()
        prepared = prepare_cuda_arguments(
            lowering=lowering,
            argument_values={
                "Ap": np.array([0, 1, 3], dtype=np.int32),
                "Ai": np.array([0, 0, 1], dtype=np.int32),
                "Ax": np.ones(3),
                "b": np.ones(2),
                "n": 2,
            },
            execution_data=SyncFreeExecutionData(
                dep_count=np.array([0, 1], dtype=np.int32),
                dag_ptr=np.array([0, 1, 1], dtype=np.int32),
                dag_index=np.array([1], dtype=np.int32),
            ),
            function_globals={"np": np},
            cuda_module=fake_cuda,
        )

        self.assertEqual(
            prepared.launcher_argument_names[-3:],
            ("dep_count", "dag_ptr", "dag_index"),
        )
        self.assertEqual(prepared.values["n"], 2)
        self.assertIsInstance(prepared.values["x"], _FakeDeviceArray)
        np.testing.assert_array_equal(
            prepared.copy_lifted_variables_to_host()["x"],
            np.zeros(2),
        )
        self.assertEqual(len(prepared.launcher_arguments), 9)

    def test_prepares_initial_dep_count_for_nested_sync_free_reset(self):
        module = ast.parse("""
def nested_sync_free(A, n, steps):
    for sweep in range(steps):
        for i in range(n):
            A[i] = A[i] + sweep
    return A
""")
        lowering = lower_to_numba_cuda(
            module,
            loop_iterator="i",
            execution_mode="sync_free",
        )
        fake_cuda = _FakeCUDA()
        prepared = prepare_cuda_arguments(
            lowering=lowering,
            argument_values={
                "A": np.zeros(2),
                "n": 2,
                "steps": 3,
            },
            execution_data=SyncFreeExecutionData(
                dep_count=np.array([0, 1], dtype=np.int32),
                dag_ptr=np.array([0, 1, 1], dtype=np.int32),
                dag_index=np.array([1], dtype=np.int32),
            ),
            function_globals={},
            cuda_module=fake_cuda,
        )

        self.assertIn(
            "initial_dep_count",
            prepared.launcher_argument_names,
        )
        np.testing.assert_array_equal(
            prepared.values["initial_dep_count"].copy_to_host(),
            [0, 1],
        )
        self.assertIsNot(
            prepared.values["dep_count"],
            prepared.values["initial_dep_count"],
        )

    def test_compiles_and_binds_kernel_and_launcher(self):
        lowering = lower_to_numba_cuda(
            _load_function("simple_sptrsv_csr"),
            loop_iterator="i",
            execution_mode="sync_free",
        )
        compiled_module = compile_cuda_module(
            lowering,
            function_globals={"np": np},
            cuda_module=_FakeCUDACompiler(),
        )

        self.assertTrue(callable(compiled_module.kernel))
        self.assertTrue(callable(compiled_module.launcher))
        self.assertIs(
            compiled_module.namespace[lowering.kernel_name],
            compiled_module.kernel,
        )
        self.assertIs(
            compiled_module.launcher.__globals__[lowering.kernel_name],
            compiled_module.kernel,
        )

    def test_executes_launcher_and_restores_original_return(self):
        lowering = lower_to_numba_cuda(
            _load_function("simple_sptrsv_csr"),
            loop_iterator="i",
            execution_mode="sync_free",
        )
        fake_cuda = _FakeCUDA()
        prepared = prepare_cuda_arguments(
            lowering=lowering,
            argument_values={
                "Ap": np.array([0], dtype=np.int32),
                "Ai": np.array([], dtype=np.int32),
                "Ax": np.array([], dtype=np.float64),
                "b": np.array([7.0, 8.0]),
                "n": 2,
            },
            execution_data=SyncFreeExecutionData(
                dep_count=np.array([0, 0], dtype=np.int32),
                dag_ptr=np.array([0, 0, 0], dtype=np.int32),
                dag_index=np.array([], dtype=np.int32),
            ),
            function_globals={"np": np},
            cuda_module=fake_cuda,
        )

        def fake_launcher(*launcher_arguments):
            prepared.values["x"].host_value[:] = [7.0, 8.0]

        result = execute_cuda_call(
            compiled_module=SimpleNamespace(
                launcher=fake_launcher,
                namespace={"cuda": fake_cuda},
            ),
            lowering=lowering,
            prepared_arguments=prepared,
            function_globals={"np": np},
        )

        np.testing.assert_array_equal(result, [7.0, 8.0])
        self.assertEqual(fake_cuda.synchronize_count, 1)

    def test_copies_returned_argument_arrays_back_to_host(self):
        module = ast.parse("""
def return_mutated_argument(A, n):
    for i in range(n):
        A[i] += 1
    return A
""")
        lowering = lower_to_numba_cuda(
            module,
            loop_iterator="i",
        )
        fake_cuda = _FakeCUDA()
        prepared = prepare_cuda_arguments(
            lowering=lowering,
            argument_values={
                "A": np.array([1.0, 2.0]),
                "n": 2,
            },
            execution_data=None,
            function_globals={},
            cuda_module=fake_cuda,
        )

        def fake_launcher(*launcher_arguments):
            prepared.values["A"].host_value[:] = [2.0, 3.0]

        result = execute_cuda_call(
            compiled_module=SimpleNamespace(
                launcher=fake_launcher,
                namespace={"cuda": fake_cuda},
            ),
            lowering=lowering,
            prepared_arguments=prepared,
            function_globals={},
        )

        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_keeps_wavefront_pointer_on_host(self):
        lowering = lower_to_numba_cuda(
            _load_function("simple_sptrsv_csr"),
            loop_iterator="i",
            execution_mode="wavefront",
        )
        wavefront_ptr = np.array([0, 1, 2], dtype=np.int32)
        prepared = prepare_cuda_arguments(
            lowering=lowering,
            argument_values={
                "Ap": np.array([0, 1, 3], dtype=np.int32),
                "Ai": np.array([0, 0, 1], dtype=np.int32),
                "Ax": np.ones(3),
                "b": np.ones(2),
                "n": 2,
            },
            execution_data=WavefrontExecutionData(
                wavefronts=np.array([0, 1], dtype=np.int32),
                wavefront_ptr=wavefront_ptr,
            ),
            function_globals={"np": np},
            cuda_module=_FakeCUDA(),
        )

        self.assertIs(prepared.values["wavefront_ptr"], wavefront_ptr)
        self.assertIsInstance(
            prepared.values["wavefronts"],
            _FakeDeviceArray,
        )

    def test_rejects_missing_or_unsupported_reduction(self):
        product_source = """
def product(A, Ai, B, n):
    for i in range(n):
        A[Ai[i]] *= B[i]
"""
        product_module = ast.parse(product_source)
        with self.assertRaises(UnsupportedAtomicOperation):
            lower_to_numba_cuda(
                product_module,
                loop_iterator="i",
                reductions=_reduction_statements(product_module, "S0"),
            )

        missing_module = ast.parse("""
def missing(A, n):
    for i in range(n):
        A[i] += 1
""")
        unrelated_module = ast.parse("""
def unrelated(A, n):
    temporary = 0
    for i in range(n):
        A[i] += 1
""")
        with self.assertRaisesRegex(
                CUDALoweringError,
                "were not found",
        ):
            lower_to_numba_cuda(
                missing_module,
                loop_iterator="i",
                reductions=_reduction_statements(
                    unrelated_module,
                    "S1",
                ),
            )

    def test_rejects_thread_unroll_without_thread_mapped_loop(self):
        with self.assertRaisesRegex(
                CUDALoweringError,
                "requires a thread-mapped loop",
        ):
            lower_to_numba_cuda(
                _load_function("simple_sptrsv_csc"),
                loop_iterator="i",
                thread_unroll_factor=2,
            )


if __name__ == "__main__":
    unittest.main()
