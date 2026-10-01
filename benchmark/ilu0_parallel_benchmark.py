"""Benchmark a synthetic ILU0-like CSR row-dependency recurrence, not numeric ILU0.

The E2E GPU cases include DAG preparation, host/device transfers, kernel launches,
and result retrieval; Matrix Market file loading is outside every timed case.
Prepared GPU cases time execution with device data already resident. The prepared
CPU case reuses its output buffer and the already loaded CSR arrays.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.io as scio
import triton
from numba import cuda

from compiler.transform_tests.ilu0 import (
    ilu0_csr_row_dependencies,
    ilu0_csr_row_dependencies_sync_free,
)
NUM_RUNS = 30
WARMUP_RUNS = 5
BENCHMARK_DIR = Path(__file__).resolve().parent
SUITESPARSE_MATRIX_DIR = BENCHMARK_DIR / "matrices" / "suitesparse"
SUITESPARSE_MANIFEST = SUITESPARSE_MATRIX_DIR / "manifest.txt"
RESULTS_DIR = BENCHMARK_DIR / "results"


def discover_suitesparse_matrices():
    if SUITESPARSE_MANIFEST.exists():
        matrix_paths = []
        with SUITESPARSE_MANIFEST.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                relative_path = line.strip()
                if not relative_path or relative_path.startswith("#"):
                    continue
                matrix_path = SUITESPARSE_MATRIX_DIR / relative_path
                if not matrix_path.is_file():
                    raise FileNotFoundError(f"Manifest matrix is missing: {matrix_path}")
                matrix_paths.append(str(matrix_path))
        if matrix_paths:
            return matrix_paths

    if SUITESPARSE_MATRIX_DIR.exists():
        matrix_paths = sorted(
            str(path) for path in SUITESPARSE_MATRIX_DIR.rglob("*.mtx")
        )
        if matrix_paths:
            return matrix_paths

    return []


downloaded_mtx_list = discover_suitesparse_matrices()
mtx_list = downloaded_mtx_list

print(f"Using {len(downloaded_mtx_list)} downloaded SuiteSparse matrices.")

method_list = [
    "wavefront-e2e",
    "wavefront-prepared",
    "sync-free-e2e",
    "sync-free-prepared",
    "cpu-row-deps",
    "cpu-row-deps-prepared",
]

configs = [
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=mtx_list,
        line_arg="provider",
        line_vals=method_list,
        line_names=[
            "Wavefront E2E",
            "Wavefront Prepared",
            "Sync-free E2E",
            "Sync-free Prepared",
            "CPU Row Dependencies",
            "CPU Prepared",
        ],
        styles=[
            ("red", "-"),
            ("red", "--"),
            ("purple", "-"),
            ("purple", "--"),
            ("orange", "--"),
            ("orange", "-"),
        ],
        ylabel="Rows/s",
        plot_name="ilu0-row-dependencies-validated",
        args={},
    )
]


def cpu_ilu0_csr_row_dependencies_into(Ap, Ai, out):
    for i in range(len(Ap) - 1):
        row_accum = 0.0
        scale = 0.5 / (Ap[i + 1] - Ap[i] + 1)
        for k_ptr in range(Ap[i], Ap[i + 1]):
            if Ai[k_ptr] < i:
                row_accum += scale * out[Ai[k_ptr]]
        out[i] = row_accum + 1.0
    return out


def cpu_ilu0_csr_row_dependencies(Ap, Ai, row_done):
    return cpu_ilu0_csr_row_dependencies_into(Ap, Ai, row_done.copy())


def load_matrix(matrix_path):
    if not os.path.isfile(matrix_path):
        raise FileNotFoundError(f"Benchmark matrix is missing: {matrix_path}")
    matrix = scio.mmread(matrix_path).tocsr()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ILU0 benchmark requires a square matrix: {matrix_path}")
    matrix.sort_indices()
    return matrix


validated_matrices = set()


def validate_matrix(matrix, matrix_path):
    """Reject incorrect GPU schedules before reporting their throughput."""
    n = matrix.shape[0]
    Ap = matrix.indptr.astype(np.int32)
    Ai = matrix.indices.astype(np.int32)
    initial = np.zeros(n, dtype=np.float64)
    reference = cpu_ilu0_csr_row_dependencies(Ap, Ai, initial)
    if not np.isfinite(reference).all():
        raise AssertionError(f"Non-finite CPU reference for {matrix_path}")

    prepared_cpu_out = np.empty_like(initial)
    for trial in range(2):
        cpu_ilu0_csr_row_dependencies_into(Ap, Ai, prepared_cpu_out)
        np.testing.assert_allclose(
            prepared_cpu_out, reference, rtol=1e-12, atol=1e-12,
            err_msg=f"CPU prepared trial {trial + 1} is incorrect for {matrix_path}",
        )

    for name, implementation in (
        ("wavefront", ilu0_csr_row_dependencies),
        ("sync-free", ilu0_csr_row_dependencies_sync_free),
    ):
        actual = implementation(Ap, Ai, initial.copy(), n)
        np.testing.assert_allclose(
            actual, reference, rtol=1e-12, atol=1e-12,
            err_msg=f"{name} end-to-end result is incorrect for {matrix_path}",
        )

        prepared = implementation.prepare_cuda_arguments(Ap, Ai, initial.copy(), n)
        initial_dep_count = prepared.host_values.get("dep_count")
        for trial in range(2):
            prepared.values["row_done"].copy_to_device(initial)
            if initial_dep_count is not None:
                prepared.values["dep_count"].copy_to_device(initial_dep_count)
            implementation.cuda_launcher(*prepared.launcher_arguments)
            cuda.synchronize()
            actual = prepared.values["row_done"].copy_to_host()
            np.testing.assert_allclose(
                actual, reference, rtol=1e-12, atol=1e-12,
                err_msg=f"{name} prepared trial {trial + 1} is incorrect for {matrix_path}",
            )
    print(f"Validated GPU results against CPU reference: {matrix_path}")


@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    matrix = load_matrix(matrices)
    if matrices not in validated_matrices:
        validate_matrix(matrix, matrices)
        validated_matrices.add(matrices)
    n = matrix.shape[0]
    Ap = matrix.indptr.astype(np.int32)
    Ai = matrix.indices.astype(np.int32)
    row_done = np.zeros(n, dtype=np.float64)
    rep, warmup = NUM_RUNS, WARMUP_RUNS

    if provider == "wavefront-e2e":
        def run_parallel_row_deps():
            ilu0_csr_row_dependencies(Ap, Ai, row_done.copy(), n)

        ms, min_ms, max_ms = do_wall_bench(
            run_parallel_row_deps,
            warmup=warmup,
            rep=rep,
        )
    elif provider == "wavefront-prepared":
        prepared = ilu0_csr_row_dependencies.prepare_cuda_arguments(
            Ap,
            Ai,
            row_done.copy(),
            n,
        )

        def run_parallel_row_deps_prepared():
            ilu0_csr_row_dependencies.cuda_launcher(
                *prepared.launcher_arguments
            )
            cuda.synchronize()

        ms, min_ms, max_ms = do_wall_bench(
            run_parallel_row_deps_prepared,
            warmup=warmup,
            rep=rep,
        )
    elif provider == "sync-free-e2e":
        def run_sync_free_row_deps():
            ilu0_csr_row_dependencies_sync_free(
                Ap,
                Ai,
                row_done.copy(),
                n,
            )

        ms, min_ms, max_ms = do_wall_bench(
            run_sync_free_row_deps,
            warmup=warmup,
            rep=rep,
        )
    elif provider == "sync-free-prepared":
        prepared = ilu0_csr_row_dependencies_sync_free.prepare_cuda_arguments(
            Ap,
            Ai,
            row_done.copy(),
            n,
        )
        initial_dep_count = prepared.host_values["dep_count"].copy()
        device_dep_count = prepared.values["dep_count"]

        def run_sync_free_row_deps_prepared():
            # Sync-free execution consumes the dependency counts. Reset them
            # for every measured factorization so later repetitions remain
            # dependency-correct.
            device_dep_count.copy_to_device(initial_dep_count)
            ilu0_csr_row_dependencies_sync_free.cuda_launcher(
                *prepared.launcher_arguments
            )
            cuda.synchronize()

        ms, min_ms, max_ms = do_wall_bench(
            run_sync_free_row_deps_prepared,
            warmup=warmup,
            rep=rep,
        )
    elif provider == "cpu-row-deps":
        def run_cpu_row_deps():
            cpu_ilu0_csr_row_dependencies(Ap, Ai, row_done)

        ms, min_ms, max_ms = do_wall_bench(
            run_cpu_row_deps,
            warmup=warmup,
            rep=rep,
        )
    elif provider == "cpu-row-deps-prepared":
        prepared_cpu_out = np.empty_like(row_done)

        def run_cpu_row_deps_prepared():
            cpu_ilu0_csr_row_dependencies_into(Ap, Ai, prepared_cpu_out)

        ms, min_ms, max_ms = do_wall_bench(
            run_cpu_row_deps_prepared,
            warmup=warmup,
            rep=rep,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    rows_per_second = lambda t_ms: n / (t_ms * 1e-3)
    print(f"{provider} -- {matrices} -- time: {ms:.3f} ms")
    return rows_per_second(ms), rows_per_second(max_ms), rows_per_second(min_ms)


def do_wall_bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    times = np.array(times)
    return (
        float(np.quantile(times, 0.5)),
        float(np.quantile(times, 0.2)),
        float(np.quantile(times, 0.8)),
    )


if __name__ == "__main__":
    if not mtx_list:
        raise SystemExit(f"No downloaded .mtx files found in {SUITESPARSE_MATRIX_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark.run(print_data=False, save_path=str(RESULTS_DIR), show_plots=False)
