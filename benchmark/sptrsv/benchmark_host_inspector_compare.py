import argparse
import csv
import os
import sys
import time

import numpy as np
import scipy.io as scio
from scipy.sparse import tril

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from benchmark.bench_utils import compute_wavefront, compact_2d_list, incoming_edge
from benchmark.sptrsv.host_inspector_numba import compute_wavefront_numba, incoming_edge_numba


DEFAULT_LIST = "/home/mad29/projects/gpu-backend/local_playground/sc26/orig_list.txt"
DEFAULT_MATRIX_ROOT = "/home/mad29/projects/testdata/matrices"
DEFAULT_OUTPUT = "/home/mad29/projects/gpu-backend/local_playground/sc26/host_inspector_compare_first10.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs Numba host inspector preprocessing."
    )
    parser.add_argument("--matrix-list", default=DEFAULT_LIST, help="Text file with matrix names.")
    parser.add_argument("--matrix-root", default=DEFAULT_MATRIX_ROOT, help="Directory containing .mtx files.")
    parser.add_argument("--limit", type=int, default=10, help="Number of matrices to benchmark.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions per matrix.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path.")
    return parser.parse_args()


def read_matrix_names(list_path, limit):
    with open(list_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names[:limit]


def resolve_matrix_path(name, matrix_root):
    if name.endswith(".mtx") and os.path.exists(name):
        return name
    if name.endswith(".mtx"):
        return os.path.join(matrix_root, name)
    return os.path.join(matrix_root, f"{name}.mtx")


def load_benchmark_matrix(matrix_path):
    matrix = scio.mmread(matrix_path)
    A = matrix.tocsc()
    A = tril(A, format="csc")
    if A.data.dtype != np.float64:
        A.data = A.data.astype(np.float64)
    A.setdiag(A.diagonal())
    return A


def time_call(fn, repeats):
    samples_ms = []
    last_result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_result = fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(samples_ms.mean()),
        "median_ms": float(np.median(samples_ms)),
        "min_ms": float(samples_ms.min()),
        "max_ms": float(samples_ms.max()),
        "last_result": last_result,
    }


def warmup_numba(A):
    compute_wavefront_numba(A)
    incoming_edge_numba(A)


def benchmark_one_matrix(A, repeats):
    py_wave_stats = time_call(lambda: compute_wavefront(A), repeats)
    wavefront_list = py_wave_stats["last_result"]

    py_compact_stats = time_call(lambda: compact_2d_list(wavefront_list), repeats)
    py_wave_ptr, py_waves = py_compact_stats["last_result"]

    py_incoming_stats = time_call(lambda: incoming_edge(A), repeats)
    py_incoming = py_incoming_stats["last_result"]

    nb_wave_stats = time_call(lambda: compute_wavefront_numba(A), repeats)
    nb_wave_ptr, nb_waves = nb_wave_stats["last_result"]

    nb_incoming_stats = time_call(lambda: incoming_edge_numba(A), repeats)
    nb_incoming = nb_incoming_stats["last_result"]

    wavefront_match = np.array_equal(py_wave_ptr, nb_wave_ptr) and np.array_equal(py_waves, nb_waves)
    incoming_match = np.array_equal(py_incoming, nb_incoming)

    return {
        "python_wavefront_mean_ms": py_wave_stats["mean_ms"],
        "python_wavefront_median_ms": py_wave_stats["median_ms"],
        "python_compact_mean_ms": py_compact_stats["mean_ms"],
        "python_compact_median_ms": py_compact_stats["median_ms"],
        "python_incoming_mean_ms": py_incoming_stats["mean_ms"],
        "python_incoming_median_ms": py_incoming_stats["median_ms"],
        "python_total_mean_ms": py_wave_stats["mean_ms"] + py_compact_stats["mean_ms"] + py_incoming_stats["mean_ms"],
        "python_total_median_ms": py_wave_stats["median_ms"] + py_compact_stats["median_ms"] + py_incoming_stats["median_ms"],
        "numba_wavefront_mean_ms": nb_wave_stats["mean_ms"],
        "numba_wavefront_median_ms": nb_wave_stats["median_ms"],
        "numba_compact_mean_ms": 0.0,
        "numba_compact_median_ms": 0.0,
        "numba_incoming_mean_ms": nb_incoming_stats["mean_ms"],
        "numba_incoming_median_ms": nb_incoming_stats["median_ms"],
        "numba_total_mean_ms": nb_wave_stats["mean_ms"] + nb_incoming_stats["mean_ms"],
        "numba_total_median_ms": nb_wave_stats["median_ms"] + nb_incoming_stats["median_ms"],
        "wavefront_match": bool(wavefront_match),
        "incoming_match": bool(incoming_match),
        "num_wavefronts": int(py_wave_ptr.size - 1),
    }


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    matrix_names = read_matrix_names(args.matrix_list, args.limit)
    if not matrix_names:
        raise ValueError("no matrix names found")

    rows = []
    warmed_up = False

    for idx, matrix_name in enumerate(matrix_names, start=1):
        matrix_path = resolve_matrix_path(matrix_name, args.matrix_root)
        print(f"[{idx}/{len(matrix_names)}] loading {matrix_name}")
        if not os.path.exists(matrix_path):
            print(f"  missing: {matrix_path}")
            rows.append(
                {
                    "matrix_name": matrix_name,
                    "matrix_path": matrix_path,
                    "status": "missing",
                }
            )
            continue

        try:
            A = load_benchmark_matrix(matrix_path)
            if not warmed_up:
                warmup_numba(A)
                warmed_up = True

            result = benchmark_one_matrix(A, args.repeats)
            result.update(
                {
                    "matrix_name": matrix_name,
                    "matrix_path": matrix_path,
                    "n": int(A.shape[0]),
                    "nnz": int(A.nnz),
                    "status": "ok",
                }
            )
            result["wavefront_speedup_mean"] = (
                result["python_wavefront_mean_ms"] / result["numba_wavefront_mean_ms"]
                if result["numba_wavefront_mean_ms"] > 0.0
                else np.nan
            )
            result["incoming_speedup_mean"] = (
                result["python_incoming_mean_ms"] / result["numba_incoming_mean_ms"]
                if result["numba_incoming_mean_ms"] > 0.0
                else np.nan
            )
            result["total_speedup_mean"] = (
                result["python_total_mean_ms"] / result["numba_total_mean_ms"]
                if result["numba_total_mean_ms"] > 0.0
                else np.nan
            )
            rows.append(result)
            print(
                "  "
                f"python_total={result['python_total_mean_ms']:.3f} ms, "
                f"numba_total={result['numba_total_mean_ms']:.3f} ms, "
                f"speedup={result['total_speedup_mean']:.2f}x"
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            rows.append(
                {
                    "matrix_name": matrix_name,
                    "matrix_path": matrix_path,
                    "status": f"error: {exc}",
                }
            )

    ensure_parent_dir(args.output)
    fieldnames = [
        "matrix_name",
        "matrix_path",
        "status",
        "n",
        "nnz",
        "num_wavefronts",
        "python_wavefront_mean_ms",
        "python_wavefront_median_ms",
        "python_compact_mean_ms",
        "python_compact_median_ms",
        "python_incoming_mean_ms",
        "python_incoming_median_ms",
        "python_total_mean_ms",
        "python_total_median_ms",
        "numba_wavefront_mean_ms",
        "numba_wavefront_median_ms",
        "numba_compact_mean_ms",
        "numba_compact_median_ms",
        "numba_incoming_mean_ms",
        "numba_incoming_median_ms",
        "numba_total_mean_ms",
        "numba_total_median_ms",
        "wavefront_speedup_mean",
        "incoming_speedup_mean",
        "total_speedup_mean",
        "wavefront_match",
        "incoming_match",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
