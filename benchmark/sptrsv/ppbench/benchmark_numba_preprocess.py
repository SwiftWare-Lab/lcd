import argparse
import csv
import os
import time

import numpy as np
import scipy.io as scio
from scipy.sparse import tril

from host_inspector_numba import (
    compute_wavefront_numba,
    incoming_edge_numba,
    sptrsv_h_level_analyze_numba_host,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Numba host-side preprocessing."
    )
    parser.add_argument(
        "--matrix-list",
        required=True,
        help="Text file with matrix names or paths.",
    )
    parser.add_argument(
        "--matrix-root",
        default="",
        help="Directory containing .mtx files when list entries are names.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into the matrix list.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="End index into the matrix list, exclusive.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed repetitions per matrix.",
    )
    parser.add_argument(
        "--band-width",
        type=int,
        default=64,
        help="Band width for h-level analysis.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="CSV output path.",
    )
    return parser.parse_args()


def read_matrix_entries(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_matrix_path(entry, matrix_root):
    if os.path.exists(entry):
        return os.path.abspath(entry)
    if entry.endswith(".mtx"):
        if not matrix_root:
            raise FileNotFoundError(f"matrix_root required for {entry}")
        return os.path.join(matrix_root, entry)
    if not matrix_root:
        raise FileNotFoundError(f"matrix_root required for {entry}")
    return os.path.join(matrix_root, f"{entry}.mtx")


def matrix_label(entry):
    base = os.path.basename(entry)
    stem, _ = os.path.splitext(base)
    return stem


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


def warmup_numba(A, band_width):
    compute_wavefront_numba(A)
    incoming_edge_numba(A)
    sptrsv_h_level_analyze_numba_host(A, band_width=band_width)


def benchmark_one_matrix(A, repeats, band_width):
    wave_stats = time_call(lambda: compute_wavefront_numba(A), repeats)
    incoming_stats = time_call(lambda: incoming_edge_numba(A), repeats)
    hlevel_stats = time_call(
        lambda: sptrsv_h_level_analyze_numba_host(A, band_width=band_width),
        repeats,
    )

    wave_ptr, waves = wave_stats["last_result"]
    h_nodes, task_ptr, h_level_ptr = hlevel_stats["last_result"]

    return {
        "num_wavefronts": int(wave_ptr.size - 1),
        "num_wave_nodes": int(waves.size),
        "num_h_nodes": int(h_nodes.size),
        "num_h_tasks": int(task_ptr.size - 1),
        "num_h_levels": int(h_level_ptr.size - 1),
        "numba_wavefront_mean_ms": wave_stats["mean_ms"],
        "numba_wavefront_median_ms": wave_stats["median_ms"],
        "numba_wavefront_min_ms": wave_stats["min_ms"],
        "numba_wavefront_max_ms": wave_stats["max_ms"],
        "numba_incoming_mean_ms": incoming_stats["mean_ms"],
        "numba_incoming_median_ms": incoming_stats["median_ms"],
        "numba_incoming_min_ms": incoming_stats["min_ms"],
        "numba_incoming_max_ms": incoming_stats["max_ms"],
        "numba_hlevel_mean_ms": hlevel_stats["mean_ms"],
        "numba_hlevel_median_ms": hlevel_stats["median_ms"],
        "numba_hlevel_min_ms": hlevel_stats["min_ms"],
        "numba_hlevel_max_ms": hlevel_stats["max_ms"],
        "numba_total_mean_ms": (
            wave_stats["mean_ms"] + incoming_stats["mean_ms"] + hlevel_stats["mean_ms"]
        ),
        "numba_total_median_ms": (
            wave_stats["median_ms"] + incoming_stats["median_ms"] + hlevel_stats["median_ms"]
        ),
    }


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def selected_entries(entries, start_index, end_index):
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index < 0 or end_index > len(entries):
        end_index = len(entries)
    if start_index > end_index:
        raise ValueError("start_index must be <= end_index")
    return entries[start_index:end_index], end_index


def main():
    args = parse_args()
    entries = read_matrix_entries(args.matrix_list)
    subset, normalized_end = selected_entries(entries, args.start_index, args.end_index)

    rows = []
    warmed_up = False

    for local_idx, entry in enumerate(subset):
        global_idx = args.start_index + local_idx
        label = matrix_label(entry)
        matrix_path = resolve_matrix_path(entry, args.matrix_root)
        print(f"[{global_idx + 1}/{len(entries)}] loading {label}")

        if not os.path.exists(matrix_path):
            print(f"  missing: {matrix_path}")
            rows.append(
                {
                    "matrix_index": global_idx,
                    "matrix_entry": entry,
                    "matrix_name": label,
                    "matrix_path": matrix_path,
                    "status": "missing",
                }
            )
            continue

        try:
            A = load_benchmark_matrix(matrix_path)
            if not warmed_up:
                warmup_numba(A, args.band_width)
                warmed_up = True

            result = benchmark_one_matrix(A, args.repeats, args.band_width)
            result.update(
                {
                    "matrix_index": global_idx,
                    "matrix_entry": entry,
                    "matrix_name": label,
                    "matrix_path": matrix_path,
                    "n": int(A.shape[0]),
                    "nnz": int(A.nnz),
                    "band_width": int(args.band_width),
                    "repeats": int(args.repeats),
                    "status": "ok",
                }
            )
            rows.append(result)
            print(
                "  "
                f"wavefront={result['numba_wavefront_mean_ms']:.3f} ms, "
                f"incoming={result['numba_incoming_mean_ms']:.3f} ms, "
                f"hlevel={result['numba_hlevel_mean_ms']:.3f} ms, "
                f"total={result['numba_total_mean_ms']:.3f} ms"
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            rows.append(
                {
                    "matrix_index": global_idx,
                    "matrix_entry": entry,
                    "matrix_name": label,
                    "matrix_path": matrix_path,
                    "status": f"error: {exc}",
                }
            )

    ensure_parent_dir(args.output)
    fieldnames = [
        "matrix_index",
        "matrix_entry",
        "matrix_name",
        "matrix_path",
        "status",
        "n",
        "nnz",
        "band_width",
        "repeats",
        "num_wavefronts",
        "num_wave_nodes",
        "num_h_nodes",
        "num_h_tasks",
        "num_h_levels",
        "numba_wavefront_mean_ms",
        "numba_wavefront_median_ms",
        "numba_wavefront_min_ms",
        "numba_wavefront_max_ms",
        "numba_incoming_mean_ms",
        "numba_incoming_median_ms",
        "numba_incoming_min_ms",
        "numba_incoming_max_ms",
        "numba_hlevel_mean_ms",
        "numba_hlevel_median_ms",
        "numba_hlevel_min_ms",
        "numba_hlevel_max_ms",
        "numba_total_mean_ms",
        "numba_total_median_ms",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"wrote {args.output} for slice [{args.start_index}, {normalized_end}) "
        f"with {len(rows)} rows"
    )


if __name__ == "__main__":
    main()
