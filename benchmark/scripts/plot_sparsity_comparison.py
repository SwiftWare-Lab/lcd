#!/usr/bin/env python3
"""Generate sparsity-sorted performance comparison line charts for Numba SpTRSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy import sparse


def compute_level_count(matrix):
    lower = sparse.tril(matrix, k=0).tocsr()
    n = lower.shape[0]
    if n == 0:
        return 0

    indptr = lower.indptr
    indices = lower.indices
    levels = np.zeros(n, dtype=np.int32)

    for row in range(n):
        start = indptr[row]
        end = indptr[row + 1]
        max_dep = 0
        for idx in range(start, end):
            col = indices[idx]
            if col < row:
                max_dep = max(max_dep, levels[col])
        levels[row] = max_dep + 1

    return int(levels.max())


def load_matrix_stats(matrix_path):
    """Return sparsity-related statistics for the given Matrix Market file."""
    matrix = scio.mmread(matrix_path)
    # Ensure we operate on COO to access row/col indices cheaply
    coo = matrix.tocoo()
    rows, cols = coo.shape
    nnz = coo.nnz

    total_entries = rows * cols
    density = nnz / total_entries if total_entries else 0.0
    sparsity_ratio = 1.0 - density

    avg_nnz_per_row = nnz / rows if rows else 0.0
    avg_nnz_per_col = nnz / cols if cols else 0.0

    if nnz > 0:
        bandwidth = np.abs(coo.row - coo.col)
        avg_bandwidth = float(np.mean(bandwidth))
        max_bandwidth = float(np.max(bandwidth))
    else:
        avg_bandwidth = 0.0
        max_bandwidth = 0.0

    return {
        "rows": rows,
        "cols": cols,
        "nnz": nnz,
        "sparsity_ratio": sparsity_ratio,
        "avg_nnz_per_row": avg_nnz_per_row,
        "avg_nnz_per_col": avg_nnz_per_col,
        "avg_bandwidth": avg_bandwidth,
        "max_bandwidth": max_bandwidth,
        "level_count": compute_level_count(matrix),
    }


def compute_stats(df):
    stats = [load_matrix_stats(path) for path in df["matrices"]]
    for key in stats[0].keys():
        df[key] = [stat[key] for stat in stats]
    df["sparsity_percent"] = df["sparsity_ratio"] * 100.0
    return df


def _format_speedup(values_a, values_b):
    ratio = values_b / np.maximum(values_a, 1e-12)
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size == 0:
        return "n/a"
    median_speedup = np.median(finite_ratio)
    return f"median speedup {median_speedup:.2f}×"


def plot_line_comparison(df, method_a, method_b, *, sort_key, sort_label, output_file, order="increasingly"):
    df_sorted = df.sort_values(sort_key).reset_index(drop=True)
    matrices = [Path(path).stem for path in df_sorted["matrices"]]
    x_pos = np.arange(len(matrices))

    plt.figure(figsize=(16, 6))
    plt.plot(
        x_pos,
        df_sorted[method_a].values,
        marker="o",
        markersize=3,
        linewidth=1.4,
        label=method_a,
    )
    plt.plot(
        x_pos,
        df_sorted[method_b].values,
        marker="s",
        markersize=3,
        linewidth=1.4,
        label=method_b,
    )

    plt.xticks(x_pos, matrices, rotation=90, fontsize=7)
    plt.ylabel("Performance (GFLOP/s)")
    plt.xlabel(f"Matrix sorted {order} by {sort_label}")
    plt.title(f"{method_a} vs {method_b} — sorted {order} by {sort_label}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    summary_text = _format_speedup(df_sorted[method_a].values, df_sorted[method_b].values)
    plt.text(
        0.01,
        0.02,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Numba SpTRSV performance vs sparsity")
    parser.add_argument("--csv", default="sptrsv-performance.csv", help="Input CSV file")
    parser.add_argument(
        "--output-prefix",
        default="sptrsv-numba-sparsity",
        help="Prefix for generated plot files",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = compute_stats(df)

    plot_line_comparison(
        df,
        "Numba-CSC-Wavefront",
        "Numba-CSC-Sync-free",
        sort_key="sparsity_percent",
        sort_label="sparsity (%)",
        output_file=f"{args.output_prefix}-csc-sorted-sparsity.png",
    )

    plot_line_comparison(
        df,
        "Numba-CSC-Wavefront",
        "Numba-CSC-Sync-free",
        sort_key="level_count",
        sort_label="level count",
        output_file=f"{args.output_prefix}-csc-sorted-levels.png",
    )

    plot_line_comparison(
        df,
        "Numba-CSR-Wavefront",
        "Numba-CSR-Sync-free",
        sort_key="sparsity_percent",
        sort_label="sparsity (%)",
        output_file=f"{args.output_prefix}-csr-sorted-sparsity.png",
    )

    plot_line_comparison(
        df,
        "Numba-CSR-Wavefront",
        "Numba-CSR-Sync-free",
        sort_key="level_count",
        sort_label="level count",
        output_file=f"{args.output_prefix}-csr-sorted-levels.png",
    )


if __name__ == "__main__":
    main()

