#!/usr/bin/env python3
"""Plot ILU0 benchmark results stored in CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio


DISPLAY_NAMES = {
    "[REF] cuSPARSE": "cuSPARSE (ref)",
    "Numba Wavefront": "Wavefront",
    "Numba Wavefront+Mapping": "Wavefront + Mapping",
    "Numba Wavefront+Unrolling": "Wavefront + Unrolling",
    "Numba Sync-free": "Sync-free",
    "Numba Sync-free+Mapping": "Sync-free + Mapping",
    "Numba Sync-free+Unrolling": "Sync-free + Unrolling",
}

WAVEFRONT_VARIANTS = [
    "Numba Wavefront",
    "Numba Wavefront+Mapping",
    "Numba Wavefront+Unrolling",
]

SYNCFREE_VARIANTS = [
    "Numba Sync-free",
    "Numba Sync-free+Mapping",
    "Numba Sync-free+Unrolling",
]


def get_matrix_nnz(matrix_path: str) -> int:
    """Return nnz for the Matrix Market file at `matrix_path`."""
    try:
        return int(scio.mmread(matrix_path).nnz)
    except Exception:
        return 0


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["nnz"] = [get_matrix_nnz(path) for path in df["matrices"]]
    df["matrix_name"] = [Path(path).stem for path in df["matrices"]]

    impl_cols = [col for col in df.columns if col not in ("matrices", "matrix_name", "nnz")]
    df = df.dropna(subset=impl_cols, how="all")
    return df


def _maybe_add_legend(ax=None, **kwargs):
    target_ax = ax or plt.gca()
    handles, labels = target_ax.get_legend_handles_labels()
    if labels:
        target_ax.legend(**kwargs)


def plot_line_chart(df: pd.DataFrame, output_file: str) -> None:
    df_sorted = df.sort_values("nnz")
    matrices = df_sorted["matrix_name"].tolist()
    x_pos = np.arange(len(matrices))

    plt.figure(figsize=(18, 7))
    for col, disp in DISPLAY_NAMES.items():
        if col not in df_sorted:
            continue
        values = df_sorted[col].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if not mask.any():
            continue
        plt.plot(
            x_pos[mask],
            values[mask],
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=disp,
        )

    ax = plt.gca()
    ax.set_xlabel("Matrix (sorted by nnz)", fontsize=16, fontweight='bold')
    ax.set_ylabel("GFLOP/s", fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(matrices, rotation=90, fontsize=7)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    _maybe_add_legend(ax=ax, loc="best", fontsize=14)
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"Line chart saved as {output_file} and {pdf_file}")


def plot_bar_chart(df: pd.DataFrame, output_file: str, num_matrices: int) -> None:
    df_sorted = df.sort_values("nnz").head(num_matrices)
    matrices = df_sorted["matrix_name"].tolist()
    x_pos = np.arange(len(matrices))

    method_cols = [col for col in DISPLAY_NAMES if col in df_sorted]
    width = 0.8 / max(len(method_cols), 1)

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, col in enumerate(method_cols):
        values = df_sorted[col].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if not mask.any():
            continue
        offset = width * (idx - len(method_cols) / 2 + 0.5)
        ax.bar(
            x_pos[mask] + offset,
            values[mask],
            width,
            label=DISPLAY_NAMES[col],
            alpha=0.8,
        )

    ax.set_xlabel("Matrix (sorted by nnz)", fontsize=16, fontweight='bold')
    ax.set_ylabel("GFLOP/s", fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(matrices, rotation=90, fontsize=7)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    _maybe_add_legend(ax, loc="best", fontsize=14)
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved as {output_file} and {pdf_file}")


def plot_best_of_each(df: pd.DataFrame, output_file: str) -> None:
    df_sorted = df.sort_values("nnz")
    matrices = df_sorted["matrix_name"].tolist()
    x_pos = np.arange(len(matrices))

    wave_cols = [col for col in WAVEFRONT_VARIANTS if col in df_sorted]
    sync_cols = [col for col in SYNCFREE_VARIANTS if col in df_sorted]

    plt.figure(figsize=(18, 7))

    if wave_cols:
        wave_best = df_sorted[wave_cols].max(axis=1).to_numpy(dtype=float)
        mask = ~np.isnan(wave_best)
        if mask.any():
            plt.plot(
                x_pos[mask],
                wave_best[mask],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label="Wavefront (best variant)",
            )

    if sync_cols:
        sync_best = df_sorted[sync_cols].max(axis=1).to_numpy(dtype=float)
        mask = ~np.isnan(sync_best)
        if mask.any():
            plt.plot(
                x_pos[mask],
                sync_best[mask],
                marker="s",
                linewidth=1.4,
                markersize=3,
                label="Sync-free (best variant)",
            )

    ax = plt.gca()
    ax.set_xlabel("Matrix (sorted by nnz)", fontsize=16, fontweight='bold')
    ax.set_ylabel("GFLOP/s", fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(matrices, rotation=90, fontsize=7)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    _maybe_add_legend(ax=ax, loc="best", fontsize=14)
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"Best-of comparison chart saved as {output_file} and {pdf_file}")


def plot_median_summary(df: pd.DataFrame, output_file: str) -> None:
    medians = {}
    for col, disp in DISPLAY_NAMES.items():
        if col not in df:
            continue
        values = df[col].dropna().to_numpy(dtype=float)
        if values.size:
            medians[disp] = np.median(values)

    wave_cols = [col for col in WAVEFRONT_VARIANTS if col in df]
    if wave_cols:
        wave_best = df[wave_cols].max(axis=1).dropna().to_numpy(dtype=float)
        if wave_best.size:
            medians["Wavefront (best variant)"] = np.median(wave_best)

    sync_cols = [col for col in SYNCFREE_VARIANTS if col in df]
    if sync_cols:
        sync_best = df[sync_cols].max(axis=1).dropna().to_numpy(dtype=float)
        if sync_best.size:
            medians["Sync-free (best variant)"] = np.median(sync_best)

    labels = list(medians.keys())
    values = [medians[label] for label in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(len(labels)), values, color="#3b82f6", alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Median GFLOP/s", fontsize=16, fontweight='bold')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"Median summary saved as {output_file} and {pdf_file}")


def print_statistics(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("ILU0 PERFORMANCE STATISTICS")
    print("=" * 80)

    for col, disp in DISPLAY_NAMES.items():
        if col not in df:
            continue
        values = df[col].dropna().to_numpy(dtype=float)
        if not values.size:
            continue
        print(f"\n{disp}:")
        print(f"  Mean:   {np.mean(values):.4f} GFLOP/s")
        print(f"  Median: {np.median(values):.4f} GFLOP/s")
        print(f"  Min:    {np.min(values):.4f} GFLOP/s")
        print(f"  Max:    {np.max(values):.4f} GFLOP/s")
        print(f"  Std:    {np.std(values):.4f} GFLOP/s")
        print(f"  Count:  {values.size} matrices")

    wave_cols = [col for col in WAVEFRONT_VARIANTS if col in df]
    if wave_cols:
        wave_best = df[wave_cols].max(axis=1).dropna().to_numpy(dtype=float)
        if wave_best.size:
            print("\nWavefront (best variant per matrix):")
            print(f"  Mean:   {np.mean(wave_best):.4f} GFLOP/s")
            print(f"  Median: {np.median(wave_best):.4f} GFLOP/s")
            print(f"  Min:    {np.min(wave_best):.4f} GFLOP/s")
            print(f"  Max:    {np.max(wave_best):.4f} GFLOP/s")
            print(f"  Std:    {np.std(wave_best):.4f} GFLOP/s")
            print(f"  Count:  {wave_best.size} matrices")

    sync_cols = [col for col in SYNCFREE_VARIANTS if col in df]
    if sync_cols:
        sync_best = df[sync_cols].max(axis=1).dropna().to_numpy(dtype=float)
        if sync_best.size:
            print("\nSync-free (best variant per matrix):")
            print(f"  Mean:   {np.mean(sync_best):.4f} GFLOP/s")
            print(f"  Median: {np.median(sync_best):.4f} GFLOP/s")
            print(f"  Min:    {np.min(sync_best):.4f} GFLOP/s")
            print(f"  Max:    {np.max(sync_best):.4f} GFLOP/s")
            print(f"  Std:    {np.std(sync_best):.4f} GFLOP/s")
            print(f"  Count:  {sync_best.size} matrices")


def main():
    parser = argparse.ArgumentParser(description="Plot ILU0 performance from CSV")
    parser.add_argument("--csv", default="ilu0-performance.csv", help="Input CSV file")
    parser.add_argument(
        "--type",
        choices=["line", "bar", "best", "median", "all"],
        default="all",
        help="Type of plot to generate",
    )
    parser.add_argument("--bar-count", type=int, default=25, help="Number of matrices to show in bar chart")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    df = prepare_dataframe(args.csv)
    print(f"Loaded {len(df)} matrices")

    if args.type in ("line", "all"):
        plot_line_chart(df, "ilu0-performance-line.png")

    if args.type in ("bar", "all"):
        plot_bar_chart(df, "ilu0-performance-bar.png", num_matrices=args.bar_count)

    if args.type in ("best", "all"):
        plot_best_of_each(df, "ilu0-performance-best-of-each.png")

    if args.type in ("median", "all"):
        plot_median_summary(df, "ilu0-performance-median-comparison.png")

    if args.stats or args.type == "all":
        print_statistics(df)


if __name__ == "__main__":
    main()

