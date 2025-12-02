#!/usr/bin/env python3
"""
SpTRSV Performance Analysis Script

This script parses the sptrsv-performance.csv file and identifies the best
performing wavefront and sync-free variants for each matrix, comparing them
against cuSPARSE as the reference implementation.

Analysis is done separately for CSC and CSR formats.
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def get_matrix_nnz(matrix_path: str) -> int:
    """Get the number of nonzeros from a matrix file."""
    try:
        matrix = scio.mmread(matrix_path)
        return matrix.nnz
    except:
        return 0


def parse_csv(csv_file: str) -> Tuple[List[str], List[str], List[List[float]]]:
    """Parse the CSV file and return headers, matrix names, and data."""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        matrices = []
        data = []
        
        for row in reader:
            matrices.append(row[0])
            data.append([float(x) for x in row[1:]])
    
    return headers[1:], matrices, data


def classify_variant(variant_name: str) -> str:
    """Classify variant as wavefront, sync-free, or other."""
    if 'WF' in variant_name or 'Wavefront' in variant_name:
        return 'wavefront'
    elif 'SF' in variant_name or 'Sync-free' in variant_name:
        return 'sync-free'
    else:
        return 'other'


def get_format(variant_name: str) -> str:
    """Get format (CSC or CSR) from variant name."""
    if 'CSC' in variant_name:
        return 'CSC'
    elif 'CSR' in variant_name:
        return 'CSR'
    else:
        return 'unknown'


def get_variant_category(variant_name: str) -> str:
    """Get more detailed variant category."""
    if 'Numba' in variant_name and 'Wavefront' in variant_name:
        return 'Numba-Wavefront (baseline)'
    elif 'Numba' in variant_name and 'WF-Map' in variant_name:
        return 'Numba-Wavefront-Map'
    elif 'WF-U' in variant_name:
        return 'Wavefront-Unrolled'
    elif 'Numba' in variant_name and 'Sync-free' in variant_name:
        return 'Numba-Sync-Free (baseline)'
    elif 'Numba' in variant_name and 'SF-Map' in variant_name:
        return 'Numba-Sync-Free-Map'
    elif 'SF-U' in variant_name:
        return 'Sync-Free-Unrolled'
    else:
        return 'Other'


def geomean(values: List[float]) -> float:
    """Calculate geometric mean of a list of values."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def analyze_format(matrices, data, variant_names, cusparse_idx, format_name, 
                   wavefront_indices, sync_free_indices):
    """Analyze performance for a specific format (CSC or CSR)."""
    
    print()
    print("=" * 80)
    print(f"{format_name} Performance Analysis")
    print("=" * 80)
    print(f"\nFormat: {format_name}")
    print(f"Wavefront variants: {len(wavefront_indices)}")
    print(f"Sync-free variants: {len(sync_free_indices)}")
    print()
    
    # Store results for summary statistics
    wf_speedups = []
    sf_speedups = []
    wf_better_count = 0
    sf_better_count = 0
    
    # Collect detailed per-matrix results
    print("-" * 80)
    print("Per-Matrix Best Configurations")
    print("-" * 80)
    
    # Also save to file
    report_file = f'sptrsv-best-configs-{format_name.lower()}.txt'
    f_report = open(report_file, 'w')
    f_report.write(f"{'=' * 80}\n")
    f_report.write(f"{format_name} Performance Analysis - Best Configurations\n")
    f_report.write(f"{'=' * 80}\n\n")
    
    for matrix_name, perf_data in zip(matrices, data):
        matrix_basename = os.path.basename(matrix_name)
        cusparse_perf = perf_data[cusparse_idx]
        
        # Find best wavefront variant
        best_wf_idx = None
        best_wf_perf = 0.0
        for idx in wavefront_indices:
            if perf_data[idx] > best_wf_perf:
                best_wf_perf = perf_data[idx]
                best_wf_idx = idx
        
        # Find best sync-free variant
        best_sf_idx = None
        best_sf_perf = 0.0
        for idx in sync_free_indices:
            if perf_data[idx] > best_sf_perf:
                best_sf_perf = perf_data[idx]
                best_sf_idx = idx
        
        # Calculate speedups
        wf_speedup = best_wf_perf / cusparse_perf if cusparse_perf > 0 else 0
        sf_speedup = best_sf_perf / cusparse_perf if cusparse_perf > 0 else 0
        
        wf_speedups.append(wf_speedup)
        sf_speedups.append(sf_speedup)
        
        if wf_speedup > sf_speedup:
            wf_better_count += 1
        else:
            sf_better_count += 1
        
        # Print and save detailed results for this matrix
        output = f"\nMatrix: {matrix_basename}\n"
        output += f"  cuSPARSE (Reference): {cusparse_perf:.6f} GFLOP/s\n\n"
        
        if best_wf_idx is not None:
            output += f"  Best Wavefront:\n"
            output += f"    Variant: {variant_names[best_wf_idx]}\n"
            output += f"    Performance: {best_wf_perf:.6f} GFLOP/s\n"
            output += f"    Speedup vs cuSPARSE: {wf_speedup:.3f}x\n"
        
        output += "\n"
        
        if best_sf_idx is not None:
            output += f"  Best Sync-Free:\n"
            output += f"    Variant: {variant_names[best_sf_idx]}\n"
            output += f"    Performance: {best_sf_perf:.6f} GFLOP/s\n"
            output += f"    Speedup vs cuSPARSE: {sf_speedup:.3f}x\n"
        
        output += "\n"
        winner = "Wavefront" if wf_speedup > sf_speedup else "Sync-Free"
        output += f"  Winner: {winner}\n"
        
        print(output, end='')
        f_report.write(output)
    
    # Compute absolute performance values
    cusparse_perfs = [perf_data[cusparse_idx] for perf_data in data]
    best_wf_perfs = []
    best_sf_perfs = []
    
    for perf_data in data:
        best_wf = max([perf_data[idx] for idx in wavefront_indices]) if wavefront_indices else 0
        best_sf = max([perf_data[idx] for idx in sync_free_indices]) if sync_free_indices else 0
        best_wf_perfs.append(best_wf)
        best_sf_perfs.append(best_sf)
    
    # Summary statistics
    summary = "\n" + "=" * 80 + "\n"
    summary += "Summary Statistics\n"
    summary += "=" * 80 + "\n\n"
    
    # Median GFLOP/s comparison
    summary += "Median GFLOP/s Across All Matrices:\n"
    summary += f"  cuSPARSE (reference):   {np.median(cusparse_perfs):.3f} GFLOP/s\n"
    summary += f"  Best Wavefront:         {np.median(best_wf_perfs):.3f} GFLOP/s\n"
    summary += f"  Best Sync-Free:         {np.median(best_sf_perfs):.3f} GFLOP/s\n\n"
    
    # Wavefront summary
    summary += "Wavefront Variants vs cuSPARSE:\n"
    summary += f"  Mean speedup: {sum(wf_speedups) / len(wf_speedups):.3f}x\n"
    summary += f"  Min speedup: {min(wf_speedups):.3f}x\n"
    summary += f"  Max speedup: {max(wf_speedups):.3f}x\n"
    summary += f"  Geometric mean: {geomean(wf_speedups):.3f}x\n"
    faster_count = sum(1 for s in wf_speedups if s > 1.0)
    summary += f"  Faster than cuSPARSE: {faster_count}/{len(wf_speedups)} matrices\n\n"
    
    # Sync-free summary
    summary += "Sync-Free Variants vs cuSPARSE:\n"
    summary += f"  Mean speedup: {sum(sf_speedups) / len(sf_speedups):.3f}x\n"
    summary += f"  Min speedup: {min(sf_speedups):.3f}x\n"
    summary += f"  Max speedup: {max(sf_speedups):.3f}x\n"
    summary += f"  Geometric mean: {geomean(sf_speedups):.3f}x\n"
    faster_count = sum(1 for s in sf_speedups if s > 1.0)
    summary += f"  Faster than cuSPARSE: {faster_count}/{len(sf_speedups)} matrices\n\n"
    
    # Head-to-head comparison
    summary += "Wavefront vs Sync-Free Head-to-Head:\n"
    summary += f"  Wavefront better: {wf_better_count}/{len(matrices)} matrices\n"
    summary += f"  Sync-Free better: {sf_better_count}/{len(matrices)} matrices\n\n"
    
    print(summary, end='')
    f_report.write(summary)
    
    # Find most frequently best-performing variants
    variant_best_counts = defaultdict(int)
    
    for perf_data in data:
        # Find best wavefront
        best_wf_idx = None
        best_wf_perf = 0.0
        for idx in wavefront_indices:
            if perf_data[idx] > best_wf_perf:
                best_wf_perf = perf_data[idx]
                best_wf_idx = idx
        
        if best_wf_idx is not None:
            variant_best_counts[variant_names[best_wf_idx]] += 1
        
        # Find best sync-free
        best_sf_idx = None
        best_sf_perf = 0.0
        for idx in sync_free_indices:
            if perf_data[idx] > best_sf_perf:
                best_sf_perf = perf_data[idx]
                best_sf_idx = idx
        
        if best_sf_idx is not None:
            variant_best_counts[variant_names[best_sf_idx]] += 1
    
    optimal_config = "\n" + "-" * 80 + "\n"
    optimal_config += "Most Frequently Best-Performing Variants (Optimal Configs)\n"
    optimal_config += "-" * 80 + "\n\n"
    
    sorted_variants = sorted(variant_best_counts.items(), key=lambda x: x[1], reverse=True)
    for variant, count in sorted_variants[:15]:  # Show top 15
        optimal_config += f"  {variant}: {count} times ({count/len(matrices)*100:.1f}%)\n"
    
    optimal_config += "\n"
    
    print(optimal_config, end='')
    f_report.write(optimal_config)
    
    f_report.close()
    print(f"Detailed report saved to {report_file}\n")
    
    # Generate plots
    print("-" * 80)
    print("Generating Plots")
    print("-" * 80)
    
    format_suffix = format_name.lower()
    plot_per_matrix_performance(matrices, data, wavefront_indices, sync_free_indices,
                                 variant_names, cusparse_idx, format_name,
                                 f'sptrsv-per-matrix-{format_suffix}.png')
    
    plot_aggregate_statistics(matrices, data, wavefront_indices, sync_free_indices,
                               variant_names, cusparse_idx, format_name,
                               f'sptrsv-aggregate-{format_suffix}.png')
    
    return {
        'wf_speedups': wf_speedups,
        'sf_speedups': sf_speedups,
        'best_wf_perfs': best_wf_perfs,
        'best_sf_perfs': best_sf_perfs,
        'cusparse_perfs': cusparse_perfs
    }


def plot_per_matrix_performance(matrices, data, wavefront_indices, sync_free_indices, 
                                  variant_names, cusparse_idx, format_name, output_file):
    """Plot best wavefront vs best sync-free performance for each matrix, sorted by nnz."""
    
    # Get nnz for each matrix and prepare data
    matrix_data = []
    for matrix_path, perf_data in zip(matrices, data):
        nnz = get_matrix_nnz(matrix_path)
        matrix_name = os.path.basename(matrix_path).replace('.mtx', '')
        cusparse_perf = perf_data[cusparse_idx]
        
        # Find best wavefront
        best_wf_perf = max([perf_data[idx] for idx in wavefront_indices]) if wavefront_indices else 0
        
        # Find best sync-free
        best_sf_perf = max([perf_data[idx] for idx in sync_free_indices]) if sync_free_indices else 0
        
        matrix_data.append({
            'name': matrix_name,
            'nnz': nnz,
            'cusparse': cusparse_perf,
            'wavefront': best_wf_perf,
            'sync_free': best_sf_perf
        })
    
    # Sort by nnz
    matrix_data.sort(key=lambda x: x['nnz'])
    
    # Prepare data for plotting
    x_pos = np.arange(len(matrix_data))
    matrix_names = [d['name'] for d in matrix_data]
    cusparse_perfs = [d['cusparse'] for d in matrix_data]
    wf_perfs = [d['wavefront'] for d in matrix_data]
    sf_perfs = [d['sync_free'] for d in matrix_data]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(x_pos, cusparse_perfs, 'o-', label='cuSPARSE (reference)', 
            linewidth=2, markersize=4, color='gold', alpha=0.8)
    ax.plot(x_pos, wf_perfs, 's-', label=f'Best Wavefront ({format_name})', 
            linewidth=2, markersize=4, color='#e74c3c', alpha=0.8)
    ax.plot(x_pos, sf_perfs, '^-', label=f'Best Sync-Free ({format_name})', 
            linewidth=2, markersize=4, color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Matrix (sorted by nnz)', fontsize=16, fontweight='bold')
    ax.set_ylabel('GFLOP/s', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(matrix_names, rotation=90, fontsize=7)
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"  Per-matrix plot saved as {output_file} and {pdf_file}")


def plot_aggregate_statistics(matrices, data, wavefront_indices, sync_free_indices, 
                                variant_names, cusparse_idx, format_name, output_file):
    """Plot aggregate statistics comparing variant categories."""
    
    # Collect speedups by category for this format
    category_speedups = defaultdict(list)
    
    for perf_data in data:
        cusparse_perf = perf_data[cusparse_idx]
        
        for idx in wavefront_indices + sync_free_indices:
            variant_name = variant_names[idx]
            category = get_variant_category(variant_name)
            speedup = perf_data[idx] / cusparse_perf if cusparse_perf > 0 else 0
            category_speedups[category].append(speedup)
    
    # Compute best wavefront and best sync-free per matrix
    best_wf_speedups = []
    best_sf_speedups = []
    
    for perf_data in data:
        cusparse_perf = perf_data[cusparse_idx]
        
        best_wf = max([perf_data[idx] for idx in wavefront_indices]) if wavefront_indices else 0
        best_sf = max([perf_data[idx] for idx in sync_free_indices]) if sync_free_indices else 0
        
        best_wf_speedups.append(best_wf / cusparse_perf if cusparse_perf > 0 else 0)
        best_sf_speedups.append(best_sf / cusparse_perf if cusparse_perf > 0 else 0)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Median speedup by category
    categories_order = [
        'Numba-Wavefront (baseline)', 'Numba-Wavefront-Map', 'Wavefront-Unrolled',
        'Numba-Sync-Free (baseline)', 'Numba-Sync-Free-Map', 'Sync-Free-Unrolled'
    ]
    
    categories_display = []
    medians = []
    colors = []
    
    for cat in categories_order:
        if cat in category_speedups:
            display_name = cat.replace(' (baseline)', '\n(baseline)')
            categories_display.append(display_name)
            speedups = category_speedups[cat]
            medians.append(np.median(speedups))
            colors.append('#e74c3c' if 'Wavefront' in cat else '#3498db')
    
    if categories_display:
        bars = ax1.bar(np.arange(len(categories_display)), medians, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax1.axhline(y=1.0, color='gold', linestyle='--', linewidth=2, 
                   label='cuSPARSE baseline (1.0x)')
        
        for bar, value in zip(bars, medians):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f'{value:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_ylabel('Median Speedup vs cuSPARSE', fontsize=16, fontweight='bold')
        ax1.set_xticks(np.arange(len(categories_display)))
        ax1.set_xticklabels(categories_display, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=14)
        
        # Remove grid
        ax1.grid(False)
        
        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # Plot 2: Best variant comparison
    labels = ['Best\nWavefront', 'Best\nSync-Free']
    means = [np.mean(best_wf_speedups), np.mean(best_sf_speedups)]
    medians_best = [np.median(best_wf_speedups), np.median(best_sf_speedups)]
    maxs = [np.max(best_wf_speedups), np.max(best_sf_speedups)]
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax2.bar(x - width, means, width, label='Mean', color='#9b59b6', 
                   alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x, medians_best, width, label='Median', color='#2ecc71', 
                   alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x + width, maxs, width, label='Max', color='#f39c12', 
                   alpha=0.8, edgecolor='black')
    
    ax2.axhline(y=1.0, color='gold', linestyle='--', linewidth=2, label='cuSPARSE baseline')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Speedup vs cuSPARSE', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax2.legend(fontsize=14)
    
    # Remove grid
    ax2.grid(False)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"  Aggregate plot saved as {output_file} and {pdf_file}")


def analyze_performance(csv_file: str):
    """Analyze performance data and generate summary report."""
    
    # Parse CSV
    variant_names, matrices, data = parse_csv(csv_file)
    
    # cuSPARSE is always the first column
    cusparse_idx = 0
    
    # Classify all variants by format
    csc_wavefront_indices = []
    csc_sync_free_indices = []
    csr_wavefront_indices = []
    csr_sync_free_indices = []
    
    for idx, name in enumerate(variant_names[1:], start=1):  # Skip cuSPARSE
        variant_type = classify_variant(name)
        format_type = get_format(name)
        
        if format_type == 'CSC':
            if variant_type == 'wavefront':
                csc_wavefront_indices.append(idx)
            elif variant_type == 'sync-free':
                csc_sync_free_indices.append(idx)
        elif format_type == 'CSR':
            if variant_type == 'wavefront':
                csr_wavefront_indices.append(idx)
            elif variant_type == 'sync-free':
                csr_sync_free_indices.append(idx)
    
    print("=" * 80)
    print("SpTRSV Performance Analysis")
    print("=" * 80)
    print(f"\nTotal matrices: {len(matrices)}")
    print(f"\nCSC variants:")
    print(f"  Wavefront: {len(csc_wavefront_indices)}")
    print(f"  Sync-free: {len(csc_sync_free_indices)}")
    print(f"\nCSR variants:")
    print(f"  Wavefront: {len(csr_wavefront_indices)}")
    print(f"  Sync-free: {len(csr_sync_free_indices)}")
    
    # Analyze CSC format
    csc_results = analyze_format(matrices, data, variant_names, cusparse_idx, 'CSC',
                                  csc_wavefront_indices, csc_sync_free_indices)
    
    # Analyze CSR format
    csr_results = analyze_format(matrices, data, variant_names, cusparse_idx, 'CSR',
                                  csr_wavefront_indices, csr_sync_free_indices)


def main():
    csv_file = '/home/mad29/projects/gpu-backend/benchmark/sptrsv-performance.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    analyze_performance(csv_file)


if __name__ == '__main__':
    main()
