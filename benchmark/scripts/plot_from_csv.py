#!/usr/bin/env python3
"""
Advanced plotting script for SpTRSV performance from CSV
Supports multiple plot types and configurations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.io as scio

def get_matrix_nnz(matrix_path):
    """Get the number of nonzeros from a matrix file"""
    try:
        matrix = scio.mmread(matrix_path)
        return matrix.nnz
    except:
        return 0

def plot_line_chart(df, output_file='sptrsv-performance-line.png'):
    """Create a line chart (similar to triton benchmark output)"""
    # Get nonzero counts for each matrix
    nnz_list = []
    for path in df['matrices']:
        nnz = get_matrix_nnz(path)
        nnz_list.append(nnz)
    
    # Add nnz to dataframe and sort by it
    df_sorted = df.copy()
    df_sorted['nnz'] = nnz_list
    df_sorted = df_sorted.sort_values('nnz')
    
    # Extract sorted matrix names
    matrices = [path.split('/')[-1].replace('.mtx', '') for path in df_sorted['matrices']]
    x_pos = np.arange(len(matrices))
    
    methods = {
        'Triton-Wavefront-CSC': ('green', '-'),
        'Triton-Sync-free-CSC': ('blue', '-'),
        'Triton-Wavefront-CSR': ('cyan', '-'),
        'Triton-Sync-free-CSR': ('magenta', '-'),
        '[REF] cuSparse': ('yellow', '-'),
        'Numba-Wavefront-CSC': ('red', '-'),
        'Numba-Sync-free-CSC': ('purple', '-'),
        'Numba-Wavefront-CSR': ('brown', '-'),
        'Numba-Sync-free-CSR': ('pink', '-'),
        '[REF] Triton-SpMV': ('orange', '--'),
    }
    
    plt.figure(figsize=(20, 8))
    
    for method_name, (color, linestyle) in methods.items():
        if method_name in df_sorted.columns:
            values = df_sorted[method_name].values
            plt.plot(x_pos, values, color=color, linestyle=linestyle, 
                    marker='o', markersize=3, label=method_name, linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Matrix (sorted by nnz)', fontsize=12)
    plt.ylabel('GFLOP/S', fontsize=12)
    plt.title('SpTRSV Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, matrices, rotation=90, fontsize=8)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Line chart saved as {output_file}")

def plot_bar_chart(df, output_file='sptrsv-performance-bar.png', num_matrices=20):
    """Create a grouped bar chart for easier comparison (first N matrices)"""
    matrices = [path.split('/')[-1].replace('.mtx', '') for path in df['matrices'][:num_matrices]]
    
    # Select only implementation methods (exclude references)
    impl_methods = [
        'Triton-Wavefront-CSC', 'Triton-Sync-free-CSC',
        'Triton-Wavefront-CSR', 'Triton-Sync-free-CSR',
        'Numba-Wavefront-CSC', 'Numba-Sync-free-CSC',
        'Numba-Wavefront-CSR', 'Numba-Sync-free-CSR'
    ]
    
    x_pos = np.arange(len(matrices))
    width = 0.1
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    colors = ['green', 'blue', 'cyan', 'magenta', 'red', 'purple', 'brown', 'pink']
    
    for i, (method, color) in enumerate(zip(impl_methods, colors)):
        if method in df.columns:
            values = df[method].values[:num_matrices]
            offset = width * (i - len(impl_methods)/2 + 0.5)
            ax.bar(x_pos + offset, values, width, label=method, color=color, alpha=0.8)
    
    ax.set_xlabel('Matrix', fontsize=12)
    ax.set_ylabel('GFLOP/S', fontsize=12)
    ax.set_title(f'SpTRSV Performance Comparison (First {num_matrices} Matrices)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(matrices, rotation=90, fontsize=8)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Bar chart saved as {output_file}")

def plot_comparison_chart(df, output_file='sptrsv-performance-comparison.png'):
    """Create comparison charts: CSC vs CSR, Wavefront vs Sync-free, Triton vs Numba"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    matrices = [path.split('/')[-1].replace('.mtx', '') for path in df['matrices']]
    x_pos = np.arange(len(matrices))
    
    # Plot 1: Triton CSC Wavefront vs CSR Wavefront
    ax = axes[0, 0]
    if 'Triton-Wavefront-CSC' in df.columns and 'Triton-Wavefront-CSR' in df.columns:
        ax.plot(x_pos, df['Triton-Wavefront-CSC'], 'g-', label='CSC', marker='o', markersize=2)
        ax.plot(x_pos, df['Triton-Wavefront-CSR'], 'c-', label='CSR', marker='s', markersize=2)
        ax.set_title('Triton Wavefront: CSC vs CSR', fontweight='bold')
        ax.set_ylabel('GFLOP/S')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Triton CSC Wavefront vs Sync-free
    ax = axes[0, 1]
    if 'Triton-Wavefront-CSC' in df.columns and 'Triton-Sync-free-CSC' in df.columns:
        ax.plot(x_pos, df['Triton-Wavefront-CSC'], 'g-', label='Wavefront', marker='o', markersize=2)
        ax.plot(x_pos, df['Triton-Sync-free-CSC'], 'b-', label='Sync-free', marker='s', markersize=2)
        ax.set_title('Triton CSC: Wavefront vs Sync-free', fontweight='bold')
        ax.set_ylabel('GFLOP/S')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Triton vs Numba (CSC Wavefront)
    ax = axes[1, 0]
    if 'Triton-Wavefront-CSC' in df.columns and 'Numba-Wavefront-CSC' in df.columns:
        ax.plot(x_pos, df['Triton-Wavefront-CSC'], 'g-', label='Triton', marker='o', markersize=2)
        ax.plot(x_pos, df['Numba-Wavefront-CSC'], 'r-', label='Numba', marker='s', markersize=2)
        ax.set_title('CSC Wavefront: Triton vs Numba', fontweight='bold')
        ax.set_xlabel('Matrix')
        ax.set_ylabel('GFLOP/S')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Best method vs cuSparse
    ax = axes[1, 1]
    method_cols = [col for col in df.columns if col not in ['matrices', '[REF] cuSparse', '[REF] Triton-SpMV']]
    if method_cols and '[REF] cuSparse' in df.columns:
        # Find best method for each matrix
        best_perf = df[method_cols].max(axis=1)
        ax.plot(x_pos, best_perf, 'g-', label='Best Method', marker='o', markersize=2)
        ax.plot(x_pos, df['[REF] cuSparse'], 'y-', label='cuSparse', marker='s', markersize=2)
        ax.set_title('Best Method vs cuSparse', fontweight='bold')
        ax.set_xlabel('Matrix')
        ax.set_ylabel('GFLOP/S')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xticks(x_pos[::10])  # Show every 10th label
        ax.set_xticklabels([matrices[i] for i in range(0, len(matrices), 10)], rotation=90, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison chart saved as {output_file}")

def print_statistics(df):
    """Print statistical summary"""
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    
    method_cols = [col for col in df.columns if col != 'matrices']
    
    for method in method_cols:
        if method in df.columns:
            values = df[method].values
            print(f"\n{method}:")
            print(f"  Mean:   {np.mean(values):.3f} GFLOP/S")
            print(f"  Median: {np.median(values):.3f} GFLOP/S")
            print(f"  Min:    {np.min(values):.3f} GFLOP/S")
            print(f"  Max:    {np.max(values):.3f} GFLOP/S")
            print(f"  Std:    {np.std(values):.3f} GFLOP/S")

def main():
    parser = argparse.ArgumentParser(description='Plot SpTRSV performance from CSV')
    parser.add_argument('--csv', default='sptrsv-performance.csv', help='Input CSV file')
    parser.add_argument('--type', choices=['line', 'bar', 'comparison', 'all'], default='all',
                        help='Type of plot to generate')
    parser.add_argument('--stats', action='store_true', help='Print statistics')
    args = parser.parse_args()
    
    # Read CSV
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} matrices")
    
    # Generate plots
    if args.type in ['line', 'all']:
        plot_line_chart(df)
    
    if args.type in ['bar', 'all']:
        plot_bar_chart(df, num_matrices=20)
    
    if args.type in ['comparison', 'all']:
        plot_comparison_chart(df)
    
    # Print statistics
    if args.stats or args.type == 'all':
        print_statistics(df)

if __name__ == '__main__':
    main()

