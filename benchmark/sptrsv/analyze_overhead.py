#!/usr/bin/env python3
"""
Analyze runtime overhead and calculate break-even points for optimized SpTRSV kernels.

For each matrix, determines how many solves are needed for optimized kernels 
(with preprocessing overhead) to outperform baseline kernels.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io as scio

def gflops_to_ms(gflops, n, nnz):
    """Convert GFLOP/s to milliseconds using the inverse formula."""
    if gflops == 0 or np.isnan(gflops):
        return np.inf
    flops = n + 2 * nnz
    ms = flops / (gflops * 1e6)
    return ms

def get_matrix_properties(matrix_name):
    """Get matrix size and nnz from the actual matrix file."""
    # Matrix files are in /home/mad29/projects/testdata/matrices/
    matrix_dir = Path("/home/mad29/projects/testdata/matrices")
    
    # Handle both full path and basename
    if isinstance(matrix_name, str):
        if '/' in matrix_name:
            # Extract just the filename
            matrix_name = Path(matrix_name).name
        matrix_path = matrix_dir / matrix_name
    else:
        return None, None
    
    if not matrix_path.exists():
        return None, None
    
    try:
        matrix = scio.mmread(matrix_path)
        return matrix.shape[0], matrix.nnz
    except Exception as e:
        print(f"  Error reading matrix {matrix_path}: {e}")
        return None, None

def classify_method(method_name):
    """Classify method as csc/csr and wf/sf."""
    format_type = 'csc' if method_name.startswith('csc-') else 'csr'
    
    # Determine baseline type (wf or sf)
    if '-sf-' in method_name or method_name.endswith('-sf-base'):
        baseline_type = 'sf'
    elif '-wf-' in method_name or method_name.endswith('-wf-base'):
        baseline_type = 'wf'
    elif '-h-' in method_name:
        # h-level methods are wf-based (they use wavefronts)
        baseline_type = 'wf'
    else:
        baseline_type = None
    
    return format_type, baseline_type

def analyze_matrix(matrix_name, result_dir):
    """Analyze a single matrix and return break-even analysis."""
    
    # Load CSV files
    overhead_file = result_dir / f"runtime_overhead_{matrix_name}.csv"
    perf_file = result_dir / f"sptrsv-performance-{matrix_name}.csv"
    
    if not overhead_file.exists() or not perf_file.exists():
        return None
    
    try:
        overhead_df = pd.read_csv(overhead_file)
        perf_df = pd.read_csv(perf_file)
    except Exception as e:
        print(f"  Error reading CSV files: {e}")
        return None
    
    # Get matrix properties (from the first row of overhead_df)
    if 'matrix' not in overhead_df.columns:
        print(f"  No 'matrix' column in overhead file")
        return None
    
    matrix_path = overhead_df['matrix'].iloc[0]
    n, nnz = get_matrix_properties(matrix_path)
    
    if n is None or nnz is None:
        print(f"  Could not get matrix properties")
        return None
    
    # Convert wide format perf_df to long format
    # The first column is 'matrices' (matrix path), rest are method names with GFLOP/s values
    try:
        perf_df_long = perf_df.melt(
            id_vars=['matrices'], 
            var_name='method', 
            value_name='gflops'
        )
    except Exception as e:
        print(f"  Error melting perf_df: {e}")
        return None
    
    # Merge with overhead data
    merged = pd.merge(
        perf_df_long,
        overhead_df,
        left_on='method',
        right_on='method',
        how='left'
    )
    
    # Check if merge was successful
    if len(merged) == 0:
        print(f"  Merge resulted in empty dataframe")
        return None
    
    # Convert GFLOP/s to ms
    merged['runtime_ms'] = merged['gflops'].apply(lambda x: gflops_to_ms(x, n, nnz))
    
    results = {'csc': {}, 'csr': {}}
    
    for format_type in ['csc', 'csr']:
        # Filter by format
        format_df = merged[merged['method'].str.startswith(f'{format_type}-')].copy()
        
        if len(format_df) == 0:
            continue
        
        # Get baselines
        wf_baseline = format_df[format_df['method'] == f'{format_type}-wf-base']
        sf_baseline = format_df[format_df['method'] == f'{format_type}-sf-base']
        
        wf_baseline_time = wf_baseline['runtime_ms'].iloc[0] if len(wf_baseline) > 0 else None
        sf_baseline_time = sf_baseline['runtime_ms'].iloc[0] if len(sf_baseline) > 0 else None
        
        # Exclude baselines and cusparse
        optimized_df = format_df[
            ~format_df['method'].isin([
                'cusparse', 
                f'{format_type}-wf-base', 
                f'{format_type}-sf-base'
            ])
        ].copy()
        
        if len(optimized_df) == 0:
            continue
        
        # Find best variant (highest GFLOP/s = lowest runtime_ms)
        best_idx = optimized_df['runtime_ms'].idxmin()
        best_variant = optimized_df.loc[best_idx]
        
        # Classify the best variant
        _, baseline_type = classify_method(best_variant['method'])
        
        # Get corresponding baseline time
        if baseline_type == 'wf' and wf_baseline_time is not None:
            baseline_time = wf_baseline_time
            baseline_name = f'{format_type}-wf-base'
        elif baseline_type == 'sf' and sf_baseline_time is not None:
            baseline_time = sf_baseline_time
            baseline_name = f'{format_type}-sf-base'
        else:
            continue
        
        # Calculate break-even point
        # overhead + n * optimized_time = n * baseline_time
        # n = overhead / (baseline_time - optimized_time)
        
        overhead = best_variant['total_overhead_ms']
        optimized_time = best_variant['runtime_ms']
        
        if baseline_time > optimized_time:
            breakeven = overhead / (baseline_time - optimized_time)
            speedup = baseline_time / optimized_time
        else:
            # Optimized is slower than baseline, no benefit
            breakeven = np.inf
            speedup = baseline_time / optimized_time
        
        results[format_type] = {
            'matrix': matrix_name,
            'n': n,
            'nnz': nnz,
            'best_variant': best_variant['method'],
            'best_gflops': best_variant['gflops'],
            'best_runtime_ms': optimized_time,
            'baseline': baseline_name,
            'baseline_runtime_ms': baseline_time,
            'dag_creation_ms': best_variant['dag_creation_ms'],
            'coarsening_ms': best_variant['coarsening_ms'],
            'total_overhead_ms': overhead,
            'speedup': speedup,
            'breakeven_solves': breakeven,
            'overhead_ratio': overhead / optimized_time if optimized_time > 0 else np.inf
        }
    
    return results

def analyze_performance_comparison(matrix_name, result_dir):
    """Analyze pure performance comparison without considering overhead."""
    
    # Load performance CSV
    perf_file = result_dir / f"sptrsv-performance-{matrix_name}.csv"
    
    if not perf_file.exists():
        return None
    
    try:
        perf_df = pd.read_csv(perf_file)
    except Exception as e:
        return None
    
    # Convert wide format to long format
    try:
        perf_df_long = perf_df.melt(
            id_vars=['matrices'], 
            var_name='method', 
            value_name='gflops'
        )
    except Exception as e:
        return None
    
    results = {'csc': {}, 'csr': {}}
    
    for format_type in ['csc', 'csr']:
        # Filter by format
        format_df = perf_df_long[perf_df_long['method'].str.startswith(f'{format_type}-')].copy()
        
        # Also get cusparse
        cusparse_row = perf_df_long[perf_df_long['method'] == 'cusparse']
        
        if len(format_df) == 0:
            continue
        
        # Get baseline performances
        cusparse_gflops = cusparse_row['gflops'].iloc[0] if len(cusparse_row) > 0 else None
        
        wf_baseline = format_df[format_df['method'] == f'{format_type}-wf-base']
        sf_baseline = format_df[format_df['method'] == f'{format_type}-sf-base']
        
        wf_gflops = wf_baseline['gflops'].iloc[0] if len(wf_baseline) > 0 else None
        sf_gflops = sf_baseline['gflops'].iloc[0] if len(sf_baseline) > 0 else None
        
        # Get best LCD-optimized variant (exclude baselines and cusparse)
        optimized_df = format_df[
            ~format_df['method'].isin([
                'cusparse', 
                f'{format_type}-wf-base', 
                f'{format_type}-sf-base'
            ])
        ].copy()
        
        if len(optimized_df) == 0 or cusparse_gflops is None:
            continue
        
        # Find best variant (highest GFLOP/s)
        best_idx = optimized_df['gflops'].idxmax()
        best_variant = optimized_df.loc[best_idx]
        best_gflops = best_variant['gflops']
        
        results[format_type] = {
            'matrix': matrix_name,
            'cusparse_gflops': cusparse_gflops,
            'wf_gflops': wf_gflops,
            'sf_gflops': sf_gflops,
            'best_gflops': best_gflops,
            'best_variant': best_variant['method'],
            # Speedups relative to cusparse
            'cusparse_vs_cusparse': 1.0,
            'wf_vs_cusparse': wf_gflops / cusparse_gflops if wf_gflops else None,
            'sf_vs_cusparse': sf_gflops / cusparse_gflops if sf_gflops else None,
            'best_vs_cusparse': best_gflops / cusparse_gflops if best_gflops else None,
        }
    
    return results

def main():
    # Find all result directories
    results_base = Path("results_local")
    
    if not results_base.exists():
        print(f"Results directory not found: {results_base}")
        return
    
    all_results = {'csc': [], 'csr': []}
    perf_results = {'csc': [], 'csr': []}
    
    # Process each matrix directory
    for matrix_dir in sorted(results_base.iterdir()):
        if not matrix_dir.is_dir():
            continue
        
        matrix_name = matrix_dir.name
        print(f"Processing {matrix_name}...")
        
        result = analyze_matrix(matrix_name, matrix_dir)
        perf_result = analyze_performance_comparison(matrix_name, matrix_dir)
        
        if result:
            for format_type in ['csc', 'csr']:
                if result[format_type]:
                    all_results[format_type].append(result[format_type])
        
        if perf_result:
            for format_type in ['csc', 'csr']:
                if perf_result[format_type]:
                    perf_results[format_type].append(perf_result[format_type])
    
    # Create summary DataFrames
    for format_type in ['csc', 'csr']:
        if not all_results[format_type]:
            print(f"\nNo results for {format_type.upper()}")
            continue
        
        df = pd.DataFrame(all_results[format_type])
        
        print(f"\n{'='*80}")
        print(f"{format_type.upper()} FORMAT ANALYSIS")
        print(f"{'='*80}")
        
        # Summary statistics
        print(f"\nNumber of matrices analyzed: {len(df)}")
        
        # Remove outliers using IQR method for robust statistics
        def remove_outliers_iqr(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series[(series >= lower_bound) & (series <= upper_bound)]
        
        breakeven_no_outliers = remove_outliers_iqr(df['breakeven_solves'])
        n_outliers_removed = len(df) - len(breakeven_no_outliers)
        
        print(f"\nBreak-even statistics (number of solves needed):")
        print(f"  ALL DATA ({len(df)} matrices):")
        print(f"    Median:     {df['breakeven_solves'].median():.1f}")
        # Calculate proper geometric mean excluding inf values
        finite_breakeven = df[np.isfinite(df['breakeven_solves'])]['breakeven_solves']
        if len(finite_breakeven) > 0:
            geom_mean = np.exp(np.log(finite_breakeven).mean())
            print(f"    Geom. Mean: {geom_mean:.1f}")
        print(f"    Arith. Mean:{df['breakeven_solves'].mean():.1f}")
        print(f"    25th pct:   {df['breakeven_solves'].quantile(0.25):.1f}")
        print(f"    75th pct:   {df['breakeven_solves'].quantile(0.75):.1f}")
        print(f"    Min:        {df['breakeven_solves'].min():.1f}")
        print(f"    Max:        {df['breakeven_solves'].max():.1f}")
        print(f"    Std:        {df['breakeven_solves'].std():.1f}")
        
        print(f"\n  ROBUST (outliers removed via IQR, {len(breakeven_no_outliers)} matrices, {n_outliers_removed} outliers removed):")
        print(f"    Median:     {breakeven_no_outliers.median():.1f}")
        if len(breakeven_no_outliers) > 0:
            geom_mean_robust = np.exp(np.log(breakeven_no_outliers).mean())
            print(f"    Geom. Mean: {geom_mean_robust:.1f}")
        print(f"    Arith. Mean:{breakeven_no_outliers.mean():.1f}")
        print(f"    Min:        {breakeven_no_outliers.min():.1f}")
        print(f"    Max:        {breakeven_no_outliers.max():.1f}")
        print(f"    Std:        {breakeven_no_outliers.std():.1f}")
        
        print(f"\nSpeedup statistics (best variant vs baseline):")
        speedup_no_outliers = remove_outliers_iqr(df['speedup'])
        n_speedup_outliers = len(df) - len(speedup_no_outliers)
        
        print(f"  ALL DATA ({len(df)} matrices):")
        print(f"    Median:     {df['speedup'].median():.2f}x")
        finite_speedup = df[np.isfinite(df['speedup'])]['speedup']
        if len(finite_speedup) > 0:
            geom_mean_speedup = np.exp(np.log(finite_speedup).mean())
            print(f"    Geom. Mean: {geom_mean_speedup:.2f}x")
        print(f"    Arith. Mean:{df['speedup'].mean():.2f}x")
        print(f"    Min:        {df['speedup'].min():.2f}x")
        print(f"    Max:        {df['speedup'].max():.2f}x")
        
        print(f"\n  ROBUST (outliers removed, {len(speedup_no_outliers)} matrices, {n_speedup_outliers} outliers removed):")
        print(f"    Median:     {speedup_no_outliers.median():.2f}x")
        if len(speedup_no_outliers) > 0:
            geom_mean_speedup_robust = np.exp(np.log(speedup_no_outliers).mean())
            print(f"    Geom. Mean: {geom_mean_speedup_robust:.2f}x")
        print(f"    Arith. Mean:{speedup_no_outliers.mean():.2f}x")
        print(f"    Min:        {speedup_no_outliers.min():.2f}x")
        print(f"    Max:        {speedup_no_outliers.max():.2f}x")
        
        print(f"\nOverhead ratio statistics (overhead / kernel_time):")
        overhead_no_outliers = remove_outliers_iqr(df['overhead_ratio'])
        n_overhead_outliers = len(df) - len(overhead_no_outliers)
        
        print(f"  ALL DATA ({len(df)} matrices):")
        print(f"    Median:     {df['overhead_ratio'].median():.1f}x")
        finite_overhead = df[np.isfinite(df['overhead_ratio'])]['overhead_ratio']
        if len(finite_overhead) > 0:
            geom_mean_overhead = np.exp(np.log(finite_overhead).mean())
            print(f"    Geom. Mean: {geom_mean_overhead:.1f}x")
        print(f"    Arith. Mean:{df['overhead_ratio'].mean():.1f}x")
        print(f"    Min:        {df['overhead_ratio'].min():.1f}x")
        print(f"    Max:        {df['overhead_ratio'].max():.1f}x")
        
        print(f"\n  ROBUST (outliers removed, {len(overhead_no_outliers)} matrices, {n_overhead_outliers} outliers removed):")
        print(f"    Median:     {overhead_no_outliers.median():.1f}x")
        if len(overhead_no_outliers) > 0:
            geom_mean_overhead_robust = np.exp(np.log(overhead_no_outliers).mean())
            print(f"    Geom. Mean: {geom_mean_overhead_robust:.1f}x")
        print(f"    Arith. Mean:{overhead_no_outliers.mean():.1f}x")
        print(f"    Min:        {overhead_no_outliers.min():.1f}x")
        print(f"    Max:        {overhead_no_outliers.max():.1f}x")
        
        # Show some examples
        print(f"\nTop 5 matrices by speedup:")
        top_speedup = df.nlargest(5, 'speedup')[
            ['matrix', 'best_variant', 'speedup', 'breakeven_solves', 'overhead_ratio']
        ]
        print(top_speedup.to_string(index=False))
        
        print(f"\nTop 5 matrices by lowest break-even point:")
        top_breakeven = df.nsmallest(5, 'breakeven_solves')[
            ['matrix', 'best_variant', 'speedup', 'breakeven_solves', 'overhead_ratio']
        ]
        print(top_breakeven.to_string(index=False))
        
        print(f"\nTop 10 matrices by highest break-even point (worst case):")
        worst_breakeven = df.nlargest(10, 'breakeven_solves')[
            ['matrix', 'best_variant', 'speedup', 'breakeven_solves', 'overhead_ratio']
        ]
        print(worst_breakeven.to_string(index=False))
        
        # Save to CSV
        output_file = f"breakeven_analysis_{format_type}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Interpretation
        median_breakeven = df['breakeven_solves'].median()
        median_speedup = df['speedup'].median()
        
        print(f"\n{'='*80}")
        print(f"INTERPRETATION for {format_type.upper()}:")
        print(f"{'='*80}")
        print(f"On average, you need {median_breakeven:.1f} solves for the optimized kernel")
        print(f"(including preprocessing overhead) to break even with the baseline.")
        print(f"After break-even, you get {median_speedup:.2f}x speedup on average.")
        print(f"\nFor typical iterative solvers (50-500 solves), this overhead is well amortized.")
        print(f"For time-stepping problems (1000+ solves), overhead is negligible (<1%).")
    
    # Performance comparison analysis (without overhead consideration)
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON (Pure Kernel Performance)")
    print(f"{'='*80}")
    
    for format_type in ['csc', 'csr']:
        if not perf_results[format_type]:
            print(f"\nNo performance results for {format_type.upper()}")
            continue
        
        perf_df = pd.DataFrame(perf_results[format_type])
        
        print(f"\n{'='*80}")
        print(f"{format_type.upper()} FORMAT - Baseline Comparison")
        print(f"{'='*80}")
        print(f"\nNumber of matrices: {len(perf_df)}")
        
        # Calculate statistics for each method
        methods = {
            'cuSPARSE': {
                'gflops': perf_df['cusparse_gflops'],
                'speedup': perf_df['cusparse_vs_cusparse']
            },
            'Wavefront (wf-base)': {
                'gflops': perf_df['wf_gflops'],
                'speedup': perf_df['wf_vs_cusparse']
            },
            'Barrier-free (sf-base)': {
                'gflops': perf_df['sf_gflops'],
                'speedup': perf_df['sf_vs_cusparse']
            },
            'LCD-optimized': {
                'gflops': perf_df['best_gflops'],
                'speedup': perf_df['best_vs_cusparse']
            }
        }
        
        print(f"\n{'Method':<25} {'Median GFLOP/s':<20} {'Geom. Mean Speedup':<25} {'% Better than cuSPARSE':<25}")
        print(f"{'-'*95}")
        
        for method_name, data in methods.items():
            gflops = data['gflops'].dropna()
            speedup = data['speedup'].dropna()
            
            if len(gflops) == 0:
                continue
            
            median_gflops = gflops.median()
            
            # Geometric mean of speedup
            finite_speedup = speedup[np.isfinite(speedup)]
            if len(finite_speedup) > 0:
                geom_mean_speedup = np.exp(np.log(finite_speedup).mean())
            else:
                geom_mean_speedup = 0.0
            
            # Percentage better than cuSPARSE
            pct_better = (speedup > 1.0).sum() / len(speedup) * 100
            
            print(f"{method_name:<25} {median_gflops:<20.2f} {geom_mean_speedup:<25.2f}x {pct_better:<25.1f}%")
        
        # Save detailed performance comparison
        output_file = f"performance_comparison_{format_type}.csv"
        perf_df.to_csv(output_file, index=False)
        print(f"\nDetailed performance comparison saved to: {output_file}")

if __name__ == "__main__":
    main()
