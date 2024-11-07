import importlib
import os
import sys
import cupy as cp
from matplotlib import pyplot as plt
import re
import numpy as np
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore
from scipy.io import mmread
import scipy.sparse as sp
import torch
import triton

dir_name = 'experiment_kernels'
markers = ['o', 's', '^', 'D', 'p', '*', 'x']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
k = 10

def load_function(pattern_size, unroll_factor):
    module_name = f'pattern_{pattern_size}_{unroll_factor[0]}x{unroll_factor[1]}'
    module = importlib.import_module(f'{dir_name}.{module_name}')
    
    func = getattr(module, "benchmark_unroll_coarsen")
    return func

def benchmark_func(func, A_cf, B):

    return func(A_cf, B)

def load_matrix(matrix_name, shape, kernel_m, kernel_n):
    A_cf = read_compressed_matrix(f"cfmtx_matrices/{matrix_name}_{shape[0]}x{shape[1]}_{kernel_m}x{kernel_n}.cfmtx")
    return A_cf

def load_cfmtx_from_path(full_cfmtx_path):
    M_cf = read_compressed_matrix(full_cfmtx_path)
    return M_cf

def load_mtx_from_path(full_mtx_path):
    A_sp = mmread(full_mtx_path)
    return A_sp

def gen_sp_matrix(n_rows, n_cols, d):
    mat_sp = sp.random(n_rows, n_cols,
                       density=d, format='csr',
                       dtype=np.int32,
                       data_rvs=lambda s: np.random.randint(1, 11, size=s))
    return mat_sp

def cuSPARSE_instance(A, B):
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    A_sparse_csr = cp.sparse.csr_matrix(cp.asarray(A))

    # GPU-based timing (start)
    start_event.record()

    # Perform sparse matrix multiplication: A_sparse_csr.dot(B_dense)
    c_cuSparse = A_sparse_csr.dot(cp.asarray(B))

    end_event.record()
    end_event.synchronize()

    elapsed_time_cuSparse = cp.cuda.get_elapsed_time(start_event, end_event)

    return elapsed_time_cuSparse  

def benchmark_cuSPARSE(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    results = []
    for _ in range(k):
        results.append(cuSPARSE_instance(A, B))
    return np.mean(results)

def cuBLAS_instance(A, B):
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()

    c_cuBLAS = cp.matmul(cp.asarray(A), cp.asarray(B))

    # Wait for all operations to complete
    end_event.record()
    end_event.synchronize()

    elapsed_time_cuBLAS = cp.cuda.get_elapsed_time(start_event, end_event)

    return elapsed_time_cuBLAS

def benchmark_cuBLAS(A, B):
    results = []
    for _ in range(k):
        results.append(cuBLAS_instance(A, B))
    return np.mean(results)

def torch_instance(A_tensor, B_tensor):
    execution_time = 0

    # Check if tensors are on a GPU
    if A_tensor.is_cuda and B_tensor.is_cuda:
        # GPU timing using torch.cuda.Event
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = torch.matmul(A_tensor, B_tensor)
        end.record()
        
        # Waits for everything to finish running
        torch.cuda.synchronize()
        
        # Get elapsed time in milliseconds
        execution_time = start.elapsed_time(end)

    return execution_time


def benchmark_torch(A, B):
    A_tensor = torch.from_numpy(A).float().to(device='cuda')
    B_tensor = torch.from_numpy(B).float().to(device='cuda')

    median_time = triton.testing.do_bench(lambda: torch.matmul(A_tensor, B_tensor), quantiles=[0.5])

    return median_time

def get_TFLOPs(shape, b_col, exec_time_ms):
    M, K = shape
    total_ops = 2 * M * K * b_col
    exec_time_s = exec_time_ms / 1000.0
    tflops = (total_ops / 1e12) / exec_time_s

    return tflops

def gen_data_extracted():
    b_cols = [8, 16, 32]
    pattern_size = 4
    unroll_factor = (1, 1)
    func = load_function(pattern_size, unroll_factor)

    selected_matrices_file = 'mtx_matrices/selected_matrices.txt'
    with open(selected_matrices_file, 'r') as f:
        selected_matrices = [line.strip() for line in f]

    for rel_path in selected_matrices:

        rel_path = os.path.splitext(rel_path)[0]
        cfmtx_dir = 'cfmtx_matrices'
        mtx_dir = 'mtx_matrices'

        full_cfmtx_path = cfmtx_dir + rel_path + '.cfmtx'
        A_cf = load_cfmtx_from_path(full_cfmtx_path)
        n_cols = A_cf.num_col

        full_mtx_path = mtx_dir + rel_path + '.mtx'
        A_sp = load_mtx_from_path(full_mtx_path)
        A = A_sp.toarray()

        for b_col in b_cols:
            B_sp = gen_sp_matrix(n_cols, b_col, 1.0)
            B = B_sp.toarray()

            # Triton
            triton_time = benchmark_func(func, A_cf, B)
            triton_tflops = get_TFLOPs((A_cf.num_row, A_cf.num_col), b_col, triton_time)

            # cuSPARSE
            cuSPARSE_time = benchmark_cuSPARSE(A, B)
            cuSPARSE_tflops = get_TFLOPs(A.shape, b_col, cuSPARSE_time)

            # cuBLAS
            cuBLAS_time = benchmark_cuBLAS(A, B)
            cuBLAS_tflops = get_TFLOPs(A.shape, b_col, cuBLAS_time)

            # torch matmul
            torch_time = benchmark_torch(A, B)
            torch_tflops = get_TFLOPs(A.shape, b_col, torch_time)

            print(f"Triton {rel_path} {b_col} {triton_time} {triton_tflops}")
            print(f"cuSPARSE {rel_path} {b_col} {cuSPARSE_time} {cuSPARSE_tflops}")
            print(f"cuBLAS {rel_path} {b_col} {cuBLAS_time} {cuBLAS_tflops}")
            print(f"torch {rel_path} {b_col} {torch_time} {torch_tflops}")

def gen_data_large():
    pattern_size = 4
    unroll_factor = (1, 1)
    A_dims = [(7000, 7000), (7000, 28000), (9000, 9000), (9000, 36000), (12000, 12000), (21000, 7000), (27000, 9000), (28000, 7000), (36000, 9000), (36000, 12000), (48000, 12000)]
    b_cols = [8, 16, 32]

    dir_name = 'cfmtx_matrices'

    func = load_function(pattern_size, unroll_factor)

     # warm up
    warmup_mtx_path = f"cfmtx_matrices/A_1024x1024_4x4.cfmtx"
    A_cf = load_cfmtx_from_path(warmup_mtx_path)
    B_sp = gen_sp_matrix(1024, 64, 1.0)
    B = B_sp.toarray()
    for _ in range(10):
        benchmark_func(func, A_cf, B)

    for A_dim in A_dims:
        rel_mtx_path = f"/A_{A_dim[0]}x{A_dim[1]}.mtx"
        rel_cfmtx_path = f"/A_{A_dim[0]}x{A_dim[1]}_{pattern_size}x4.cfmtx"
        mtx_dir = 'mtx_matrices'
        full_mtx_path = mtx_dir + rel_mtx_path
        full_cfmtx_path = dir_name + rel_cfmtx_path
        A_cf = load_cfmtx_from_path(full_cfmtx_path)
        A_sp = load_mtx_from_path(full_mtx_path)
        A = A_sp.toarray()
        n_cols = A_cf.num_col
        for b_col in b_cols:
            B_sp = gen_sp_matrix(n_cols, b_col, 1.0)
            B = B_sp.toarray()
            avg_time = benchmark_func(func, A_cf, B)
            avg_tflops = get_TFLOPs((A_cf.num_row, A_cf.num_col), b_col, avg_time)
            cuSPARSE_time = benchmark_cuSPARSE(A, B)
            cuBLAS_time = benchmark_cuBLAS(A, B)
            cuSPARSE_tflops = get_TFLOPs(A.shape, b_col, cuSPARSE_time)
            cuBLAS_tflops = get_TFLOPs(A.shape, b_col, cuBLAS_time)
            torch_time = benchmark_torch(A, B)
            torch_tflops = get_TFLOPs(A.shape, b_col, torch_time)
            print(f"Triton {rel_cfmtx_path} {b_col} {avg_time} {avg_tflops}")
            print(f"cuSPARSE {rel_mtx_path} {b_col} {cuSPARSE_time} {cuSPARSE_tflops}")
            print(f"cuBLAS {rel_mtx_path} {b_col} {cuBLAS_time} {cuBLAS_tflops}")
            print(f"torch {rel_mtx_path} {b_col} {torch_time} {torch_tflops}")

def gen_plot_large(data_out):
    output_dir =  'output/'
    data_path = output_dir + data_out

    data = {}
    b_cols = [8, 16, 32]

    triton_pattern = re.compile(r'A_(\d+x\d+)_\d+x\d+\.cfmtx')
    other_pattern = re.compile(r'A_(\d+x\d+)\.mtx')

    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            kernel = parts[0]
            path = parts[1]
            b_col = int(parts[2])
            tflops = float(parts[4])

            if kernel == "Triton":
                match = triton_pattern.search(path)
            else:
                match = other_pattern.search(path)

            if match:
                matrix_size = match.group(1)
            
            # Organize by matrix size and b_col
                if matrix_size not in data:
                    data[matrix_size] = {8: {}, 16: {}, 32: {}}
                data[matrix_size][b_col][kernel] = tflops

        libraries = ['Triton', 'cuSPARSE', 'cuBLAS', 'torch']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(data_out.split('.')[0], fontsize=16)

        # Plot for each b_col in a separate subplot
        for idx, b_col in enumerate(b_cols):
            ax = axes[idx]
            
            # Prepare data for this b_col
            matrix_sizes = list(data.keys())  # List of matrix sizes for the x-axis
            num_matrices = len(matrix_sizes)
            bar_width = 0.2
            x = np.arange(num_matrices)  # the label locations
            
            # Plot each library as a set of bars within each b_col plot
            for i, library in enumerate(libraries):
                # Collect TFLOPS values for the current library across all matrix sizes for this b_col
                tflops_values = [
                    data[matrix_size].get(b_col, {}).get(library, 0) for matrix_size in matrix_sizes
                ]
                
                # Plot bars for the current library
                ax.bar(x + i * bar_width, tflops_values, bar_width, label=library, color=colors[i])
        
            # Labels and titles
            ax.set_title(f'Performance Comparison for b_col = {b_col}')
            ax.set_xlabel('Matrix Size')
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(matrix_sizes, rotation=60)

        # Set the y-axis label for the first plot
        axes[0].set_ylabel('TFLOPS')

        # Add a single legend for all subplots
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure to a PNG file
        output_png_path = os.path.join(output_dir, f"{data_out.split('.')[0]}.png")
        plt.savefig(output_png_path, format='png')

def gen_plot_dlmc(data_out, matrix_dir):
    output_dir = 'output/'
    data_path = output_dir + data_out

    data = {}
    b_cols = [8, 16, 32]
    
    # Read data file and organize by matrix size, nnz, and b_col
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            kernel = parts[0]
            path = matrix_dir + parts[1] + '.mtx'
            b_col = int(parts[2])
            tflops = float(parts[4])

            # Read matrix file to get size and nnz
            matrix = mmread(path).tocsc()
            nnz = matrix.nnz
            matrix_size = f"{matrix.shape[0]}x{matrix.shape[1]}"
            
            # Organize data by matrix size and b_col
            if matrix_size not in data:
                data[matrix_size] = {8: [], 16: [], 32: []}
            data[matrix_size][b_col].append((nnz, kernel, tflops))
    
    libraries = ['Triton', 'cuSPARSE', 'cuBLAS', 'torch']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors for the libraries
    matrix_sizes = list(data.keys())
    
    # Set up a grid for the plots
    fig, axes = plt.subplots(len(matrix_sizes), len(b_cols), figsize=(18, 6 * len(matrix_sizes)), sharey=True)
    fig.suptitle("Performance Comparison by Matrix Size and b_col", fontsize=16)

    # Generate subplots for each matrix size and b_col
    for row_idx, matrix_size in enumerate(matrix_sizes):
        for col_idx, b_col in enumerate(b_cols):
            ax = axes[row_idx, col_idx]
            ax.set_title(f"{matrix_size}, b_col = {b_col}")

            # Sort data by nnz for the current matrix size and b_col
            sorted_data = sorted(data[matrix_size][b_col], key=lambda x: x[0])
            library_data = {lib: [] for lib in libraries}

            # Organize TFLOPS by library for sorted nnz values
            for nnz, kernel, tflops in sorted_data:
                library_data[kernel].append((nnz, tflops))

            # Plot each library's TFLOPS data
            for i, library in enumerate(libraries):
                if library in library_data and library_data[library]:
                    x_vals, y_vals = zip(*library_data[library])
                    ax.plot(x_vals, y_vals, label=library, color=colors[i])

            # Labels and formatting
            ax.set_xlabel('Number of Non-Zero Elements (nnz)')
            if col_idx == 0:
                ax.set_ylabel('TFLOPS')
            ax.set_xscale('log')  # Use logarithmic scale for nnz
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    output_png_path = os.path.join(output_dir, f"{data_out.split('.')[0]}_combined.png")
    plt.savefig(output_png_path, format='png')


def plot_large_set(data_out):
    
    # Sample data file (replace this with the actual path to your data file)
    data_file = 'large_matrices_data.out'

    # Initialize dictionaries to store data
    sizes = []
    tflops_data = {8: [], 16: [], 32: []}
    b_cols = [8, 16, 32]  # Define the repeating pattern for b_col

    # Read and parse the data file
    with open(data_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            parts = line.split()
            path = parts[0]
            tflops = float(parts[2])

            mtx_dir = "mtx_matrices"
            full_mtx_path = mtx_dir + path.replace(".cfmtx", ".mtx")
            A_sp = load_mtx_from_path(full_mtx_path)
            A = A_sp.toarray()

            # Extract matrix size from the path
            match = re.search(r'A_(\d+)x(\d+)_4x4\.cfmtx', path)
            if match:
                size_a = f"{match.group(1)}x{match.group(2)}"
                if size_a not in sizes:
                    sizes.append(size_a)
                
                # Determine the b_col based on the position in the group (0 -> 8, 1 -> 16, 2 -> 32)
                b_col = b_cols[i % 3]

                B_sp = gen_sp_matrix(A_sp.shape[1], b_col)
                B = B_sp.toarray()
                avg_cuSPARSE_time = benchmark_cuSPARSE(A, B)
                avg_cuBLAS_time = benchmark_cuBLAS(A, B)

                avg_cuSPARSE_tflops = get_TFLOPs(A_sp.shape, b_col, avg_cuSPARSE_time)
                avg_cuBLAS_tflops = get_TFLOPs(A_sp.shape, b_col, avg_cuBLAS_time)
                
                # Store TFLOPS data for each b_col
                tflops_data[b_col].append(tflops)
                tflops_data[b_col].append(avg_cuSPARSE_tflops)
                tflops_data[b_col].append(avg_cuBLAS_tflops)


    # Set up the bar chart
    _, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.25  # Width of each bar
    index = range(len(sizes))  # X-axis positions for each group

    # Plot bars for each b_col value
    for j, b_col in enumerate(b_cols):
        offset = j * bar_width
        ax.bar(
            [x + offset for x in index],
            tflops_data[b_col],
            bar_width,
            label=f'b_col = {b_col}'
        )

    # Labeling
    ax.set_xlabel('Matrix Size of A')
    ax.set_ylabel('TFLOPS')
    ax.set_title('TFLOPS for Different Matrix Sizes and b_col Values')
    ax.set_xticks([x + bar_width for x in index])  # Adjust x-axis ticks to the middle of each group
    ax.set_xticklabels(sizes, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    # Save plot to file
    plt.savefig("output_large_set.png")

class CodeGenPlot:
    def __init__(self, dim, b_col, pattern_sizes, unroll_factors):
        self.dim = dim
        self.b_col = b_col
        self.pattern_sizes = pattern_sizes
        self.unroll_factors = unroll_factors
        self.loop_order = ['i', 'j']
        self.data = {}
        self.cuBLAS_res = 0
        self.cuSPARSE_res = 0

    def get_title(self):
        return f"A_dim: {self.dim} B_col: {self.b_col} loop_order: {self.loop_order}"

    def get_x_label(self):
        return "Sweep On M"
    
    def get_y_label(self):
        return "TFLOP/s"
    
    def generate_data(self):
        A_sp = mmread(f"mtx_matrices/A_{self.dim[0]}x{self.dim[1]}.mtx").tocsr()
        A = A_sp.toarray()
        B_sp = gen_sp_matrix(self.dim[1], self.b_col, 1.0)
        B = B_sp.toarray()

        for unroll_factor in self.unroll_factors:
            self.data[unroll_factor] = []

        for pattern_size in self.pattern_sizes:
            A_cf = load_matrix("A", self.dim, pattern_size, 4)
            for unroll_factor in self.unroll_factors:
                func = load_function(pattern_size, unroll_factor)

                # Warmup run for kernel
                for _ in range(20):
                    func(A_cf, B)

                # Benchmark the kernel
                average_time = benchmark_func(func, A_cf, B)
                triton_tflops = get_TFLOPs(self.dim, self.b_col, average_time)
                self.data[unroll_factor].append(triton_tflops)

        cuBLAS_time = benchmark_cuBLAS(A, B)
        self.cuBLAS_res = get_TFLOPs(self.dim, self.b_col, cuBLAS_time)
        cuSPARSE_time = benchmark_cuSPARSE(A, B)
        self.cuSPARSE_res = get_TFLOPs(self.dim, self.b_col, cuSPARSE_time)

    def gen_plot(self, ax):

        for i, ((unroll_i, unroll_j), y_values) in enumerate(self.data.items()):
            label = f"Unroll i: {unroll_i}, Unroll j: {unroll_j}"
            ax.plot(self.pattern_sizes, y_values, label=label, marker=markers[i % len(markers)], color=colors[i % len(colors)])

        ax.axhline(y=self.cuSPARSE_res, color='red', linestyle='--', linewidth=1, label=f'cuSPARSE')
        ax.axhline(y=self.cuBLAS_res, color='blue', linestyle='--', linewidth=1, label=f'cuBLAS')

        ax.set_title(self.get_title())
        ax.set_xlabel(self.get_x_label())
        ax.set_ylabel(self.get_y_label())
        ax.legend()
        ax.grid(True)

        return


def main():
    pattern_sizes = [4]
    unroll_factors = [(1, 1)]
    A_dims = [(7000, 7000)]
    b_cols = [8, 16, 32]

    plots = [CodeGenPlot(A_dim, b_col, pattern_sizes, unroll_factors) 
         for A_dim in A_dims 
         for b_col in b_cols]

    for plot in plots:
        plot.generate_data()

    plot_r = len(A_dims)
    plot_c = len(b_cols)
    _, axes = plt.subplots(plot_r, plot_c, figsize=(25, 6))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flat
    else:
        # Wrap `axes` in a list to make it iterable, consistent with `axes.flat` behavior.
        axes_flat = [axes]

    idx = 0
    for _ in b_cols:
        for _ in A_dims:
            plots[idx].gen_plot(axes_flat[idx])
            idx += 1

    plt.tight_layout()
    plt.savefig("output.png", dpi=300)

if __name__ == "__main__":
    #gen_data_large()
    out_file = 'data_large_a100_071124_1352.out'
    gen_plot_large(out_file)