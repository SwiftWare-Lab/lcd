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
    results = []
    for _ in range(k):
        results.append(func(A_cf, B))
    return np.mean(results)

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
    dir_name = 'cfmtx_matrices'

    selected_matrices_file = 'mtx_matrices/selected_matrices.txt'
    with open(selected_matrices_file, 'r') as f:
        selected_matrices = [line.strip() for line in f]

    for rel_path in selected_matrices:
        rel_cfmtx_path = os.path.splitext(rel_path)[0] + '.cfmtx'
        full_cfmtx_path = dir_name + rel_cfmtx_path
        A_cf = load_cfmtx_from_path(full_cfmtx_path)
        n_cols = A_cf.num_col
        for b_col in b_cols:
            B_sp = gen_sp_matrix(n_cols, b_col, 1.0)
            B = B_sp.toarray()
            avg_time = benchmark_func(func, A_cf, B)
            avg_tflops = get_TFLOPs((A_cf.num_row, A_cf.num_col), b_col, avg_time)
            print(f"{rel_path} {avg_time} {avg_tflops}")

def gen_data_large():
    pattern_size = 4
    unroll_factor = (1, 1)
    A_dims = [(7000, 7000), (7000, 28000), (9000, 9000), (9000, 36000), (12000, 12000), (21000, 7000), (27000, 9000), (28000, 7000), (36000, 9000), (36000, 12000), (48000, 12000)]
    b_cols = [8, 16, 32]

    dir_name = 'cfmtx_matrices'

    func = load_function(pattern_size, unroll_factor)

    for A_dim in A_dims:
        rel_cfmtx_path = f"/A_{A_dim[0]}x{A_dim[1]}_{pattern_size}x4.cfmtx"
        full_cfmtx_path = dir_name + rel_cfmtx_path
        A_cf = load_cfmtx_from_path(full_cfmtx_path)
        n_cols = A_cf.num_col
        for b_col in b_cols:
            B_sp = gen_sp_matrix(n_cols, b_col, 1.0)
            B = B_sp.toarray()
            avg_time = benchmark_func(func, A_cf, B)
            avg_tflops = get_TFLOPs((A_cf.num_row, A_cf.num_col), b_col, avg_time)
            print(f"{rel_cfmtx_path} {avg_time} {avg_tflops}")

def plot_large_set():
    
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
    plot_large_set()