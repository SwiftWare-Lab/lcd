import importlib
import sys
import cupy as cp
from matplotlib import pyplot as plt
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
    for _ in range(10):
        results.append(func(A_cf, B))
    return np.mean(results)

def load_matrix(matrix_name, shape, kernel_m, kernel_n):
    A_cf = read_compressed_matrix(f"cfmtx_matrices/{matrix_name}_{shape[0]}x{shape[1]}_{kernel_m}x{kernel_n}.cfmtx")
    return A_cf

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
        return "Execution Time (ms)"
    
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
                self.data[unroll_factor].append(average_time)

        self.cuBLAS_res = benchmark_cuBLAS(A, B)
        self.cuSPARSE_res = benchmark_cuSPARSE(A, B)

    def gen_plot(self, ax):

        for i, ((unroll_i, unroll_j), y_values) in enumerate(self.data.items()):
            label = f"Unroll i: {unroll_i}, Unroll j: {unroll_j}"
            ax.plot(self.pattern_sizes, y_values, label=label, marker=markers[i % len(markers)], color=colors[i % len(colors)])

        ax.axhline(y=self.cuSPARSE_res, color='gray', linestyle='--', linewidth=1, label=f'cuSPARSE')
        ax.axhline(y=self.cuBLAS_res, color='blue', linestyle='--', linewidth=1, label=f'cuBLAS')

        ax.set_title(self.get_title())
        ax.set_xlabel(self.get_x_label())
        ax.set_ylabel(self.get_y_label())
        ax.legend()
        ax.grid(True)

        return


def main():
    pattern_sizes = [1, 2, 3, 4, 5]
    unroll_factors = [(1, 1), (1, 2), (1, 4), (2, 1), (4, 1)]
    A_dims = [(768, 768)]
    b_cols = [32, 64, 128, 256]

    plots = [CodeGenPlot(A_dims[0], b_col, pattern_sizes, unroll_factors) for b_col in b_cols]

    for plot in plots:
        plot.generate_data()

    plot_r = 1
    plot_c = len(b_cols)
    fig, axes = plt.subplots(plot_r, plot_c, figsize=(10, 8))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flat
    else:
        # Wrap `axes` in a list to make it iterable, consistent with `axes.flat` behavior.
        axes_flat = [axes]

    idx = 0
    for b_col in b_cols:
        plots[idx].gen_plot(axes_flat[idx])
        idx += 1

    plt.tight_layout()
    plt.savefig("output.png", dpi=300)

if __name__ == "__main__":
    main()