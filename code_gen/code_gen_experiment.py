import importlib
import sys

import numpy as np
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore

import scipy.sparse as sp

class CodeGenPlot:
    def __init__(self, dim, b_col, pattern_sizes, unroll_factors):
        self.dim = dim
        self.b_col = b_col
        self.pattern_sizes = pattern_sizes
        self.unroll_factors = unroll_factors
        self.loop_order = ['i', 'j']

    def get_title(self):
        return f"A_dim: {self.dim} B_col: {self.b_col} loop_order: {self.loop_order}"

    def get_x_label(self):
        return "Sweep On M"
    
    def get_y_label(self):
        return "Execution Time (ms)"

dir_name = 'experiment_kernels'

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

def main():
    pattern_size = 2
    unroll_factor = (1, 2)
    func = load_function(pattern_size, unroll_factor)

    A_cf = read_compressed_matrix("cfmtx_matrices/A_768x768_2x4.cfmtx")

    B_sp = sp.random(768, 64,
                       density=1.0, format='csr',
                       dtype=np.int32,
                       data_rvs=lambda s: np.random.randint(1, 11, size=s))
    B = B_sp.toarray()

    for _ in range(10):
        print(benchmark_func(func, A_cf, B))

if __name__ == "__main__":
    main()