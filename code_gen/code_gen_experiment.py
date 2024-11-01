import importlib
import sys

import numpy as np
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore

import scipy.sparse as sp


dir_name = 'experiment_kernels'

def load_function(pattern_size, unroll_factor):
    module_name = f'pattern_{pattern_size}_{unroll_factor[0]}x{unroll_factor[1]}'
    module = importlib.import_module(f'{dir_name}.{module_name}')
    
    func = getattr(module, "benchmark_unroll_coarsen")
    return func

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
        print(func(A_cf, B))

if __name__ == "__main__":
    main()