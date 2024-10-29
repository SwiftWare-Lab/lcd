import scipy.sparse as sp
import numpy as np

# for converting scipy arrays to cfmtx
import sys
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore
from scipy.io import mmwrite
import os

# for generating, running kernel code
sys.path.insert(1, '../code_gen')
from code_gen_utils import generate_kernel_code # type: ignore


# for plotting
import matplotlib.pyplot as plt



loop_orders = [['i', 'j'], ['j', 'i']]
A_dims = [(768, 768), (2048, 2048), (1024, 1024), (4096, 2048), (2048, 4096)]
B_cols = [128, 256, 512, 1024]
pattern_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
kernel_n = 5
unroll_factors = [(1,1), (2,1), (4,1), (8,1), (1,2), (1,4), (1,8)]
A_density = 0.5
B_density = 1.0

markers = ['o', 's', '^', 'D', 'p', '*', 'x']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def csr_to_cfmtx(M_sp, kernel_m, kernel_n):
    mmwrite("M.mtx", M_sp)
    os.system(f"python3 ../utils/compressed_format_generator.py M.mtx {kernel_m} {kernel_n} > /dev/null 2>&1")
    print("RAN COMPRESSED FORMAT GENERATOR")
    M_cf = read_compressed_matrix("M.cfmtx")
    return M_cf

def get_patterns(kernel_m):
    patterns = {}
    for i in range(1, 2 ** (kernel_m * 1)):
        x = 2 ** (kernel_m * 1) - i
        arr = np.array([True if x == '1' else False for x in bin(x)[2:].zfill(kernel_m * 1)])
        patterns[tuple(arr)] = i - 1

    return patterns

def transform_patterns(patterns):
    transformed = {}
    for key, value in patterns.items():
        # Convert the tuple to a string of '1's and '0's
        key_str = ''.join(['1' if k else '0' for k in key])
        # Map the original value to this new string
        transformed[value] = key_str
    return transformed


def gen_sp_matrix(n_rows, n_cols, d):
    mat_sp = sp.random(n_rows, n_cols,
                       density=d, format='csr',
                       dtype=np.int32,
                       data_rvs=lambda s: np.random.randint(1, 11, size=s))
    return mat_sp

def benchmark_gen_code(code, A_cf, B):
    return 0

def gen_plot(ax, loop_order, A_dim, B_col):
    A_sp = gen_sp_matrix(A_dim[0], A_dim[1], A_density)
    B_sp = gen_sp_matrix(A_dim[1], B_col, B_density)
    B = B_sp.toarray()

    data = {}

    for unroll_factor in unroll_factors:
        results = []
        for pattern_size in pattern_sizes:

            A_cf = csr_to_cfmtx(A_sp, pattern_size, kernel_n)

            transformed_unroll_factor = {'i': unroll_factor[0], 'j': unroll_factor[1]}
            gen_file_code = generate_kernel_code(A_cf.patterns, transformed_unroll_factor, loop_order)

            averaged_exec_time = benchmark_gen_code(gen_file_code, A_cf, B)
            results.append(averaged_exec_time)

        data[unroll_factor] = results

    for i, ((unroll_i, unroll_j), y_values) in enumerate(data.items()):
        label = f"Unroll i: {unroll_i}, Unroll j: {unroll_j}"
        ax.plot(pattern_sizes, y_values, label=label, marker=markers[i % len(markers)], color=colors[i % len(colors)])

    ax.set_title(f"(loop order: {loop_order[0]}, {loop_order[1]}) A: {A_dim[0]}x{A_dim[1]}, B_col: {B_col}")
    ax.set_xlabel('Sweep On M')
    ax.set_ylabel('Execution Time (ms)')
    ax.legend()
    ax.grid(True)

    return


rows, cols = len(loop_orders) * len(A_dims), len(B_cols)
fig, axes = plt.subplots(rows, cols, figsize=(10, 8))

axes_flat = axes.flat

idx = 0

for loop_order in loop_orders:
    for A_dim in A_dims:
        for B_col in B_cols:
            # generate plot this setting
            #gen_plot(axes_flat[idx], loop_order, A_dim, B_col)
            print(f"loop_order: {loop_order}, A_dim: {A_dim}, B_col: {B_col}")
            idx += 1

# Adjust layout and show
plt.tight_layout()
plt.savefig("output.png", dpi=300)