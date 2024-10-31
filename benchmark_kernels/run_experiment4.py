import scipy.sparse as sp
import numpy as np
import cupy as cp

# for converting scipy arrays to cfmtx
import sys
sys.path.insert(1, '../utils')
from read_compressed_format import read_compressed_matrix # type: ignore
from scipy.io import mmwrite
import os

# for generating, running kernel code
sys.path.insert(1, '../code_gen')
from code_gen_utils import generate_kernel_code, run_generated_kernel # type: ignore


# for plotting
import matplotlib.pyplot as plt



loop_orders = [['i', 'j']]
A_dims = [(768, 768)]
B_cols = [32, 64, 128, 256]
pattern_sizes = [2, 3, 4, 5]
kernel_n = 4
unroll_factors = [(1,1), (1,2), (1,4), (2,1), (4, 1)]
A_density = 0.1
B_density = 1.0
k = 10 # Number of iterations to average

markers = ['o', 's', '^', 'D', 'p', '*', 'x']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def csr_to_cfmtx(M_sp, kernel_m, kernel_n):
    mmwrite("M.mtx", M_sp)
    os.system(f"python3 ../utils/compressed_format_generator.py M.mtx {kernel_m} {kernel_n} > /dev/null 2>&1")
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
    results = []
    for _ in range(k):
        results.append(run_generated_kernel(code, A_cf, B))
    return np.mean(results)

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
    c_cuSparse_ref = c_cuSparse.get()

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

def gen_plot(ax, loop_order, A_dim, B_col):
    print("Inside gen_plot()")
    A_sp = gen_sp_matrix(A_dim[0], A_dim[1], A_density)
    B_sp = gen_sp_matrix(A_dim[1], B_col, B_density)
    A = A_sp.toarray()
    B = B_sp.toarray()

    data = {}

    for unroll_factor in unroll_factors:
        results = []
        for pattern_size in pattern_sizes:

            A_cf = csr_to_cfmtx(A_sp, pattern_size, kernel_n)
            print("Converted to cfmtx")
            transformed_unroll_factor = {'i': unroll_factor[0], 'j': unroll_factor[1]}
            gen_file_code = generate_kernel_code(A_cf.patterns, transformed_unroll_factor, loop_order)
            print("Generated code")
            averaged_exec_time = benchmark_gen_code(gen_file_code, A_cf, B)
            print("Got benchmark time")
            results.append(averaged_exec_time)

        data[unroll_factor] = results
    print("Exited loop next")

    for i, ((unroll_i, unroll_j), y_values) in enumerate(data.items()):
        label = f"Unroll i: {unroll_i}, Unroll j: {unroll_j}"
        ax.plot(pattern_sizes, y_values, label=label, marker=markers[i % len(markers)], color=colors[i % len(colors)])
    print("Exited plotting lop")

    # TODO: add horizontal lines for cuSPARSE, cuBLAS
    cuSPARSE_res = benchmark_cuSPARSE(A, B)
    print("Benchmarked cuSPARSE")
    cuBLAS_res = benchmark_cuBLAS(A, B)
    print("Benchmarked cuBLAS")

    ax.axhline(y=cuSPARSE_res, color='gray', linestyle='--', linewidth=1, label=f'cuSPARSE')
    ax.axhline(y=cuBLAS_res, color='blue', linestyle='--', linewidth=1, label=f'cuBLAS')
    print("Added cuSPARSE, cuBLAS lines")

    ax.set_title(f" A: {A_dim[0]}x{A_dim[1]}, B_col: {B_col}")
    ax.set_xlabel('Sweep On M')
    ax.set_ylabel('Execution Time (ms)')
    ax.legend()
    ax.grid(True)
    print(f"A_dim: {A_dim} B_col: {B_col} loop_order: {loop_order}")

    return


rows, cols = len(loop_orders) * len(A_dims), len(B_cols)
fig, axes = plt.subplots(rows, cols, figsize=(10, 8))

if isinstance(axes, np.ndarray):
    axes_flat = axes.flat
else:
    # Wrap `axes` in a list to make it iterable, consistent with `axes.flat` behavior.
    axes_flat = [axes]

idx = 0

for loop_order in loop_orders:
    for A_dim in A_dims:
        for B_col in B_cols:
            # generate plot this setting
            print("Calling gen_plot()")
            gen_plot(axes_flat[idx], loop_order, A_dim, B_col)
            #print(f"loop_order: {loop_order}, A_dim: {A_dim}, B_col: {B_col}")
            idx += 1

# Adjust layout and show
plt.tight_layout()
plt.savefig("output.png", dpi=300)