import scipy.sparse as sp
import numpy as np

# for converting scipy arrays to cfmtx
import sys
sys.path.insert(1, '../utils')
from scipy.io import mmwrite
import os

def gen_sp_matrix(n_rows, n_cols, d):
    mat_sp = sp.random(n_rows, n_cols,
                       density=d, format='csr',
                       dtype=np.int32,
                       data_rvs=lambda s: np.random.randint(1, 11, size=s))
    return mat_sp

def generate_cftmx_matrix(A_sp, kernel_m, kernel_n, name):
    # Naming format is {name}_{num_rows}x{num_cols}_{kernel_m}x{kernel_n}.mtx
    dir_name = "cfmtx_matrices/"
    file_name = f"{name}_{A_sp.shape[0]}x{A_sp.shape[1]}_{kernel_m}x{kernel_n}"

    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    mmwrite(dir_name + file_name + ".mtx", A_sp)
    os.system(f"python3 ../utils/compressed_format_generator.py {dir_name + file_name}.mtx {kernel_m} {kernel_n} > /dev/null 2>&1")
    # remove .mtx file
    os.system(f"rm {dir_name + file_name}.mtx")

def generate_batch_cfmtx_matrices(mat_sizes, kernel_sizes, density):
    for mat_size in mat_sizes:
        for kernel_size in kernel_sizes:
            M_sp = gen_sp_matrix(mat_size[0], mat_size[1], density)
            generate_cftmx_matrix(M_sp, kernel_size[0], kernel_size[1], "A")

def main():
    mat_sizes = [(768, 768)]
    kernel_sizes = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)]
    density = 0.5
    generate_batch_cfmtx_matrices(mat_sizes, kernel_sizes, density)

if __name__ == "__main__":
    main()