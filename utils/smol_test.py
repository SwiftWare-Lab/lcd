import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
from read_compressed_format import read_compressed_matrix
from compressed_format_generator import csr_to_compressed_format_converter
import os

A_sp = sp.random(4, 4, density=1.0, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
mmwrite(f"A.mtx", A_sp)
os.system(f"python3 compressed_format_generator.py A.mtx 2 2 > /dev/null 2>&1")
A_cf1 = read_compressed_matrix("A.cfmtx")



kernel_m = 2
kernel_n = 2

patterns = {}
for i in range(1, 2 ** (kernel_m * 1)):
    x = 2 ** (kernel_m * 1) - i
    arr = np.array([True if x == '1' else False for x in bin(x)[2:].zfill(kernel_m * 1)])
    patterns[tuple(arr)] = i - 1

A_cf2 = csr_to_compressed_format_converter(A_sp, patterns, kernel_m, kernel_n)

print(A_cf1.patterns)
print(A_cf2.patterns)

print(A_cf1.padded_values_data_array == A_cf2.padded_values_data_array)
print(A_cf1.indices_array == A_cf2.indices_array)
print(A_cf1.row_panels_pointer == A_cf2.row_panels_pointer)
print(A_cf1.kernel_cols == A_cf2.vector_length)
print(A_cf1.kernel_rows == A_cf2.kernel_rows)
