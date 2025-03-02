# Script to generate sparse/dense matrices with varying densities

import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
import os

matrices_dir = '../matrices'

if not os.path.exists(matrices_dir):
    os.makedirs(matrices_dir)

# Define the dimensions of the matrices
row_n= 768
cols = [32, 64, 128, 256, 512]
densities = np.arange(0.1, 0.8, 0.1)

for density in densities:
    A_sp = sp.random(row_n, 512, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
    mmwrite(f"{matrices_dir}/A_sparse_{row_n}x512_{density:.2f}.mtx", A_sp)
    for col in cols:
        B_sp = sp.random(512, col, density=1.0, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
        mmwrite(f"{matrices_dir}/B_sparse_512x{col}_1.0.mtx", B_sp)
