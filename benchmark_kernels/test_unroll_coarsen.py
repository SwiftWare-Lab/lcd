from scipy.io import mmread
import os
import sys
import numpy as np
from benchmark_unroll_coarsen2 import test_unroll_coarsen

sys.path.insert(1, '../utils')

from read_compressed_format import read_compressed_matrix

# tests for benchmark_unroll_coarsen2.py

cols = [32, 64, 128, 256, 512]
row_n = 768
densities = np.arange(0.1, 0.8, 0.1)

passed = 0
total = 0

for col in cols:
    for density in densities:
        A_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.mtx"
        B_path = f"../matrices/B_sparse_512x{col}_1.0.mtx"

        # Load A and B as np arrays
        A_sp = mmread(A_path)
        A_sp = A_sp.tocsr()
        B_sp = mmread(B_path)

        # Convert scipy arrays to numpy arrays
        A = A_sp.toarray().astype(np.int32)
        B = B_sp.toarray().astype(np.float32)
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

        os.system(f"python3 ../utils/compressed_format_generator.py {A_path} 3 4 > /dev/null 2>&1")

        A_cf_path = f"../matrices/A_sparse_{row_n}x128_{density:.2f}.cfmtx"
        A_cf = read_compressed_matrix(A_cf_path)

        exec_time, C = test_unroll_coarsen(A_cf, B)

        C_ref = A @ B

        if np.allclose(C, C_ref, atol=1e-3):
            passed += 1

        else:
            print(f"Test failed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx")
            print(f"Expected: {C_ref}")
            print(f"Got: {C}")
        
        total += 1

print(f"Passed {passed}/{total} tests")

                                       