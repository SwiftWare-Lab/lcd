import os
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import cusparse_sptrsv
import sys

matrix_path = sys.argv[1]

A = scipy.io.mmread(matrix_path).tocsr()
A = sp.tril(A).tocsr()

rowptr = A.indptr.astype(np.int32)
colind = A.indices.astype(np.int32)
values = A.data.astype(np.float64)

solver = cusparse_sptrsv.CuSparseSpTRSV(
    rowptr, colind, values, lower=True, unit_diag=False
)

# analysis timing
t0 = time.perf_counter()
solver.analyze()
t1 = time.perf_counter()

# warmup
for _ in range(10):
    solver.solve()

# solve timing
iters = 100
t0 = time.perf_counter()
for _ in range(iters):
    solver.solve()
t1 = time.perf_counter()

avg_ms = (t1 - t0) * 1000.0 / iters
gflops = 2.0 * solver.nnz / (avg_ms * 1e-3) / 1e9

matrix_name = os.path.splitext(os.path.basename(matrix_path))[0]
print(f"{matrix_name},{solver.nnz},{avg_ms:.6f},{gflops:.6f}")