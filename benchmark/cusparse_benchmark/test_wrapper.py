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

# Benchmark analysis
t0 = time.perf_counter()
solver.analyze()
t1 = time.perf_counter()
analysis_ms = (t1 - t0) * 1e3

b = np.ones(A.shape[0], dtype=np.float64)

# Warmup
for _ in range(10):
    solver.solve(b)

# Benchmark solve
iters = 100
t0 = time.perf_counter()
for _ in range(iters):
    x = solver.solve(b)
t1 = time.perf_counter()

avg_ms = (t1 - t0) * 1e3 / iters
gflops = 2.0 * solver.nnz / (avg_ms * 1e-3) / 1e9

print(f"{matrix_path},{solver.nnz},{avg_ms:.6f},{gflops:.6f}")
