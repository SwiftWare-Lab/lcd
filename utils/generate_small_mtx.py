import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite

# Generate a small 4x4 sparse matrix with random non-zero values
rows, cols = 4, 3
density = 0.3  # 30% of the matrix will be non-zero

# Create the sparse matrix in CSR format
sparse_matrix = sp.random(rows, cols, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))

# Save the sparse matrix to a .mtx (Matrix Market) file
mmwrite("small_matrix.mtx", sparse_matrix)

# Print out the matrix
print("Sparse Matrix (CSR):\n", sparse_matrix)
# Print out the matrix as an array
print("Matrix as Array:\n", sparse_matrix.toarray())