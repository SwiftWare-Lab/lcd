import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# === Set your CSV file path here ===
csv_file = sys.argv[1]

# Read CSV file
df = pd.read_csv(csv_file)

# Extract matrix ID from first column path
# Example:
# /scratch/.../1138_bus/1138_bus.mtx  -> 1138_bus
def extract_matrix_id(path):
    return os.path.basename(path).replace(".mtx", "")

matrix_ids = df.iloc[:, 0].apply(extract_matrix_id)

# Extract performance columns
cusparse = df["CuSparse"]
triton = df["Best TR"]

# Use numeric indices for x-axis (matrix IDs hidden)
x = range(len(matrix_ids))

# Plot
plt.figure()
plt.plot(x, cusparse,  label="cusparse")
plt.plot(x, triton, label="triton-optimized")

plt.xlabel("Matrix ID")
plt.ylabel("Throughput (GFLOP/s)")
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()