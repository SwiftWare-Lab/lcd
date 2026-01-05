import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load data
# -------------------------------
csv_file = "cudagraph-results2.csv"
df = pd.read_csv(csv_file)

# Sanity checks
required_cols = {"impl", "matrix", "gflops", "nnz"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# -------------------------------
# Pivot and sort by nnz
# -------------------------------
pivot = df.pivot(index="matrix", columns="impl", values="gflops")
nnz_map = df.drop_duplicates("matrix").set_index("matrix")["nnz"]
pivot["nnz"] = nnz_map

pivot = pivot.sort_values("nnz")

matrices = pivot.index.tolist()
x = np.arange(len(matrices))

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(12, 5))
ax = plt.gca()

ax.plot(
    x,
    pivot["cudagraph"],
    marker="o",
    linewidth=2,
    label="CUDA Graph"
)

ax.plot(
    x,
    pivot["cusparse"],
    marker="s",
    linewidth=2,
    label="cuSPARSE"
)

ax.set_xticks(x)
# ax.set_xticklabels(matrices, rotation=45, ha="right")

ax.set_xlabel("Matrix (sorted by nnz)")
ax.set_ylabel("GFLOP/s")
ax.set_xticks([])
# ax.set_title("Sparse Triangular Solve Performance Trend")

ax.legend()
# ax.grid(axis="y", linestyle="--", alpha=0.5)

# Remove top/right spines (paper-quality)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("sptrsv_gflops_trend.pdf")
plt.show()