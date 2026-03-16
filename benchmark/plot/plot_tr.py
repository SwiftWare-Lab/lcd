import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = "~/Downloads/sptrsv-raw-updated(in).csv"

df = pd.read_csv(csv_file)
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

for col in df.columns:
    if col != "matrices":
        df[col] = pd.to_numeric(df[col], errors="coerce")

x = np.arange(len(df))

def optimized_of_prefixes(dataframe, prefixes):
    cols = [c for c in dataframe.columns if any(c.startswith(p) for p in prefixes)]
    return dataframe[cols].max(axis=1)

cusparse_vals = df["cusparse-new"]

# # CSC
# csc_wf_base = df["csc-wf-base"]
# csc_sf_base = df["csc-sf-base"]
tr_csc_optimized_opt = optimized_of_prefixes(df, ["tr_csc"])
#
# # CSR
# csr_wf_base = df["csr-wf-base"]
# csr_sf_base = df["csr-sf-base"]
# csr_optimized_opt = optimized_of_prefixes(df, ["csr-wf", "csr-sf", "csr-h"])
tr_csr_optimized_opt = optimized_of_prefixes(df, ["tr_csr"])



def make_plot(x, y1, y2, title, out_file):

    plt.figure(figsize=(14,6))

    plt.plot(x, y1, label="[Baseline] cuSPARSE", color="gold", linewidth=2, marker='o')
    plt.plot(x, y2, label="LCD-Optimized TR", color="red", linewidth=2, marker='o')
    # plt.plot(x, y3, label="[Baseline] Barrier-free", color="purple", linewidth=2, marker='o')
    # plt.plot(x, y4, label="LCD-Optimized", color="red", linewidth=2, marker='o')

    plt.xlabel("Matrix ID", fontsize=15, fontweight='bold')
    plt.ylabel("Throughput (GFLOP/s)", fontsize=15, fontweight='bold')
    # plt.title(title)

    # Required tick settings
    plt.xticks(np.arange(0, 121, 20))
    plt.yticks(np.arange(0, 11, 2))

    # plt.xlim(0, 1)
    # plt.ylim(0, 10)

    # plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.show()


make_plot(
    x,
    cusparse_vals,
    tr_csc_optimized_opt,
    "CSC SpTRSV Throughput",
    "csc_plot_TR.png"
)

make_plot(
    x,
    cusparse_vals,
    tr_csr_optimized_opt,
    "CSR SpTRSV Throughput",
    "csr_plot_TR.png"
)