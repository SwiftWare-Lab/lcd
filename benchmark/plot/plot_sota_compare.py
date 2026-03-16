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

def best_of_prefixes(dataframe, prefixes):
    cols = [c for c in dataframe.columns if any(c.startswith(p) for p in prefixes)]
    return dataframe[cols].max(axis=1)

yy_values = df["YY"]
ag_values = df["AG-Sptrsv"]

# CSC
csc_wf_base = df["csc-wf-base"]
csc_sf_base = df["csc-sf-base"]
csc_best_opt = best_of_prefixes(df, ["csc-wf", "csc-sf", "csc-h"])

# CSR
csr_wf_base = df["csr-wf-base"]
csr_sf_base = df["csr-sf-base"]
csr_best_opt = best_of_prefixes(df, ["csr-wf", "csr-sf", "csr-h"])


def make_plot(x, y1, y2, y3, title, out_file):

    plt.figure(figsize=(14,6))

    plt.plot(x, y1, label="YY", color="maroon", linewidth=2, marker='o')
    plt.plot(x, y2, label="AG", color="black", linewidth=2, marker='o')
    plt.plot(x, y3, label="LCD-Optimizied", color="red", linewidth=2, marker='o')
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
    yy_values,
    ag_values,
    csr_best_opt,
    "CSC SpTRSV Throughput",
    "csr_plot_sota.png"
)