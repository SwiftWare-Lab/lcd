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

cusparse_vals = df["cusparse-new"]
cusparse_gmean = cusparse_vals.prod() ** (1.0 / len(cusparse_vals))
print(f"cuSPARSE Geometric Mean: {cusparse_gmean:.2f} GFLOP/s")


# CSC
csc_wf_base = df["csc-wf-base"]
csc_sf_base = df["csc-sf-base"]
csc_best_opt = best_of_prefixes(df, ["csc-wf", "csc-sf", "csc-h"])
csc_best_gmean = csc_best_opt.prod() ** (1.0 / len(csc_best_opt))
print(f"Best CSC Geometric Mean: {csc_best_gmean:.2f} GFLOP/s")

# CSR
csr_wf_base = df["csr-wf-base"]
csr_sf_base = df["csr-sf-base"]
csr_best_opt = best_of_prefixes(df, ["csr-wf", "csr-sf", "csr-h"])
csr_best_gmean = csr_best_opt.prod() ** (1.0 / len(csr_best_opt))
print(f"Best CSR Geometric Mean: {csr_best_gmean:.2f} GFLOP/s")

print("CSC Speedup over cuSPARSE (Geometric Mean): {:.2f}x".format(csc_best_gmean / cusparse_gmean))
print("CSR Speedup over cuSPARSE (Geometric Mean): {:.2f}x".format(csr_best_gmean / cusparse_gmean))

cusparse_avg = cusparse_vals.mean()
csc_wf_avg = csc_best_opt.mean()
csr_wf_avg = csr_best_opt.mean()
print(f"cuSPARSE Average: {cusparse_avg:.2f} GFLOP/s")
print(f"Best CSC Average: {csc_wf_avg:.2f} GFLOP/s")
print(f"Best CSR Average: {csr_wf_avg:.2f} GFLOP/s")
print("CSC Speedup over cuSPARSE (Average): {:.2f}x".format(csc_wf_avg / cusparse_avg))

cusparse_median = cusparse_vals.median()
csc_wf_median = csc_best_opt.median()
csr_wf_median = csr_best_opt.median()
print(f"cuSPARSE Median: {cusparse_median:.2f} GFLOP/s")
print(f"Best CSC Median: {csc_wf_median:.2f} GFLOP/s")
print(f"Best CSR Median: {csr_wf_median:.2f} GFLOP/s")


csc_speedups = csc_best_opt / cusparse_vals
csr_speedups = csr_best_opt / cusparse_vals

gmean_speedup_csc = csc_speedups.prod() ** (1.0 / len(csc_speedups))
gmean_speedup_csr = csr_speedups.prod() ** (1.0 / len(csr_speedups))
print(f"Geometric Mean Speedup of Best CSC over cuSPARSE: {gmean_speedup_csc:.2f}x")
print(f"Geometric Mean Speedup of Best CSR over cuSPARSE: {gmean_speedup_csr:.2f}x")

csc_tp_1 = csc_best_opt > 1.0
csr_tp_1 = csr_best_opt > 1.0



def make_plot(x, y1, y2, y3, y4, title, out_file):

    plt.figure(figsize=(14,6))

    plt.plot(x, y1, label="[Baseline] cuSPARSE", color="gold", linewidth=2, marker='o')
    plt.plot(x, y2, label="[Baseline] Wavefront", color="green", linewidth=2, marker='o')
    plt.plot(x, y3, label="[Baseline] Barrier-free", color="purple", linewidth=2, marker='o')
    plt.plot(x, y4, label="LCD-Optimized", color="red", linewidth=2, marker='o')

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
    csc_wf_base,
    csc_sf_base,
    csc_best_opt,
    "CSC SpTRSV Throughput",
    "csc_plot_updated_cusparse.png"
)

make_plot(
    x,
    cusparse_vals,
    csr_wf_base,
    csr_sf_base,
    csr_best_opt,
    "CSR SpTRSV Throughput",
    "csr_plot_updated_cusparse.png"
)