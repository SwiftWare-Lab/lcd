import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

def geomean(arr):
    return np.exp(np.mean(np.log(arr)))

log_file = sys.argv[1]
df_results = pd.read_csv(log_file, skipinitialspace=True)
print(df_results.keys())
impls = df_results['implementation'].unique()
matrix_list = df_results['matrix'].unique()
# nnz_list = np.array([df_results[df_results['matrix'] == mat]['nnz'].unique()[0] for mat in matrix_list])
timings = {impl: df_results[df_results['implementation'] == impl]['median_time'] for impl in impls}
print(len(timings['spmm_softmax_unfused']))
# print(nnz_list)
# gflops = {impl: 2 * nnz_list * 128 / timings[impl] / 1e9 for impl in impls}
# print(gflops['spmm_softmax_fused_v2'])
# print(gflops['spmm_softmax_unfused'])
# fig, ax = plt.subplots(figsize=(15, 8))
plt.plot(np.array(timings['spmm_softmax_fused_v2']))
plt.plot(np.array(timings['spmm_softmax_fused_v1']))
plt.plot(np.array(timings['spmm_softmax_unfused']))
print(geomean(np.array(timings['spmm_softmax_unfused']) / np.array(timings['spmm_softmax_fused_v2'])))
print(geomean(np.array(timings['spmm_softmax_unfused']) / np.array(timings['spmm_softmax_fused_v1'])))
# plt.axhline(y=1, color='r', linestyle='--', linewidth=2)
plt.ylabel("Time (s)")
# plt.xlabel("Unfused Gflops")
# plt.xscale('log')
plt.show()

print(np.average(np.array(timings['sddmm_spmm_unfused']) / np.array(timings['sddmm_spmm_fused_v2'])))

# print(timings)