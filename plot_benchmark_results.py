import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

log_file = sys.argv[1]
df_results = pd.read_csv(log_file, skipinitialspace=True)
print(df_results.keys())
impls = df_results['implementation'].unique()
matrix_list = df_results['matrix'].unique()
nnz_list = np.array([df_results[df_results['matrix'] == mat]['nnz'].unique()[0] for mat in matrix_list])
timings = {impl: df_results[df_results['implementation'] == impl]['median_time'] for impl in impls}
print(len(timings['sddmm_spmm_unfused']))
# print(nnz_list)
gflops = {impl: 2 * nnz_list * 128 / timings[impl] / 1e9 for impl in impls}
print(gflops['sddmm_spmm_fused_v2'])
print(gflops['sddmm_spmm_unfused'])
# fig, ax = plt.subplots(figsize=(15, 8))
plt.scatter(np.array(gflops['sddmm_spmm_unfused']), np.array(gflops['sddmm_spmm_fused_v2']) / np.array(gflops['sddmm_spmm_unfused']))
plt.axhline(y=1, color='r', linestyle='--', linewidth=2)
plt.ylabel("Speed Up over Unfused")
plt.xlabel("Unfused Gflops")
# plt.xscale('log')
plt
plt.show()

print(np.average(np.array(timings['sddmm_spmm_unfused']) / np.array(timings['sddmm_spmm_fused_v2'])))

# print(timings)