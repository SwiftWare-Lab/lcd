import pandas as pd
import sys
import numpy as np

log_file = sys.argv[1]
df_results = pd.read_csv(log_file, skipinitialspace=True)
print(df_results.keys())
impls = df_results['implementation'].unique()
matrix_list = df_results['matrix'].unique()
timings = {impl: df_results[df_results['implementation'] == impl]['median_time'] for impl in impls}

print(np.average(np.array(timings['sddmm_spmm_unfused']) / np.array(timings['sddmm_spmm_fused_v1'])))

# print(timings)