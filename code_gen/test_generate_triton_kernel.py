import os
import numpy as np
import scipy.sparse as sp
import torch
from triton_kernel import unroll_coarsen_kernel
import sys
from scipy.io import mmread

sys.path.insert(1, '../utils')
from cfmtx_utils import preprocess_patterns, create_array_offsets, flatten_array
from read_compressed_format import read_compressed_matrix

def run_kernel(A_cf, B):
    # Conver the patterns dict to a list for the GPU
    pattern_list = preprocess_patterns(A_cf.patterns)

    # Create the offsets for the Avp and Aip arrays
    Avp_offsets = create_array_offsets(A_cf.padded_values_data_array)
    Aip_offsets = create_array_offsets(A_cf.indices_array)

    # Load everything into the GPU
    device = torch.device('cuda')

    # Flatten and load arrays for A
    A_vp_flattened = flatten_array(A_cf.padded_values_data_array)
    A_vp = torch.tensor(A_vp_flattened, dtype=torch.int32, device=device)
    A_ip_flattened = flatten_array(A_cf.indices_array)
    A_ip = torch.tensor(A_ip_flattened, dtype=torch.int32, device=device)
    A_rp = torch.tensor(A_cf.row_panels_pointer, dtype=torch.int32, device=device)
    patterns_ptr = torch.tensor(pattern_list, dtype=torch.int32, device=device)
    Avp_offsets_ptr = torch.tensor(Avp_offsets, dtype=torch.int32, device=device)
    Aip_offsets_ptr = torch.tensor(Aip_offsets, dtype=torch.int32, device=device)

    # Load B and C arrays
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.zeros((A_cf.num_row, B.shape[1]), dtype=torch.float32, device=device)

    # Launch the kernel
    num_row_panels = len(A_cf.row_panels_pointer[0]) - 1
    grid = (num_row_panels, len(A_cf.patterns))

    # Create CUDA events to record the time
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Launch Triton kernel (or any CUDA operation)
    start.record()

    unroll_coarsen_kernel[grid](A_vp, A_ip, A_rp,
                                B_ptr, C_ptr,
                                patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr,
                                A_cf.kernel_rows, A_cf.kernel_cols,
                                B.shape[1])

    stop.record()

    stop.synchronize()

    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms, C_ptr.cpu().numpy()

# tests for code generated triton kernel

cols = [32, 64, 128, 256, 512]
row_n = 768
densities = np.arange(0.1, 0.8, 0.1)

passed = 0
total = 0

for col in cols:
    for density in densities:
        A_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.mtx"
        B_path = f"../matrices/B_sparse_512x{col}_1.0.mtx"

        # Load A and B as np arrays
        A_sp = mmread(A_path)
        A_sp = A_sp.tocsr()
        B_sp = mmread(B_path)

        # Convert scipy arrays to numpy arrays
        A = A_sp.toarray().astype(np.float32)
        B = B_sp.toarray().astype(np.float32)
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

        os.system(f"python3 ../utils/compressed_format_generator.py {A_path} 3 4 > /dev/null 2>&1")

        A_cf_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.cfmtx"
        A_cf = read_compressed_matrix(A_cf_path)

        C = run_kernel(A_cf, B)

        C_ref = A @ B

        

        if np.allclose(C, C_ref, atol=1e-3):
            print(f"Test passed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx")
            passed += 1

        else:
            print(f"Test failed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx")
            print(f"Expected: {C_ref}")
            print(f"Got: {C}")

    

        total += 1

print(f"Passed {passed}/{total} tests")

