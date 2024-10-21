import sys
import os

sys.path.insert(1, '../utils')
import numpy as np
import scipy.sparse as sp
import torch
import triton
import triton.language as tl
from cfmtx_utils import preprocess_patterns, create_array_offsets, flatten_array
from scipy.io import mmwrite
from read_compressed_format import read_compressed_matrix


@triton.jit
def unroll_coarsen_kernel(Avp, Aip, Arp, B, C, patterns, Avp_offsets, Aip_offsets, k_rows: tl.constexpr,
                          k_cols: tl.constexpr, N: tl.constexpr):
    row_pi = tl.program_id(axis=0)  # Index of row panel we are working on
    pattern_id = tl.program_id(axis=1)

    # Get the row panel start and end indices
    panel_start = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi)
    panel_end = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi + 1)

    if panel_start == panel_end:
        # row panel for this pattern is empty
        return

    row_idx = row_pi * k_rows  # We are on the row_pi-th row panel, each row panel has k_rows rows
    pattern_ptr = patterns + (k_rows * pattern_id)  # Points to first element of the pattern

    # Count the number of '1' bits in the pattern
    num_ones = 0
    for i in range(k_rows):
        if (tl.load(pattern_ptr + i) == 1):
            num_ones += 1

    pattern_offset = k_cols

    Avp_start_idx = tl.load(
        Avp_offsets + pattern_id)  # Index of the first value in the values array for the entire pattern
    values_ptr = Avp + (Avp_start_idx) + (
            panel_start * num_ones)  # Points to the first value in the values array for this row panel

    num_cols = (panel_end - panel_start)  # Number of total columns in this row panel for this pattern
    num_kernels = num_cols // k_cols  # Number of kernels in this row panel

    Aip_start_idx = tl.load(
        Aip_offsets + pattern_id)  # Index of the first index in the indices array for the entire pattern
    indices_ptr = Aip + Aip_start_idx + panel_start  # Points to the first index in the indices array for this row panel

    # 111
    if pattern_id == 0:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):

                A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j)
                A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j + pattern_offset)
                A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (2 * pattern_offset))

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_2 = tl.zeros((N,), dtype=tl.float32)

                C_0 = A_0 * B_row
                C_1 = A_1 * B_row
                C_2 = A_2 * B_row

                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)
    # 110
    elif pattern_id == 1:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):
                
                A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j)
                A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j + pattern_offset)

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_1 = tl.zeros((N,), dtype=tl.float32)

                C_0 = A_0 * B_row
                C_1 = A_1 * B_row

                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

    # 101
    elif pattern_id == 2:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):
                
                A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j)
                A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (1 * pattern_offset))

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_2 = tl.zeros((N,), dtype=tl.float32)

                C_0 = A_0 * B_row
                C_2 = A_2 * B_row

                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

    # 100
    elif pattern_id == 3:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):
                
                A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j)

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)

                C_0 = A_0 * B_row

                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

    # 011
    elif pattern_id == 4:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):

                A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j)
                A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (pattern_offset))

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_2 = tl.zeros((N,), dtype=tl.float32)

                C_1 = A_1 * B_row
                C_2 = A_2 * B_row

                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

    # 010
    elif pattern_id == 5:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):

                A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j)

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)

                C_1 = A_1 * B_row

                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

    # 001
    elif pattern_id == 6:
        # Loop over every kernel in the row panel
        for i in range(num_kernels):
            # Loop over every column in the kernel
            for j in range(k_cols):

                A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j)

                idx = tl.load(indices_ptr + (i * k_cols) + j)

                # Load a row of B with at row index idx
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)

                C_2 = A_2 * B_row

                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


def benchmark_unroll_coarsen(A_cf, B):
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

    # This is for testing, remove lateer
    # A_ptr = torch.tensor(A_np, dtype=torch.float32, device=device)

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

    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms


def test_unroll_coarsen(A_cf, B):
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

def compare_matrices(A, B):
    # loop over all entries in the matrix
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != B[i, j]:
                print(f"FAILED AT {i}, {j}")
                print(f"FIRST MATRIX VALUE: {A[i, j]}")
                print(f"SECOND MATRIX VALUE: {B[i, j]}")
                return False
    return True

# A small test
def main():

    num_rows = 768
    num_cols = 512
    density = 0.2

    A_sp = sp.random(num_rows, num_cols, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
    A_np = A_sp.toarray()

    mmwrite("A.mtx", A_sp)
    os.system("python3 ../utils/compressed_format_generator.py A.mtx 3 4")
    A_cf = read_compressed_matrix("A.cfmtx")

    B = np.random.randint(1, 11, (512, 32)).astype(np.float32)

    elapsed_time_ms, C = test_unroll_coarsen(A_cf, B)

    C_ref = A_sp @ B

    print(compare_matrices(C, C_ref))
    print(elapsed_time_ms)

if __name__ == '__main__':
    main()

