import sys
sys.path.insert(1, '../utils')
import torch
import triton
import triton.language as tl
from cfmtx_utils import preprocess_patterns, create_array_offsets, flatten_array

@triton.jit
def unroll_coarsen_kernel(Avp, Aip, Arp, B, C, k_rows: tl.constexpr, k_cols, patterns, Avp_offsets, Aip_offsets, N: tl.constexpr):
    row_pi = tl.program_id(axis=0)
    pattern_id = tl.program_id(axis=1)

    tl.device_print("pattern_id:", pattern_id)

    # Get the row panel start and end indices
    panel_start = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi)
    panel_end = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi + 1)

    if panel_start == panel_end:
        # row panel for this pattern is empty
        return
    
    row_idx = row_pi * k_rows # We are on the row_pi-th row panel, each row panel has k_rows rows
    pattern_ptr = patterns + (k_rows * pattern_id)  # Points to first element of the pattern

    # Count the number of '1' bits in the pattern
    num_ones = 0
    for i in range(k_rows):
        if(tl.load(pattern_ptr + i) == 1):
            num_ones += 1
    
    pattern_offset = k_cols # Used to move from one bit in the pattern to the next in the values array

    Avp_start_idx = tl.load(Avp_offsets + pattern_id) # Index of the first value in the values array for the entire pattern
    values_ptr = Avp + (Avp_start_idx) + (panel_start * num_ones) # Points to the first value in the values array for this row panel

    num_cols = (panel_end - panel_start) # Number of total columns in this row panel for this pattern
    num_kernels = num_cols // k_cols # Number of kernels in this row panel

    Aip_start_idx = tl.load(Aip_offsets + pattern_id) # Index of the first index in the indices array for the entire pattern
    indices_ptr = Aip + Aip_start_idx + panel_start # Points to the first index in the indices array for this row panel

    # Loop over every kernel in the row panel
    for i in range(num_kernels):
        # Loop over every column in the kernel
        for j in range(k_cols):
            # Some columns are just columns of zeroes, we want to skip those
            first_value = tl.load(values_ptr + (num_ones * k_cols * i) + j)
            tl.cast(first_value, dtype=tl.float32)
            if (first_value != 0.0):

                # This part only runs if a column of non-zero values is found
                ones_count = 0  # Used to keep track of the '1' bits we've processed

                # Loop over every bit in the pattern, there are k_rows bits
                for k in range(k_rows):
                    # Only process the non-zero values
                    if tl.load(pattern_ptr + k) == 1:

                        # Process the value we are at
                        A_value = tl.load(values_ptr + (num_ones * k_cols * i) + j + (ones_count * pattern_offset))

                        idx = tl.load(indices_ptr + (i * k_cols) + j) # The column index for this value

                        # Load a row of B with at row index idx
                        B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                        C_acc = tl.zeros((N,), dtype=tl.float32)
                        C_acc = A_value * B_row

                        C_old = tl.load(C + (row_idx + k) * N + tl.arange(0, N))
                        tl.store(C + ((row_idx + k) * N) + tl.arange(0, N), C_acc + C_old)        
                    
                        ones_count += 1 # We've proccessed one '1' bit, move onto the next one
    


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

    # Create CUDA events to record the time
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Launch Triton kernel (or any CUDA operation)
    start.record()

    unroll_coarsen_kernel[grid](A_vp, A_ip, A_rp,
                                B_ptr, C_ptr,
                                A_cf.kernel_rows, A_cf.kernel_cols, 
                                patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr,
                                B.shape[1])
    
    stop.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms