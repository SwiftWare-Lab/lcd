from generate_triton_kernel_modular import generate_kernel
import importlib.util

imports = """
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
"""

benchmarking_code = """
def benchmark_unroll_coarsen(A_cf, B):
    # Convert the patterns dict to a list for the GPU
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
                                B.shape[1],
                                num_warps=8,
                                num_stages=3)
    stop.record()

    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms
"""

def transform_patterns(patterns):
    index_dict = {}
    for num, binary_str in patterns.items():
        # Find indices where '1' appears in the string
        indices = [i for i, char in enumerate(binary_str) if char == '1']
        index_dict[num] = indices
    return index_dict

def generate_kernel_code(patterns, unroll_factors, loop_order):
    transformed_patterns = transform_patterns(patterns)

    gen_code = generate_kernel(transformed_patterns, unroll_factors, loop_order)

    code_lines = gen_code.splitlines()
    gen_code = "\n".join(code_lines[2:])

    gen_code = imports + gen_code

    gen_code += benchmarking_code

    return gen_code

def run_generated_kernel(gen_file, A_cf, B):
            
    output_file = "generated_kernel.py"
    with open(output_file, "w") as f:
        f.write(gen_file)

    spec = importlib.util.spec_from_file_location("gen_code", output_file)
    gen_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_code)

    return gen_code.benchmark_unroll_coarsen(A_cf, B)