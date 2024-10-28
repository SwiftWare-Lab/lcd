#!/bin/bash
#SBATCH --job-name=generate_and_run  # Name of your job
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory per node
#SBATCH --time=00:00:50              # Time limit (hh:mm:ss)
#SBATCH --output=generate_and_run_%j.out  # Standard output and error log

# Load necessary modules
module load gcc/12.3 cuda/12.0
source ~/triton-env/bin/activate  # Activate your virtual environment

# Step 1: Run the code generator
python3 generate_triton_kernel_modular.py > out.py

# Step 2: Remove the first two lines of out.py and replace with the desired imports
sed -i '1,2d' out.py

# Add the required imports to the beginning of out.py
cat <<EOL | cat - out.py > temp && mv temp out.py
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
EOL

# Step 3: Append functions and main method to out.py
cat <<EOL >> out.py

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
                                B.shape[1])
    stop.record()

    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms

def test_unroll_coarsen(A_cf, B):
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
                                B.shape[1])

    stop.record()
    torch.cuda.synchronize()

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
    os.system("python3 ../utils/compressed_format_generator.py A.mtx 2 4")
    A_cf = read_compressed_matrix("A.cfmtx")

    B = np.random.randint(1, 11, (512, 32)).astype(np.float32)

    elapsed_time_ms, C = test_unroll_coarsen(A_cf, B)

    C_ref = A_sp @ B

    print(compare_matrices(C, C_ref))
    print(elapsed_time_ms)

if __name__ == '__main__':
    main()
EOL

# Step 4: Run the modified out.py script
python3 out.py