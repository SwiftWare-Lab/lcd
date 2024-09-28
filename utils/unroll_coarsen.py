import os
import torch
import triton
import triton.language as tl
import read_compressed_format
import math
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite, mmread

@triton.jit
def unroll_coarsen_kernel(Avp, Aip, Arp, B, C, k_rows: tl.constexpr, k_cols, patterns, Avp_offsets, Aip_offsets, N: tl.constexpr, A_ptr, K: tl.constexpr, C_ref):
    row_pi = tl.program_id(0)
    pattern_id = tl.program_id(1)

    panel_start = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi)
    panel_end = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi + 1)

    if panel_start == panel_end:
        # row panel for this pattern is empty
        return
    
    row_idx = row_pi * k_rows # We are on the row_pi-th row panel, each row panel has k_rows rows
    pattern_ptr = patterns + (k_rows * pattern_id)  # Points to first element of the pattern
    #num_ones = tl.sum(pattern == 1)
    num_ones = 0
    for i in range(k_rows):
        if(tl.load(pattern_ptr + i) == 1):
            num_ones += 1
    
    # cast num_ones to tl.int1
    #num_ones = tl.cast(num_ones, dtype=tl.int32)
    pattern_offset = k_cols
    Avp_start_idx = tl.load(Avp_offsets + pattern_id)
    Avp_end_idx = tl.load(Avp_offsets + pattern_id + 1)

    values_ptr = Avp + (Avp_start_idx) + (panel_start * num_ones)
    #values = tl.load(Avp + Avp_start_idx + tl.arange(0, MAXIMUM_VALUES_SIZE))

    num_cols = (panel_end - panel_start)
    num_kernels = num_cols // k_cols

    idx_count = 0

    Aip_start_idx = tl.load(Aip_offsets + pattern_id)
    #Aip_end_idx = tl.load(Aip_offsets + pattern_id + 1)

    indices_ptr = Aip + Aip_start_idx + panel_start
    #indices = tl.load(Aip + Aip_start_idx + tl.arange(0, MAXIMUM_VALUES_SIZE))

    # Loop over every kernel in the row panel
    for i in range(num_kernels):
        # Loop over every column in the kernel
        for j in range(k_cols):
            # Some columns are just columns of zeroes, we want to skip those
            if (tl.load(values_ptr + (num_ones * k_cols * i) + j) != 0.0):

                # This part only runs if a column of non-zero values is found
                ones_count = 0
                # Loop over every bit in the pattern, there are k_rows bits
                for k in range(k_rows):
                    if tl.load(pattern_ptr + k) == 1:
                        # Process the value we are at
                        A_value = tl.load(values_ptr + (num_ones * k_cols * i) + j + (ones_count * pattern_offset))

                        idx = tl.load(Aip + Aip_start_idx + idx_count)

                        # convert A_value to type float
                        A_value = tl.cast(A_value, dtype=tl.float32)
                        A_cmp = tl.load(A_ptr + K * (row_idx + k) + idx)

                        # Get the row of B with row index idx
                        b_row = tl.load(B + idx * N + tl.arange(0, N))

                        C_row_idx = (row_idx + k) * N + tl.arange(0, N)
                        C_val = A_value * b_row
                        C_old = tl.load(C + C_row_idx)
                        tl.store(C + C_row_idx, C_val + C_old)  # Use tl.store to write result into C
                    
                        ones_count += 1 # We've proccessed one '1' bit, move onto the next one
                
                idx_count += 1 # We've successfully processed a column, move onto the next one
    
def preprocess_patterns(patterns):

    pattern_list = []
    for i in range(len(patterns)):
        pattern = [int(x) for x in patterns[i]]
        pattern_list.append(pattern)

    return pattern_list

def create_array_offsets(array):
    
    offsets = np.zeros(len(array) + 1, dtype=np.int32)
    for i in range(len(array)):
        offsets[i + 1] = offsets[i] + len(array[i])

    return offsets

def next_power_of_two(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

def flatten_array(array):
    flattened = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            flattened.append(array[i][j])

    return flattened

def launch_unroll_coarsen(A_cf, B, A):

    pattern_list = preprocess_patterns(A_cf.patterns)
    Avp_offsets = create_array_offsets(A_cf.padded_values_data_array)
    Aip_offsets = create_array_offsets(A_cf.indices_array)

    # Load everything into the GPU
    device = torch.device('cuda')
    A_vp_flattened = flatten_array(A_cf.padded_values_data_array)
    A_vp = torch.tensor(A_vp_flattened, dtype=torch.float32, device=device)
    A_ip_flattened = flatten_array(A_cf.indices_array)
    A_ip = torch.tensor(A_ip_flattened, dtype=torch.int32, device=device)
    A_rp = torch.tensor(A_cf.row_panels_pointer, dtype=torch.int32, device=device)
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.zeros((A_cf.num_row, B.shape[1]), dtype=torch.float32, device=device)
    A_ptr = torch.tensor(A, dtype=torch.float32, device=device)
    patterns_ptr = torch.tensor(pattern_list, dtype=torch.int32, device=device)
    Avp_offsets_ptr = torch.tensor(Avp_offsets, dtype=torch.int32, device=device)
    Aip_offsets_ptr = torch.tensor(Aip_offsets, dtype=torch.int32, device=device)

    num_row_panels = len(A_cf.row_panels_pointer[0]) - 1
    grid = (num_row_panels, len(A_cf.patterns))

    C_ref = A @ B
    C_cmp = torch.tensor(C_ref, dtype=torch.float32, device=device)

    unroll_coarsen_kernel[grid](A_vp, A_ip, A_rp, B_ptr, C_ptr, A_cf.kernel_rows, A_cf.kernel_cols, patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr, B.shape[1], A_ptr, A.shape[1], C_cmp)

    

    # Sync the device
    torch.cuda.synchronize()
    print(C_ptr.cpu().numpy())
    print(C_ref)
    # Compare A and A_ptr
    print(np.allclose(C_ref, C_ptr.cpu().numpy(), atol=1e-6))
    return C_ptr.cpu().numpy()


def cfmtx_spMM(Avp, Aip, Arp, B, C, k_rows, k_cols, pattern_id, patterns, row_pi):

    panel_start = Arp[pattern_id][row_pi]
    panel_end = Arp[pattern_id][row_pi + 1]

    if panel_start == panel_end:
        # row panel for this pattern is empty
        return
    
    row_idx = row_pi * k_rows # We are on the row_idx-th row panel, each row panel has k_rows rows

    pattern = patterns[pattern_id]
    num_ones = pattern.count('1')
    pattern_offset = k_cols # Go from one value in the pattern to the next, we pack k_cols values together so offset is k_cols
    values = Avp[pattern_id][panel_start * num_ones: panel_end * num_ones]

    num_cols = len(values) // num_ones # We are only containing the non-zero values (even the padded ones) and there are num_cols of them
    num_kernels = num_cols// k_cols

    idx_count = 0

    indices = Aip[pattern_id][panel_start: panel_end]

    # Loop over every kernel in the row panel
    for i in range(num_kernels):
        # Loop over every column in the kernel
        for j in range(k_cols):
            # Some columns are just columns of zeroes, we want to skip those
            if(values[(num_ones * k_cols * i) + j] != 0.0):
                # This part only runs if a column of non-zero values is found
                ones_count = 0
                # Loop over every bit in the pattern, there are k_rows bits
                for k in range(k_rows):
                    if pattern[k] == '1':
                        # Process the value we are at
                        A_value = values[(num_ones * k_cols * i) + (j) + (ones_count * pattern_offset)]
                        # A_value is at row row_idx + k, col indices[idx_count]
                        # Loop over the columns of B
                        b_row = B[indices[idx_count], :]
                        # Multiply and accumulate into C
                        for l in range(len(b_row)):
                            C[row_idx + k, l] += A_value * b_row[l]
                    
                        ones_count += 1 # We've proccessed one '1' bit, move onto the next one
                
                idx_count += 1 # We've successfully processed a column, move onto the next one

def read_cfmtx_matrix(Avp, Aip, Arp, k_rows, k_cols, pattern_id, patterns, row_pi, A_matrix):
# Performs spMM using the CompressedMatrix format for A

    row_panel_idx = row_pi

    panel_start = Arp[pattern_id][row_panel_idx]
    panel_end = Arp[pattern_id][row_panel_idx + 1]


    if panel_start == panel_end:
        # row panel for this pattern is empty
        return
    
    row_idx = row_panel_idx * k_rows # Each row panel has k_rows rows

    pattern = patterns[pattern_id]
    num_ones = pattern.count('1')
    pattern_offset = k_cols # Go from one value in the pattern to the next, we pack k_cols values together so offset is k_cols
    values = Avp[pattern_id][panel_start * num_ones: panel_end * num_ones]
    #print(f"Values: {values}")
    num_cols = len(values) // num_ones # We are only containing the non-zero values (even the padded ones) and there are num_cols of them
    num_kernels = num_cols// k_cols

    idx_count = 0
    indices = Aip[pattern_id][panel_start: panel_end]
    # Loop over every kernel in the row panel
    for i in range(num_kernels):
        # Loop over every column in the kernel
        for j in range(k_cols):
            # Some columns are just columns of zeroes, we want to skip those
            if(values[(num_ones * k_cols * i) + j] != 0.0):

                # This part only runs if a column of non-zero values is found
                ones_count = 0
                # Loop over every bit in the pattern, there are k_rows bits
                for k in range(k_rows):
                    if pattern[k] == '1':
                        # Process the value we are at
                        A_value = values[(num_ones * k_cols * i) + (j) + (ones_count * pattern_offset)] # We are in i-th kernel, j-th column, ones_count row
                        #print(f"{A_value} is at row {row_idx + k} and col {indices[idx_count]}")
                        if A_matrix[row_idx + k][indices[idx_count]] != A_value:
                            print("Error")
                        ones_count += 1 # We've proccessed one '1' bit, move onto the next one

                idx_count += 1 # We've successfully processed a column, move onto the next one
    
# main function
def main():
    
    rows, cols = 5, 5
    density = 0.5
    sparse_matrix = sp.random(rows, cols, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
    mmwrite("med_matrix.mtx", sparse_matrix)
    os.system("python3 compressed_format_generator.py med_matrix.mtx 2 2")
    # make os run the command "python3 compressed_format_generator.py med_matrix.mtx 2 3"
    # Read med_matrix.mtx and print it
    #sparse_matrix_path = "med_matrix.mtx"
    path = "med_matrix.cfmtx"
    compressed_matrix = read_compressed_format.read_compressed_matrix(path)
    # Read the matrix using mm
    #sparse_matrix = mmread(sparse_matrix_path)
    

    # Create a dense matrix that has 9 rows
    B = np.random.rand(rows, cols).astype(np.float32)
    # Create a matrix that is 9 x 9 and zero'd out
    C = np.zeros((rows, cols), dtype=np.float32)

    C_ref = sparse_matrix.toarray() @ B
    Avp = compressed_matrix.padded_values_data_array
    Aip = compressed_matrix.indices_array
    Arp = compressed_matrix.row_panels_pointer
    k_rows = compressed_matrix.kernel_rows
    k_cols = compressed_matrix.kernel_cols
    patterns = compressed_matrix.patterns
    num_panels = math.ceil(compressed_matrix.num_row / compressed_matrix.kernel_rows)
    for i in range(len(compressed_matrix.patterns)):
        for j in range(num_panels):
            cfmtx_spMM(Avp, Aip, Arp, B, C, k_rows, k_cols, i, patterns, j)

    #print(C)
    #print(C_ref)
    # Compare C and C_ref
    print(np.allclose(C, C_ref, atol=1e-6))

    


    """
    # loop over every kernel in the compressed matrix
    num_panels = math.ceil(compressed_matrix.num_row / compressed_matrix.kernel_rows)
    for i in range(num_panels):
        for j in range(len(compressed_matrix.patterns)):
            read_cfmtx_matrix(compressed_matrix.padded_values_data_array, compressed_matrix.indices_array, compressed_matrix.row_panels_pointer, compressed_matrix.kernel_rows, compressed_matrix.kernel_cols, j, compressed_matrix.patterns, i, sparse_matrix.toarray())
    print("Success")
    """
def test():

    rows, cols = 4, 4
    density = 1.0
    sparse_matrix = sp.random(rows, cols, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
    mmwrite("med_matrix.mtx", sparse_matrix)
    os.system("python3 compressed_format_generator.py med_matrix.mtx 2 2")
    A = mmread("med_matrix.mtx")
    path = "med_matrix.cfmtx"
    compressed_matrix = read_compressed_format.read_compressed_matrix(path)
    rows = cols = compressed_matrix.num_row


    B = np.random.rand(rows, cols).astype(np.float32)

    C = launch_unroll_coarsen(compressed_matrix, B, A.toarray())

    A = mmread("med_matrix.mtx")
    C_ref = A.toarray() @ B
    #print(C)
    #print(C_ref)
    #print(np.allclose(C, C_ref, atol=1e-6))



if __name__ == "__main__":
    test()

