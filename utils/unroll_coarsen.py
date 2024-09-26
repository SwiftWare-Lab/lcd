import os
import triton
import triton.language as tl
import read_compressed_format
import math
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite, mmread

#@triton.jit
def unroll_coarsen_kernel(Avp, Aip, Arp, M: tl.constexpr, K: tl.constexpr, k_rows, k_cols, B_ptr, C_ptr, pattern_id, patterns, row_pi, A_matrix):
# Performs spMM using the CompressedMatrix format for A

    row_panel_idx = row_pi


    panel_start = Arp[pattern_id][row_panel_idx]
    try :

        panel_end = Arp[pattern_id][row_panel_idx + 1]
    except:
        print(Arp[pattern_id])
        print(row_panel_idx)

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
    
    rows, cols = 9, 9
    density = 0.5
    #sparse_matrix = sp.random(rows, cols, density=density, format='csr', dtype=np.int32, data_rvs=lambda s: np.random.randint(1, 11, size=s))
    #print()
    #mmwrite("med_matrix.mtx", sparse_matrix)
    #print("Matrix as Array:\n", sparse_matrix.toarray())
    # make os run the command "python3 compressed_format_generator.py med_matrix.mtx 2 3"
    # Read med_matrix.mtx and print it
    sparse_matrix_path = "med_matrix.mtx"
    # Read the matrix using mm
    sparse_matrix = mmread(sparse_matrix_path)
    print(sparse_matrix.toarray())
    os.system("python3 compressed_format_generator.py med_matrix.mtx 4 4")



    path = "med_matrix.cfmtx"
    compressed_matrix = read_compressed_format.read_compressed_matrix(path)



    # loop over every kernel in the compressed matrix
    num_panels = math.ceil(compressed_matrix.num_row / compressed_matrix.kernel_rows)
    for i in range(num_panels):
        for j in range(len(compressed_matrix.patterns)):
            unroll_coarsen_kernel(compressed_matrix.padded_values_data_array, compressed_matrix.indices_array, compressed_matrix.row_panels_pointer, compressed_matrix.num_row, compressed_matrix.num_col, compressed_matrix.kernel_rows, compressed_matrix.kernel_cols, 0, 0, j, compressed_matrix.patterns, i, sparse_matrix.toarray())
    print("Success")
            
if __name__ == "__main__":
    main()

