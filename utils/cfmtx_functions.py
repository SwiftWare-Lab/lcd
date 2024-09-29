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
    #num_cols = len(values) // num_ones # We are only containing the non-zero values (even the padded ones) and there are num_cols of them
    num_cols = panel_end - panel_start
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