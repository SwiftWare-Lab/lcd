import read_compressed_format

path = "med_matrix.cfmtx"
compressed_matrix = read_compressed_format.read_compressed_matrix(path)

print(compressed_matrix.padded_values_data_array)
print(compressed_matrix.indices_array)
print(compressed_matrix.patterns)
print(compressed_matrix.row_panels_pointer)
print(compressed_matrix.num_row)
print(compressed_matrix.num_col)
print(compressed_matrix.kernel_rows)
print(compressed_matrix.kernel_cols)

        
