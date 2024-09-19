import numpy
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.io import mmread
import time
import argparse
import os


class CompressedFormat:
    def __init__(self, patterns: object, num_row, num_col, kernel_rows, vector_length) -> object:
        self.padded_values_data_array = [[] for _ in range(len(patterns))]
        self.indices_array = [[] for _ in range(len(patterns))]
        self.patterns = patterns
        self.row_panels_pointer = [[0] for _ in range(len(patterns))]
        self.num_row = num_row
        self.num_col = num_col
        self.kernel_rows = kernel_rows
        self.vector_length = vector_length

    def get_number_of_nonzero_elements(self):
        return sum(element != 0 for row in self.padded_values_data_array for element in row)


def read_smtx_matrix(path):
    with open(path, 'r') as file:
        line = file.readline()
        line = line.replace(',', ' ')
        first_line = line.split()

        rows = int(first_line[0])
        cols = int(first_line[1])
        nnz = int(first_line[2])

        # Initialize CSR arrays
        indptr = np.zeros(rows + 1, dtype=np.int64)
        indices = np.zeros(nnz, dtype=np.int64)
        data = np.ones(nnz, dtype=np.float64)  # Initializing values to 1.0 by default

        # Read Lp
        indptr = np.fromstring(file.readline(), sep=' ', dtype=np.int64)

        # Read Li
        indices = np.fromstring(file.readline(), sep=' ', dtype=np.int64)

    return rows, cols, indptr, indices, data


def transform_row_panel(format_object: CompressedFormat, row_panel_array, row_panel_indices_array, patterns,
                        pack_col_size):
    """
    This function takes the row panel data and transform them in a way that we have packs of data for each pattern
    This packs of data are based on the number of non-zeros in the pattern and the vector length

    steps:
    1. for each pattern, find the number of non-zeros
    2. find the number of columns in the pack (vector length / number of non-zeros) for example when vector length is 32
    and the number of non-zeros is 4, then the number of columns in the pack is 8
    3. find the number of rows in the pack which is the number of non-zeros
    4. pad the row panel to be multiple of the vector length
    5. for each pack, find the data and indices and add them to the format object


    :param format_object:
    :param row_panel_array:
    :param row_panel_indices_array:
    :param patterns:
    :param pack_col_size:
    :return:
    """
    for key, val in patterns.items():
        nnz = np.sum(key)
        pack_row_size = nnz

        vector_length = pack_col_size * nnz

        reminder = len(row_panel_array[val]) % (pack_col_size * pack_row_size)
        # reminder = 0

        if len(row_panel_array[val]) != 0 and reminder != 0:
            row_panel_array[val] = row_panel_array[val] + [0] * (vector_length - reminder)

        number_of_packs = len(row_panel_array[val]) // vector_length

        for i_pack in range(number_of_packs):
            pack = row_panel_array[val][i_pack * vector_length:(i_pack + 1) * vector_length]
            for p_row in range(pack_row_size):
                for p_col in range(pack_col_size):
                    format_object.padded_values_data_array[val].append(pack[p_row + p_col * pack_row_size])

        format_object.indices_array[val].extend(row_panel_indices_array[val])
        reminder_cols = len(row_panel_indices_array[val]) % pack_col_size
        # reminder_cols = 0

        if len(row_panel_indices_array[val]) != 0 and reminder_cols != 0:
            col_max = max(row_panel_indices_array[val])
            format_object.indices_array[val].extend([col_max] * (pack_col_size - reminder_cols))

        format_object.row_panels_pointer[val].append(len(format_object.indices_array[val]))


def csr_to_compressed_format_converter(input_csr_matrix, patterns, row_panel_size=4, p_col_size=4):
    """
    This function finds take csr as input and create the compressed format

    the steps are:
    1. pad the whole matrix to be multiple of the row panel size
    2. for each row panel, find the patterns and collect the values and indices
    3. transform the row panel to data and pad them to create pack of data


    :param input_csr_matrix: the csr matrix to be compressed
    :param patterns: the patterns that should be found in the matrix
    :param row_panel_size: the size of rows of the kernel
    :param p_col_size: the length of the vector, default is 32 which is for GPU and also for CPU AVX-512 it is 16
    :return:
    """

    num_rows, num_cols = input_csr_matrix.shape

    remainder = num_rows % row_panel_size
    if remainder != 0:
        rows_to_add = row_panel_size - remainder
    else:
        rows_to_add = 0

    empty_rows = csr_matrix((rows_to_add, input_csr_matrix.shape[1]))
    input_csr_matrix = vstack([input_csr_matrix, empty_rows])

    indices_all = input_csr_matrix.indices.copy()
    indices_all = np.append(indices_all, num_cols)
    values_all = input_csr_matrix.data.copy()
    values_all = np.append(values_all, 0)
    tp_val = len(indices_all) - 1
    num_rows_new = input_csr_matrix.shape[0]

    format_object = CompressedFormat(patterns, num_rows, num_cols, row_panel_size, p_col_size)

    for start_row in range(0, num_rows, row_panel_size):
        end_row = min(start_row + row_panel_size, num_rows_new)
        row_panel_values_array = [[] for _ in range(len(patterns))]
        row_panel_indices_array = [[] for _ in range(len(patterns))]

        pointer_end = input_csr_matrix.indptr[start_row + 1:end_row + 1].copy()

        pointer = input_csr_matrix.indptr[start_row:end_row].copy()

        while not np.all(pointer_end <= pointer):

            for i in range(len(pointer)):
                if pointer[i] == pointer_end[i]:
                    pointer[i] = tp_val

            indices = indices_all[pointer]
            min_col = indices.min()
            matches = min_col == indices

            val = patterns.get(tuple(matches), -1)
            if val != -1:
                zero_vars = pointer[matches]
                values = values_all[zero_vars]
                row_panel_indices_array[val].append(min_col)
                row_panel_values_array[val].extend(values)
                values_all[zero_vars] = 0

            pointer[matches] += 1
            # print(pointer)

            for i in range(len(pointer)):
                if pointer[i] == pointer_end[i]:
                    pointer[i] = tp_val

        transform_row_panel(format_object, row_panel_values_array, row_panel_indices_array, patterns,
                            p_col_size)

    return format_object


def format_writer(name: str, format: CompressedFormat):
    """
    This function writes the compressed format to a file that we call it cfmtx file

    steps:
    1. write the number of rows, number of columns, number of patterns, and the vector length
    2. write the patterns and their ids
    3. write the row panels pointer, indices, and the padded values for each pattern

    :param name: the name of the file
    :param format: the compressed format object

    """
    with open(name, "w") as file:
        line_p1 = f"{format.num_row} {format.num_col} {format.kernel_rows} {len(format.patterns)} {format.vector_length}\n"
        file.write(line_p1)

        for key, val in format.patterns.items():
            key_str = ''.join(str(int(x)) for x in key)
            line_p2 = f"{key_str} {val} \n"
            file.write(line_p2)

        for i in range(len(format.patterns)):
            line_1_p3 = ' '.join(str(x) for x in format.row_panels_pointer[i])
            line_2_p3 = ' '.join(str(x) for x in format.indices_array[i])
            line_3_p3 = ' '.join(str(x) for x in format.padded_values_data_array[i])
            file.write(f"{line_1_p3}\n")
            file.write(f"{line_2_p3}\n")
            file.write(f"{line_3_p3}\n")


def format_loader(path):
    with open(path, 'r') as file:
        first_line = file.readline().strip()
        num_rows, num_cols, row_kernel_size, num_patterns, vector_length = map(int, first_line.split())

        patterns = {}
        for _ in range(num_patterns):
            line = file.readline().strip().split()
            pattern_name = line[0]
            np_pattern = np.array([x == '1' for x in pattern_name], dtype=bool)
            pattern_id = int(line[1])
            patterns[tuple(np_pattern)] = pattern_id

            format_object = CompressedFormat(patterns, num_rows, num_cols, row_kernel_size, vector_length)
            format_object.row_panels_pointer = []
            format_object.indices_array = []
            format_object.padded_values_data_array = []

        for i in range(num_patterns):
            row_pattern_pointer = list(map(int, file.readline().strip().split()))
            pattern_indices = list(map(int, file.readline().strip().split()))
            pattern_values = list(map(float, file.readline().strip().split()))

            format_object.row_panels_pointer.append(row_pattern_pointer)
            format_object.indices_array.append(pattern_indices)
            format_object.padded_values_data_array.append(pattern_values)

        return format_object


def format_converter_to_coo(format_object: CompressedFormat):
    """
    This function converts the compressed format to COO format

    steps:
    1. for each pattern, find the indices and values
    2. for each pack, find the indices and values
    3. add the indices and values to the COO format

    :param format_object: the compressed format object
    :return: the COO format
    """
    rows = []
    cols = []
    data = []

    for key, val in format_object.patterns.items():

        nnz = np.sum(np.array(key))
        nnz_index = np.where(np.array(key) == True)

        nnz_counter = 0
        for j in range(len(format_object.row_panels_pointer[val]) - 1):

            start = format_object.row_panels_pointer[val][j]
            end = format_object.row_panels_pointer[val][j + 1]

            step_size = cf.vector_length // nnz
            pattern_vector_length = step_size * nnz
            for k in range(start, end, int(step_size)):
                for i in range(pattern_vector_length):
                    if format_object.padded_values_data_array[val][nnz_counter + i] != 0:
                        rows.append(j * cf.kernel_rows + nnz_index[0][i // step_size])
                        cols.append(format_object.indices_array[val][k + i % step_size])
                        data.append(format_object.padded_values_data_array[val][nnz_counter + i])
                nnz_counter += pattern_vector_length

    return rows, cols, data


def sort_coo(row, col, data):
    # Combine the arrays into a list of tuples
    combined = list(zip(row, col, data))

    # Sort the combined list first by array1, then by array2
    sorted_combined = sorted(combined, key=lambda x: (x[0], x[1]))

    # Unzip the sorted combined list back into individual arrays
    sorted_array1, sorted_array2, sorted_array3 = zip(*sorted_combined)

    # Convert the sorted arrays back to lists (optional)
    sorted_row = list(sorted_array1)
    sorted_col = list(sorted_array2)
    sorted_data = list(sorted_array3)

    return sorted_row, sorted_col, sorted_data


def mtx_to_cfmtx_convertor(path, kernel_m, kernel_n):
    patterns = {}
    for i in range(1, 2 ** (kernel_m * 1)):
        x = 2 ** (kernel_m * 1) - i
        arr = np.array([True if x == '1' else False for x in bin(x)[2:].zfill(kernel_m * 1)])
        patterns[tuple(arr)] = i - 1

    sparse_matrix = mmread(path)
    csr = sparse_matrix.tocsr()

    cf = csr_to_compressed_format_converter(csr, patterns, kernel_m, kernel_n)
    format_writer(path[:-4] + ".cfmtx", cf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that demonstrates argument parsing.")

    # Add arguments
    parser.add_argument('path', type=str, default="")
    parser.add_argument("kernel_m", type=int, default=4)
    parser.add_argument("kernel_n", type=int, default=4)
    # Parse the arguments
    args = parser.parse_args()

    if args.path == "":
        raise ValueError("Please provide the path to the matrix file")

    kernel_n = args.kernel_n

    path = args.path

    kernel_m = args.kernel_m

    if path.endswith(".mtx"):

        print("start converting")
        mtx_to_cfmtx_convertor(path, kernel_m, kernel_n)
        print("done converting")

    else:
        print("this format is not supported")
