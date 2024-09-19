class CompressedMatrix:
    def __init__(self, patterns, num_row, num_col, kernel_rows, kernel_cols):
        self.padded_values_data_array = [[] for _ in range(len(patterns))]
        self.indices_array = [[] for _ in range(len(patterns))]
        self.patterns = patterns
        self.row_panels_pointer = [[] for _ in range(len(patterns))]
        self.num_row = num_row
        self.num_col = num_col
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols

    def __repr__(self):
        return (f"CompressedMatrix(num_row={self.num_row}, num_col={self.num_col}, "
                f"kernel_rows={self.kernel_rows}, kernel_cols={self.kernel_cols}, "
                f"patterns={self.patterns})")


def read_compressed_matrix(file_path):
    """
    Reads matrix data from a file and populates a CompressedMatrix object.

    The file should have the following format:
    - The first line contains five integers: number of rows, number of columns, kernel rows, number of patterns, kernel columns.
    - The subsequent lines contain data for each pattern:
      - Each pattern starts with an integer (pattern ID).
      - Followed by three lines:
        1. A space-separated list of integers representing the row panel pointers.
        2. A space-separated list of integers representing the indices.
        3. A space-separated list of floating-point numbers representing the values.

    Parameters:
    file_path (str): Path to the input file containing the matrix data.

    Returns:
    CompressedMatrix: A populated CompressedMatrix object with patterns, row panel pointers, indices, and values.

    Example:
        matrix = read_compressed_matrix("sample_matrix.txt")
        print(matrix)
    """
    with open(file_path, 'r') as file:
        # Read the first line: num_row, num_col, kernel_rows, num_patterns, kernel_cols
        first_line = file.readline().strip().split()
        num_row, num_col, kernel_rows, num_patterns, kernel_cols = map(int, first_line)

        # Read patterns (pattern id for each pattern)
        patterns = {}
        for _ in range(num_patterns):
            pattern_bin, pattern_id = file.readline().strip().split()
            patterns[int(pattern_id)] = pattern_bin

        # Create the CompressedMatrix instance
        matrix = CompressedMatrix(patterns, num_row, num_col, kernel_rows, kernel_cols)

        # Read row panel pointers, indices, and values for each pattern
        for i in range(num_patterns):
            # Read row panel pointers
            row_panels_pointer = list(map(int, file.readline().strip().split()))
            matrix.row_panels_pointer[i] = row_panels_pointer

            # Read indices
            indices = list(map(int, file.readline().strip().split()))
            matrix.indices_array[i] = indices

            # Read values
            values = list(map(float, file.readline().strip().split()))
            matrix.padded_values_data_array[i] = values

    return matrix
