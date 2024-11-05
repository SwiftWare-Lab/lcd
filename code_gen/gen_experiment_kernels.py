import os
from code_gen_utils import generate_kernel_code

def patterns_from_m(m):
    # Calculate the starting number based on 2^m - 1
    start = 2 ** m - 1
    
    # Create the dictionary for mapping
    bit_string_map = {}
    
    for i in range(start):
        # Map each integer to its binary representation in m bits
        bit_string_map[i] = format(start - i, f'0{m}b')
        
    return bit_string_map

def generate_kernel_file_contents(kernel_m,  unroll_factor, loop_order):

    patterns = patterns_from_m(kernel_m)
    transformed_unroll_factor = {'i': unroll_factor[0], 'j': unroll_factor[1]}
    gen_file_code = generate_kernel_code(patterns, transformed_unroll_factor, loop_order)
    return gen_file_code

def main():

    pattern_sizes = [1, 2, 3, 4, 5, 6]
    unroll_factors = [(1, 1), (1, 2), (1, 4), (2, 1), (4, 1), (1, 8), (1, 16), (8, 1), (16, 1)]
    loop_order = ['i', 'j']

    dir_name = "experiment_kernels/"

    for m in pattern_sizes:
        for unroll_factor in unroll_factors:
                gen_file_code = generate_kernel_file_contents(m, unroll_factor, loop_order)
                file_name = f"pattern_{m}_{unroll_factor[0]}x{unroll_factor[1]}.py"

                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                # Store generated code in the file
                with open(dir_name + file_name, "w") as f:
                    f.write(gen_file_code)


if __name__ == "__main__":
    main()
