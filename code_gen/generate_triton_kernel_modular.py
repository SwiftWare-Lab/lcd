class TritonAccessFunction:
    def __init__(self, name, base_ptr, offset_expr, load_type='tl.load'):
        self.name = name
        self.base_ptr = base_ptr
        self.offset_expr = offset_expr
        self.load_type = load_type

    def get_load_code(self, indent):
        # Generates code for loading a value with an offset
        return f"{indent}{self.name} = {self.load_type}({self.base_ptr} + ({self.offset_expr}))\n"

class TritonCodelet:
    def __init__(self, name, iteration_space, access_funcs, patterns, operation="*", unroll_factors=None, loop_order=None):
        self.name = name
        self.iteration_space = iteration_space
        self.access_funcs = access_funcs
        self.patterns = patterns
        self.operation = operation
        self.unroll_factors = unroll_factors or {}
        self.loop_order = loop_order or list(iteration_space.keys())

    def gen_code(self):

        # function signature
        code = "import triton\nimport triton.language as tl\n\n"
        code += "@triton.jit\ndef unroll_coarsen_kernel("
        code += "Avp, Aip, Arp, B, C, patterns, Avp_offsets, Aip_offsets, "
        code += f"k_rows: tl.constexpr, k_cols: tl.constexpr, N: tl.constexpr):\n"
        indent = '    '

        # basic setup
        code += self.generate_setup_code(indent)

        for pattern_id, active_rows in self.patterns.items():
            code += f"{indent}if pattern_id == {pattern_id}:\n"
            inner_indent = indent + '    '
            # write loop nest
            code += self.gen_loop_nest(inner_indent)

            inner_indent += '    '
            inner_indent += '    '

            unroll_factor = self.unroll_factors.get('j', 1)
            for i in range(unroll_factor):
                code += f"{inner_indent}# Unrolled iteration {i}\n"

                # Generate code for loading A matrix
                value_idx = 0
                for row in active_rows:
                    code += f"{inner_indent}A_{row}_{i} = tl.load(values_ptr + (num_ones * k_cols * i) + (j + {i}) + ({value_idx} * pattern_offset))\n"
                    value_idx += 1
                
                code += "\n"
                
                # Generate code for idx and B_row
                code += f"{inner_indent}idx_{i} = tl.load(indices_ptr + (i * k_cols) + j)\n"
                code += f"{inner_indent}B_row_{i} = tl.load(B + (idx_{i} * N) + tl.arange(0, N))\n\n"

                # Computations for C matrix
                for row in active_rows:
                    # C = tl.zeros(tl.float32, N)
                    code += f"{inner_indent}C_{row}_{i} = tl.zeros((N,), dtype=tl.float32)\n"

                    # C = A * B
                    code += f"{inner_indent}C_{row}_{i} = A_{row}_{i} * B_row_{i}\n"
                    # tl.atomic_add(C..."
                    code += f"{inner_indent}tl.atomic_add(C + ((row_idx + {row}) * N) + tl.arange(0, N), C_{row}_{i})\n\n"
                code += "\n"

        return code
    
    def generate_setup_code(self, indent="    "):
        code = ""
        code += f"{indent}row_pi = tl.program_id(axis=0)  # Index of row panel we are working on\n"
        code += f"{indent}pattern_id = tl.program_id(axis=1)\n\n"
        code += f"{indent}# Get the row panel start and end indices\n"
        code += f"{indent}panel_start = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi)\n"
        code += f"{indent}panel_end = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi + 1)\n\n"
        code += f"{indent}if panel_start == panel_end:\n"
        code += f"{indent*2}# row panel for this pattern is empty\n"
        code += f"{indent*2}return\n\n"
        code += f"{indent}row_idx = row_pi * k_rows  # We are on the row_pi-th row panel, each row panel has k_rows rows\n"
        code += f"{indent}pattern_ptr = patterns + (k_rows * pattern_id)  # Points to first element of the pattern\n\n"
        code += f"{indent}# Count the number of '1' bits in the pattern\n"
        code += f"{indent}num_ones = 0\n"
        code += f"{indent}for i in range(k_rows):\n"
        code += f"{indent*2}if (tl.load(pattern_ptr + i) == 1):\n"
        code += f"{indent*3}num_ones += 1\n\n"
        code += f"{indent}pattern_offset = k_cols\n\n"
        code += f"{indent}Avp_start_idx = tl.load(Avp_offsets + pattern_id)  # Index of the first value in the values array for the entire pattern\n"
        code += f"{indent}values_ptr = Avp + (Avp_start_idx) + (panel_start * num_ones)  # Points to the first value in the values array for this row panel\n\n"
        code += f"{indent}num_cols = (panel_end - panel_start)  # Number of total columns in this row panel for this pattern\n"
        code += f"{indent}num_kernels = num_cols // k_cols  # Number of kernels in this row panel\n\n"
        code += f"{indent}Aip_start_idx = tl.load(Aip_offsets + pattern_id)  # Index of the first index in the indices array for the entire pattern\n"
        code += f"{indent}indices_ptr = Aip + Aip_start_idx + panel_start  # Points to the first index in the indices array for this row panel\n\n"
        return code

    def gen_loop_nest(self, base_indent):
        code = ""
        indent = base_indent

        # Iterate over loop_order to generate loops in the specified order
        for var in self.loop_order:
            start, end, step = self.iteration_space[var]
            unroll_factor = self.unroll_factors.get(var, 1)

            # Create a loop that steps by the unroll factor
            code += f"{indent}for {var} in range({start}, {end}, {step * unroll_factor}):\n"
            indent += '    '

        return code

# Example usage of the new class
iteration_space = {'i': [0, 'num_kernels', 1], 'j': [0, 'k_cols', 1]}
patterns = {
    0: [0, 1, 2],  # Corresponds to pattern 111
    1: [0, 1],     # Corresponds to pattern 110
    2: [0, 2],     # Corresponds to pattern 101
    3: [0],        # Corresponds to pattern 100
    4: [1, 2],     # Corresponds to pattern 011
    5: [1],        # Corresponds to pattern 010
    6: [2]         # Corresponds to pattern 001
}
# Unroll 'j' by a factor of 2 and change loop order
unroll_factors = {}
loop_order = ['i', 'j']

# Define access functions
access_A_0 = TritonAccessFunction("A_0", "values_ptr", "(num_ones * k_cols * i) + j_unroll")
access_A_1 = TritonAccessFunction("A_1", "values_ptr", "(num_ones * k_cols * i) + j_unroll + pattern_offset")
access_A_2 = TritonAccessFunction("A_2", "values_ptr", "(num_ones * k_cols * i) + j_unroll + (2 * pattern_offset)")
access_B_row = TritonAccessFunction("B_row", "B + (idx * N)", "tl.arange(0, N)")

# Combine everything into a codelet with unrolling and loop exchange
codelet = TritonCodelet(
    name="unroll_coarsen",
    iteration_space=iteration_space,
    access_funcs=[access_A_0, access_A_1, access_A_2, access_B_row],
    patterns=patterns,
    unroll_factors=unroll_factors,
    loop_order=loop_order
)

# Generate the code
print(codelet.gen_code())