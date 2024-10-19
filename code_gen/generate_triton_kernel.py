import sys

def generate_triton_kernel(k_rows, k_cols):
    kernel_template = """
import triton
import triton.language as tl

@triton.jit
def unroll_coarsen_kernel(Avp, Aip, Arp, B, C, patterns, Avp_offsets, Aip_offsets, k_rows: tl.constexpr, 
                                            k_cols: tl.constexpr, N: tl.constexpr):
    row_pi = tl.program_id(axis=0)
    pattern_id = tl.program_id(axis=1)
    panel_start = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi)
    panel_end = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi + 1)

    if panel_start == panel_end:
        return

    row_idx = row_pi * k_rows
    pattern_ptr = patterns + (k_rows * pattern_id)
    num_ones = 0
    for i in range(k_rows):
        if (tl.load(pattern_ptr + i) == 1):
            num_ones += 1

    pattern_offset = k_cols
    Avp_start_idx = tl.load(Avp_offsets + pattern_id)
    values_ptr = Avp + (Avp_start_idx) + (panel_start * num_ones)
    num_cols = (panel_end - panel_start)
    num_kernels = num_cols // k_cols
    Aip_start_idx = tl.load(Aip_offsets + pattern_id)
    indices_ptr = Aip + Aip_start_idx + panel_start

    for i in range(num_kernels):
        for j in range(k_cols):
            first_value = tl.load(values_ptr + (num_ones * k_cols * i) + j)
            if (first_value != 0.0):
{cases}
    """
    indentation = "    "
    cases = ""
    # Generate code for each possible pattern.
    for pattern in range(2 ** k_rows - 1):
        inverted_pattern = (2 ** k_rows - 1) - pattern
        pattern_bits = f"{inverted_pattern:0{k_rows}b}"

        # Generate if pattern check if statement
        case_code = indentation * 4
        case_code += f"if pattern_id == {pattern}:\n"
        case_code += indentation * 5
        case_code += f"# Pattern Bits is {pattern_bits}\n"
        case_code += "\n"
        
        # Generate A matrix loads
        for r in range(k_rows):
            if pattern_bits[r] == "1":
                case_code += f"{indentation * 5}A_{r} = tl.load(values_ptr + (num_ones * k_cols * i) + j + ({r} * pattern_offset))\n"

        case_code += "\n"

        # Generate idx computation
        case_code += f"{indentation * 5}idx = tl.load(indices_ptr + (i * k_cols) + j)\n"
        case_code += "\n"

        # Generate B row load  
        case_code += f"{indentation * 5}B_row = tl.load(B + (idx * N) + tl.arange(0, N))\n"
        case_code += "\n"

        # Generate C matrix loads and computations
        for r in range(k_rows):
            if pattern_bits[r] == "1":
                case_code += f"{indentation * 5}C_{r} = tl.zeros((N,), dtype=tl.float32)\n"
                case_code += f"{indentation * 5}C_{r} = A_{r} * B_row\n"
                case_code += f"{indentation * 5}tl.atomic_add(C + (row_idx + {r}) * N + tl.arange(0, N), C_{r})\n\n"
        
        case_code += "\n"

        cases += case_code

    # Fill in the template with the generated case code
    return kernel_template.format(k_rows=k_rows, k_cols=k_cols, cases=cases)

# Small test for codegen
def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_triton_kernel.py arg1 arg2")
        sys.exit(1)

    k_rows = int(sys.argv[1])
    k_cols = int(sys.argv[2])

    # File to write the generated code
    with open("triton_kernel.py", "w") as f:
        f.write(generate_triton_kernel(k_rows, k_cols))

if __name__ == "__main__":
    main()