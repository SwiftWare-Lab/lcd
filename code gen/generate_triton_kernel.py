# NOT YET WORKING CODE

def generate_triton_kernel(k_rows, k_cols):
    kernel_template = """
@triton.jit
def unroll_coarsen_kernel_{k_rows}x{k_cols}(Avp, Aip, Arp, B, C, patterns, Avp_offsets, Aip_offsets, k_rows: tl.constexpr, 
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

    cases = ""
    # Generate code for each possible pattern.
    for pattern in range(2 ** k_rows):
        pattern_bits = f"{pattern:0{k_rows}b}"  # Get the binary pattern
        case_code = f"if pattern_id == {pattern}:\n"
        loads = []
        computes = []
        atomic_adds = []

        # Generate loads and computation based on pattern
        for r in range(k_rows):
            if pattern_bits[r] == "1":
                loads.append(f"A_{r} = tl.load(values_ptr + (num_ones * k_cols * i) + j + ({r} * pattern_offset))")
                computes.append(f"C_{r} = A_{r} * B_row")
                atomic_adds.append(f"tl.atomic_add(C + ((row_idx + {r}) * N) + tl.arange(0, N), C_{r})")

        # Join the generated code for this pattern
        case_code += "\n    ".join(loads) + "\n"
        case_code += "    idx = tl.load(indices_ptr + (i * k_cols) + j)\n"
        case_code += "    B_row = tl.load(B + (idx * N) + tl.arange(0, N))\n"
        case_code += "\n    ".join(f"C_{r} = tl.zeros((N,), dtype=tl.float32)" for r in range(k_rows) if pattern_bits[r] == "1") + "\n"
        case_code += "\n    ".join(computes) + "\n"
        case_code += "\n    ".join(atomic_adds) + "\n"

        cases += case_code

    # Fill in the template with the generated case code
    return kernel_template.format(k_rows=k_rows, k_cols=k_cols, cases=cases)