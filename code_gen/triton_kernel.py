
import triton
import triton.language as tl

@triton.jit
def unroll_coarsen_kernel_3x4(Avp, Aip, Arp, B, C, patterns, Avp_offsets, Aip_offsets, k_rows: tl.constexpr, 
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
                if pattern_id == 0:
                    # Pattern Bits is 111
                    A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (0 * pattern_offset))
                    A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (1 * pattern_offset))
                    A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (2 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_0 = tl.zeros((N,), dtype=tl.float32)
                    C_0 = A_0 * B_row
                    tl.atomic_add(C + (row_idx + 0) * N + tl.arange(0, N), C_0)
                    C_1 = tl.zeros((N,), dtype=tl.float32)
                    C_1 = A_1 * B_row
                    tl.atomic_add(C + (row_idx + 1) * N + tl.arange(0, N), C_1)
                    C_2 = tl.zeros((N,), dtype=tl.float32)
                    C_2 = A_2 * B_row
                    tl.atomic_add(C + (row_idx + 2) * N + tl.arange(0, N), C_2)
                if pattern_id == 1:
                    # Pattern Bits is 110
                    A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (0 * pattern_offset))
                    A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (1 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_0 = tl.zeros((N,), dtype=tl.float32)
                    C_0 = A_0 * B_row
                    tl.atomic_add(C + (row_idx + 0) * N + tl.arange(0, N), C_0)
                    C_1 = tl.zeros((N,), dtype=tl.float32)
                    C_1 = A_1 * B_row
                    tl.atomic_add(C + (row_idx + 1) * N + tl.arange(0, N), C_1)
                if pattern_id == 2:
                    # Pattern Bits is 101
                    A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (0 * pattern_offset))
                    A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (2 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_0 = tl.zeros((N,), dtype=tl.float32)
                    C_0 = A_0 * B_row
                    tl.atomic_add(C + (row_idx + 0) * N + tl.arange(0, N), C_0)
                    C_2 = tl.zeros((N,), dtype=tl.float32)
                    C_2 = A_2 * B_row
                    tl.atomic_add(C + (row_idx + 2) * N + tl.arange(0, N), C_2)
                if pattern_id == 3:
                    # Pattern Bits is 100
                    A_0 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (0 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_0 = tl.zeros((N,), dtype=tl.float32)
                    C_0 = A_0 * B_row
                    tl.atomic_add(C + (row_idx + 0) * N + tl.arange(0, N), C_0)
                if pattern_id == 4:
                    # Pattern Bits is 011
                    A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (1 * pattern_offset))
                    A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (2 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_1 = tl.zeros((N,), dtype=tl.float32)
                    C_1 = A_1 * B_row
                    tl.atomic_add(C + (row_idx + 1) * N + tl.arange(0, N), C_1)
                    C_2 = tl.zeros((N,), dtype=tl.float32)
                    C_2 = A_2 * B_row
                    tl.atomic_add(C + (row_idx + 2) * N + tl.arange(0, N), C_2)
                if pattern_id == 5:
                    # Pattern Bits is 010
                    A_1 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (1 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_1 = tl.zeros((N,), dtype=tl.float32)
                    C_1 = A_1 * B_row
                    tl.atomic_add(C + (row_idx + 1) * N + tl.arange(0, N), C_1)
                if pattern_id == 6:
                    # Pattern Bits is 001
                    A_2 = tl.load(values_ptr + (num_ones * k_cols * i) + j + (2 * pattern_offset))
                    idx = tl.load(indices_ptr + (i * k_cols) + j)
                    B_row = tl.load(B + (idx * N) + tl.arange(0, N))
                    C_2 = tl.zeros((N,), dtype=tl.float32)
                    C_2 = A_2 * B_row
                    tl.atomic_add(C + (row_idx + 2) * N + tl.arange(0, N), C_2)

    