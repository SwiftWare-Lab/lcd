import triton
import triton.language as tl

@triton.jit
def unroll_coarsen_kernel(Avp, Aip, Arp, B, C, patterns, Avp_offsets, Aip_offsets, k_rows: tl.constexpr, k_cols: tl.constexpr, N: tl.constexpr):
    row_pi = tl.program_id(axis=0)  # Index of row panel we are working on
    pattern_id = tl.program_id(axis=1)

    # Get the row panel start and end indices
    panel_start = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi)
    panel_end = tl.load(Arp + (tl.num_programs(0) + 1) * pattern_id + row_pi + 1)

    if panel_start == panel_end:
        # row panel for this pattern is empty
        return

    row_idx = row_pi * k_rows  # We are on the row_pi-th row panel, each row panel has k_rows rows
    pattern_ptr = patterns + (k_rows * pattern_id)  # Points to first element of the pattern

    # Count the number of '1' bits in the pattern
    num_ones = 0
    for i in range(k_rows):
        if (tl.load(pattern_ptr + i) == 1):
            num_ones += 1

    pattern_offset = k_cols

    Avp_start_idx = tl.load(Avp_offsets + pattern_id)  # Index of the first value in the values array for the entire pattern
    values_ptr = Avp + (Avp_start_idx) + (panel_start * num_ones)  # Points to the first value in the values array for this row panel

    num_cols = (panel_end - panel_start)  # Number of total columns in this row panel for this pattern
    num_kernels = num_cols // k_cols  # Number of kernels in this row panel

    Aip_start_idx = tl.load(Aip_offsets + pattern_id)  # Index of the first index in the indices array for the entire pattern
    indices_ptr = Aip + Aip_start_idx + panel_start  # Points to the first index in the indices array for this row panel

    if pattern_id == 0:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (2 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (1 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))
                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (1 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (2 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (1 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (2 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


    if pattern_id == 1:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))
                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


    if pattern_id == 2:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


    if pattern_id == 3:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)


                A_0_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_0_1 = tl.zeros((N,), dtype=tl.float32)
                C_0_1 = A_0_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)


    if pattern_id == 4:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))
                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (1 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


    if pattern_id == 5:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


                A_1_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_1_1 = tl.zeros((N,), dtype=tl.float32)
                C_1_1 = A_1_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


    if pattern_id == 6:
        for i in range(0, num_kernels - num_kernels % 2, 2):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


            for j in range(0, k_cols - k_cols % 2, 2):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 1) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 1)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 1) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


        for i in range(num_kernels - num_kernels % 2, num_kernels):
            for j in range(0, k_cols - k_cols % 2, 2):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


                A_2_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 1) + (0 * pattern_offset))

                idx_1 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 1))
                B_row_1 = tl.load(B + (idx_1 * N) + tl.arange(0, N))

                C_2_1 = tl.zeros((N,), dtype=tl.float32)
                C_2_1 = A_2_1 * B_row_1
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_1)


            for j in range(k_cols - k_cols % 2, k_cols):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)



