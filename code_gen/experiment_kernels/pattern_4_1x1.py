
import sys
import os

sys.path.insert(1, '../utils')
import numpy as np
import scipy.sparse as sp
import torch
import triton
import triton.language as tl
from cfmtx_utils import preprocess_patterns, create_array_offsets, flatten_array
from scipy.io import mmwrite
from read_compressed_format import read_compressed_matrix

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
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (3 * pattern_offset))

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

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (3 * pattern_offset))

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

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (3 * pattern_offset))

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

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (3 * pattern_offset))

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

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 1:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


    if pattern_id == 2:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 3:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


    if pattern_id == 4:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 5:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


    if pattern_id == 6:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 7:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_0_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_0_0 = tl.zeros((N,), dtype=tl.float32)
                C_0_0 = A_0_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_0 = tl.zeros((N,), dtype=tl.float32)
                C_0 = A_0 * B_row
                tl.atomic_add(C + ((row_idx + 0) * N) + tl.arange(0, N), C_0)


    if pattern_id == 8:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (2 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (2 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 9:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
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


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
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


    if pattern_id == 10:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 11:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_1_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_1_0 = tl.zeros((N,), dtype=tl.float32)
                C_1_0 = A_1_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_1 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_1 = tl.zeros((N,), dtype=tl.float32)
                C_1 = A_1 * B_row
                tl.atomic_add(C + ((row_idx + 1) * N) + tl.arange(0, N), C_1)


    if pattern_id == 12:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (1 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (1 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


    if pattern_id == 13:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_2_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_2_0 = tl.zeros((N,), dtype=tl.float32)
                C_2_0 = A_2_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_2 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_2 = tl.zeros((N,), dtype=tl.float32)
                C_2 = A_2 * B_row
                tl.atomic_add(C + ((row_idx + 2) * N) + tl.arange(0, N), C_2)


    if pattern_id == 14:
        for i in range(0, num_kernels - num_kernels % 1, 1):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)


        for i in range(num_kernels - num_kernels % 1, num_kernels):
            for j in tl.range(0, k_cols - k_cols % 1, 1, num_stages=3):
                A_3_0 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j + 0) + (0 * pattern_offset))

                idx_0 = tl.load(indices_ptr + ((i + 0) * k_cols) + (j + 0))
                B_row_0 = tl.load(B + (idx_0 * N) + tl.arange(0, N))

                C_3_0 = tl.zeros((N,), dtype=tl.float32)
                C_3_0 = A_3_0 * B_row_0
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3_0)


            for j in tl.range(k_cols - k_cols % 1, k_cols, num_stages=3):
                A_3 = tl.load(values_ptr + (num_ones * k_cols * (i + 0)) + (j) + (0 * pattern_offset))

                idx = tl.load(indices_ptr + ((i + 0) * k_cols) + j)
                B_row = tl.load(B + (idx * N) + tl.arange(0, N))

                C_3 = tl.zeros((N,), dtype=tl.float32)
                C_3 = A_3 * B_row
                tl.atomic_add(C + ((row_idx + 3) * N) + tl.arange(0, N), C_3)

def run_unroll_corasen(A_vp, A_ip, A_rp,
                                B_ptr, C_ptr,
                                patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr,
                                kernel_rows, kernel_cols,
                                B_col,
                                grid):
    
    unroll_coarsen_kernel[grid](A_vp, A_ip, A_rp,
                                B_ptr, C_ptr,
                                patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr,
                                kernel_rows, kernel_cols,
                                B_col)
    

def benchmark_unroll_coarsen(A_cf, B):
    # Convert the patterns dict to a list for the GPU
    pattern_list = preprocess_patterns(A_cf.patterns)

    # Create the offsets for the Avp and Aip arrays
    Avp_offsets = create_array_offsets(A_cf.padded_values_data_array)
    Aip_offsets = create_array_offsets(A_cf.indices_array)

    # Load everything into the GPU
    device = torch.device('cuda')

    # Flatten and load arrays for A
    A_vp_flattened = flatten_array(A_cf.padded_values_data_array)
    A_vp = torch.tensor(A_vp_flattened, dtype=torch.int32, device=device)
    A_ip_flattened = flatten_array(A_cf.indices_array)
    A_ip = torch.tensor(A_ip_flattened, dtype=torch.int32, device=device)
    A_rp = torch.tensor(A_cf.row_panels_pointer, dtype=torch.int32, device=device)
    patterns_ptr = torch.tensor(pattern_list, dtype=torch.int32, device=device)
    Avp_offsets_ptr = torch.tensor(Avp_offsets, dtype=torch.int32, device=device)
    Aip_offsets_ptr = torch.tensor(Aip_offsets, dtype=torch.int32, device=device)

    # Load B and C arrays
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.zeros((A_cf.num_row, B.shape[1]), dtype=torch.float32, device=device)

    # Launch the kernel
    num_row_panels = len(A_cf.row_panels_pointer[0]) - 1
    grid = (num_row_panels, len(A_cf.patterns))

    kernel_launcher = lambda: run_unroll_corasen(
        A_vp, A_ip, A_rp,
        B_ptr, C_ptr,
        patterns_ptr, Avp_offsets_ptr, Aip_offsets_ptr,
        A_cf.kernel_rows,  A_cf.kernel_cols,
        B.shape[1],
        grid
    )

    median_time = triton.testing.do_bench(kernel_launcher, quantiles=[0.5])

    return median_time
