import numpy as np
import scipy.sparse as sp
import torch
import triton
import triton.language as tl

@triton.jit
def enumerated_kernel(A_ptr, B_ptr, C_ptr, N: tl.constexpr, K: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Get the warp index (1 warp per 3 rows)
    row_idx = (tl.program_id(0) // 32) * 3
    # Get the thread index within the warp (0 to 31)
    thread_idx = tl.program_id(0) % 32

    # Initialize accumulators for the results of three rows
    C0_acc = tl.zeros((N,), dtype=tl.float32)
    C1_acc = tl.zeros((N,), dtype=tl.float32)
    C2_acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over columns of A and rows of B with unrolling
    for k in range(0, K, 2):
        # Unroll iteration 1
        if k < K:
            # Calculate offsets of A and B values
            A0_offset = (row_idx * stride_am + k * stride_ak)
            A1_offset = ((row_idx + 1) * stride_am + k * stride_ak)
            A2_offset = ((row_idx + 2) * stride_am + k * stride_ak)

            B_num_cols = N // 32
            B_start = thread_idx * B_num_cols
            B_end = B_start + B_num_cols
            B_indices = tl.arange(0, N)
            B_mask = (B_indices >= B_start) & (B_indices < B_end)
            B_offset = (k * stride_bk + B_indices * stride_bn)

            # Each thread loads three of A's values and part of B's row
            A0 = tl.load(A_ptr + A0_offset)
            A1 = tl.load(A_ptr + A1_offset)
            A2 = tl.load(A_ptr + A2_offset)
            B_row = tl.load(B_ptr + B_offset, mask=B_mask, other=0.0)

            # Each thread accumulates partial results for its portion of the row
            C0_acc += A0 * B_row
            C1_acc += A1 * B_row
            C2_acc += A2 * B_row

        # Unroll iteration 2
        if k + 1 < K:
            # Calculate offsets of A and B values
            A0_offset = (row_idx * stride_am + (k + 1) * stride_ak)
            A1_offset = ((row_idx + 1) * stride_am + (k + 1) * stride_ak)
            A2_offset = ((row_idx + 2) * stride_am + (k + 1) * stride_ak)

            B_num_cols = N // 32
            B_start = thread_idx * B_num_cols
            B_end = B_start + B_num_cols
            B_indices = tl.arange(0, N)
            B_mask = (B_indices >= B_start) & (B_indices < B_end)
            B_offset = ((k + 1) * stride_bk + B_indices * stride_bn)

            # Each thread loads three of A's values and part of B's row
            A0 = tl.load(A_ptr + A0_offset)
            A1 = tl.load(A_ptr + A1_offset)
            A2 = tl.load(A_ptr + A2_offset)
            B_row = tl.load(B_ptr + B_offset, mask=B_mask, other=0.0)

            if (A0 != 0 and A1 != 0) and A2 != 0:
                C0_acc += A0 * B_row
                C1_acc += A1 * B_row
                C2_acc += A2 * B_row

            if (A0 != 0 and A1 != 0) and A2 == 0:
                C0_acc += A0 * B_row
                C1_acc += A1 * B_row
            
            if (A0 != 0 and A1 == 0) and A2 != 0:
                C0_acc += A0 * B_row
                C2_acc += A2 * B_row
            
            if (A0 == 0 and A1 != 0) and A2 != 0:
                C1_acc += A1 * B_row
                C2_acc += A2 * B_row

            if (A0 != 0 and A1 == 0) and A2 == 0:
                C0_acc += A0 * B_row

            if (A0 == 0 and A1 != 0) and A2 == 0:
                C1_acc += A1 * B_row
            
            if (A0 == 0 and A1 == 0) and A2 != 0:
                C2_acc += A2 * B_row

    # Perform warp-level reduction to sum results across all warp threads
    C0_acc = tl.sum(C0_acc, axis=0)
    C1_acc = tl.sum(C1_acc, axis=0)
    C2_acc = tl.sum(C2_acc, axis=0)

    # Store each thread's result to C
    C0_offset = row_idx * stride_cm + thread_idx * stride_cn
    C1_offset = (row_idx + 1) * stride_cm + thread_idx * stride_cn
    C2_offset = (row_idx + 2) * stride_cm + thread_idx * stride_cn
    tl.store(C_ptr + C0_offset, C0_acc)
    tl.store(C_ptr + C1_offset, C1_acc)
    tl.store(C_ptr + C2_offset, C2_acc)


# Handler Function to Launch the Kernel
def benchmark_enumerated(A, B):
    # Get matrix dimensions
    M, K = A.shape  # A is MxK
    K, N = B.shape  # B is KxN
    
    # Convert NumPy arrays to PyTorch tensors and move them to GPU
    device = torch.device('cuda')
    A_ptr = torch.tensor(A, dtype=torch.float32, device=device)
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Define strides for the matrices
    stride_am, stride_ak = A_ptr.stride()
    stride_bk, stride_bn = B_ptr.stride()
    stride_cm, stride_cn = C_ptr.stride()

    # Define grid - M warps, each warp handles three rows
    grid = (M // 3 * 32,)
    
     # Create CUDA events to record the time
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Launch Triton kernel (or any CUDA operation)
    start.record()

    enumerated_kernel[grid](A_ptr, B_ptr, C_ptr, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

    stop.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms