import triton
import triton.language as tl
import numpy as np
import torch

@triton.jit
def warp_level_kernel(A_ptr, B_ptr, C_ptr, N: tl.constexpr, K: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Get the row index for the current warp (1 warp per row)
    row_idx = tl.program_id(0) // 32
    # Get the thread index within the warp (0 to 31)
    thread_idx = tl.program_id(0) % 32

    # Initialize an accumulator for the result of this row
    C_acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over columns of A and rows of B
    for k in range(K):
        A_offset = (row_idx * stride_am + k * stride_ak)
        B_num_cols = tl.constexpr(N // 32)
        B_start = tl.constexpr(thread_idx * B_num_cols)
        B_end = tl.constexpr(B_start + B_num_cols)
        B_indices = tl.arange(0, N)
        B_mask = (B_indices >= B_start) & (B_indices < B_end)
        B_offset = (k * stride_bk + B_indices * stride_bn)

        # Each thread loads part of A's row and part of B's row
        A_value = tl.load(A_ptr + A_offset)
        B_row = tl.load(B_ptr + B_offset, mask=B_mask, other=0.0)

        # Each thread accumulates partial results for its portion of the row
        C_acc += A_value * B_row

    # Perform warp-level reduction to sum results across all warp threads
    C_acc = tl.sum(C_acc, axis=0)

    # Store the result in C, each thread writes its part of the row
    C_offset = row_idx * stride_cm + thread_idx * stride_cn
    tl.store(C_ptr + C_offset, C_acc)



# Handler Function to Launch the Kernel
def benchmark_warp_level(A, B):
    # Get matrix dimensions
    M, K = A.shape  # A is MxK
    K, N = B.shape  # B is KxN
    
    # Convert NumPy arrays to PyTorch tensors and move them to GPU
    device = torch.device('cuda')
    A_ptr = torch.tensor(A, dtype=torch.float32, device=device)
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    # Create a zero tensor for the result
    C = np.zeros((M, N), dtype=np.float32)
    C_ptr = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Define grid - M warps, each warp handles one row
    grid = (M * 32,)

     # Create CUDA events to record the time
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Launch Triton kernel (or any CUDA operation)
    start.record()

    # Launch the kernel with one warp per row
    warp_level_kernel[grid](
        A_ptr, B_ptr, C_ptr, 
        N, K, 
        A.strides[0] // A.itemsize, A.strides[1] // A.itemsize,
        B.strides[0] // B.itemsize, B.strides[1] // B.itemsize,
        C.strides[0] // C.itemsize, C.strides[1] // C.itemsize
    )

    stop.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(stop)

    return elapsed_time_ms