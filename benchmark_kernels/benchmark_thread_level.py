import numpy as np
import triton
import triton.language as tl
import torch


@triton.jit
def thread_level_kernel(A_ptr, B_ptr, C_ptr, N: tl.constexpr, K: tl.constexpr, stride_am, stride_ak, stride_bk,
                        stride_bn, stride_cm, stride_cn):
    # Get the row index for the current thread (one thread per row)
    row_idx = tl.program_id(0)

    # Initialize an accumulator for the result of this row
    C_acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over columns of A and rows of B
    for k in range(K):
        # Perform pointer arithmetic and cast all indices/strides to int32
        A_offset = (row_idx * stride_am + k * stride_ak)
        B_offset = (k * stride_bk + tl.arange(0, N) * stride_bn)

        A_value = tl.load(A_ptr + A_offset)
        B_row = tl.load(B_ptr + B_offset)

        # Accumulate the result for the current row
        C_acc += A_value * B_row

    # Store the result in C, ensuring indices are cast to int32
    C_offset = (row_idx * stride_cm + tl.arange(0, N) * stride_cn)
    tl.store(C_ptr + C_offset, C_acc)


# A and B are numpy arrays

# Handler Code to Launch the Kernel
def benchmark_thread_level(A, B):
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

    # Define grid - M threads, each handling one row
    grid = (M,)

    # Create CUDA events to record the time
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Launch Triton kernel (or any CUDA operation)
    start.record()

    # Launch the kernel with the defined grid
    thread_level_kernel[grid](
        A_ptr, B_ptr, C_ptr,
        N, K,
        A.strides[0] // A.itemsize, A.strides[1] // A.itemsize,
        B.strides[0] // B.itemsize, B.strides[1] // B.itemsize,
        C.strides[0] // C.itemsize, C.strides[1] // C.itemsize
    )

    stop.record()

    stop.synchronize()

    elapsed_time_ms = start.elapsed_time(stop)

    # print("C Thread Level :", C_ptr.cpu().numpy()[0:5, 0:5])

    # execution_time_s = elapsed_time_ms / 1e3  # in seconds

    return elapsed_time_ms
