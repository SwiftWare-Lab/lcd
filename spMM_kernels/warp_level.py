import triton
import triton.language as tl

@triton.jit
def warp_level_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Get the row index for the current warp (1 warp per row)
    row_idx = tl.program_id(0)
    warp_idx = tl.thread_id() % 32  # Each warp has 32 threads
    
    # Initialize an accumulator for the result of this row
    C_acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over the K dimension in strides of 32
    for k in range(warp_idx, K, 32):
        A_value = tl.load(A_ptr + row_idx * stride_am + k * stride_ak)  # Load A[row_idx, k]
        B_row = tl.load(B_ptr + k * stride_bk + tl.arange(0, N) * stride_bn)  # Load B[k, :]
        
        # Accumulate partial result for this warp thread
        C_acc += A_value * B_row

    # Perform warp-level reduction to sum results across all warp threads
    C_acc = tl.sum(C_acc, axis=0)

    # Store the result in C
    if warp_idx == 0:  # Only one thread in the warp stores the result
        tl.store(C_ptr + row_idx * stride_cm + tl.arange(0, N) * stride_cn, C_acc)


# Handler Function to Launch the Kernel
def launch_warp_level(A, B, C):
    # Get matrix dimensions
    M, K = A.shape  # A is MxK
    K, N = B.shape  # B is KxN
    
    # Convert matrices to pointers for Triton
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    # Define grid - M warps, each warp handles one row
    grid = (M,)

    # Launch the kernel with one warp per row
    warp_level_kernel[grid](
        A_ptr, B_ptr, C_ptr, 
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1)
    )