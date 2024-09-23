import triton
import triton.language as tl

@triton.jit
def _3row_warp_unroll_kernel(A_ptr, B_ptr, C_ptr, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Get the row index for the current warp (1 warp handles 3 rows)
    row_idx = tl.program_id(0) * 3  # Handle 3 rows at a time
    warp_idx = tl.thread_id() % 32  # Each warp has 32 threads
    
    # Initialize accumulators for the result of these 3 rows
    c00 = tl.zeros((2,), dtype=tl.float32)
    c01 = tl.zeros((2,), dtype=tl.float32)
    c10 = tl.zeros((2,), dtype=tl.float32)
    c11 = tl.zeros((2,), dtype=tl.float32)
    c20 = tl.zeros((2,), dtype=tl.float32)
    c21 = tl.zeros((2,), dtype=tl.float32)

    # Loop over K dimension in strides of 32
    for k in range(warp_idx, K, 32):
        A0 = tl.load(A_ptr + row_idx * stride_am + k * stride_ak)
        A1 = tl.load(A_ptr + (row_idx + 1) * stride_am + k * stride_ak)
        A2 = tl.load(A_ptr + (row_idx + 2) * stride_am + k * stride_ak)
        
        B_col0 = tl.load(B_ptr + k * stride_bk + tl.arange(0, 2) * stride_bn)
        B_col1 = tl.load(B_ptr + k * stride_bk + tl.arange(2, 4) * stride_bn)

        # Accumulate results for 3 rows and 2 columns
        if A0 != 0:
            c00 += A0 * B_col0
            c01 += A0 * B_col1
        if A1 != 0:
            c10 += A1 * B_col0
            c11 += A1 * B_col1
        if A2 != 0:
            c20 += A2 * B_col0
            c21 += A2 * B_col1

    # Warp-level reduction to sum the results from all threads in the warp
    c00 = tl.sum(c00, axis=0)
    c01 = tl.sum(c01, axis=0)
    c10 = tl.sum(c10, axis=0)
    c11 = tl.sum(c11, axis=0)
    c20 = tl.sum(c20, axis=0)
    c21 = tl.sum(c21, axis=0)

    # Store results back to C
    if warp_idx == 0:  # Only one thread stores the result
        tl.store(C_ptr + row_idx * stride_cm + tl.arange(0, 2) * stride_cn, c00)
        tl.store(C_ptr + (row_idx + 1) * stride_cm + tl.arange(0, 2) * stride_cn, c10)
        tl.store(C_ptr + (row_idx + 2) * stride_cm + tl.arange(0, 2) * stride_cn, c20)


# Handler Function to Launch the Kernel
def launch_3row_warp_unroll(A, B, C):
    # Get matrix dimensions
    M, K = A.shape  # A is MxK
    K, N = B.shape  # B is KxN
    
    # Convert matrices to pointers for Triton
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    # Define grid - M // 3 warps, each warp handles 3 rows
    grid = (M // 3,)

    # Launch the kernel with one warp per 3 rows
    _3row_warp_unroll_kernel[grid](
        A_ptr, B_ptr, C_ptr, 
        K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1)
    )