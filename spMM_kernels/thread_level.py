import triton
import triton.language as tl

@triton.jit
def thread_level_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Get the row index for the current thread (one thread per row)
    row_idx = tl.program_id(0)
    
    # Initialize an accumulator for the result of this row
    C_acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over columns of A and rows of B
    for k in range(K):
        A_value = tl.load(A_ptr + row_idx * stride_am + k * stride_ak)
        B_row = tl.load(B_ptr + k * stride_bk + tl.arange(0, N) * stride_bn)
        
        # Accumulate the result for the current row
        C_acc += A_value * B_row

    # Store the result in C
    tl.store(C_ptr + row_idx * stride_cm + tl.arange(0, N) * stride_cn, C_acc)


# Handler Code to Launch the Kernel
def launch_thread_level(A, B, C):
    # Get matrix dimensions
    M, K = A.shape  # A is MxK
    K, N = B.shape  # B is KxN
    
    # Convert matrices to pointers for Triton
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    # Define grid - M threads, each handling one row
    grid = (M,)
    
    # Launch the kernel with the defined grid
    thread_level_kernel[grid](
        A_ptr, B_ptr, C_ptr, 
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1)
    )