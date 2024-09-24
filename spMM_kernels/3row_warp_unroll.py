import numpy as np
import scipy.sparse as sp
import torch
import triton
import triton.language as tl

@triton.jit
def _3row_warp_unroll_kernel(A_ptr, B_ptr, C_ptr, N: tl.constexpr, K: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Get the row index for the current warp (1 warp handles 3 rows)
    row_idx = tl.program_id(0) // 32  # Handle 3 rows at a time
    # Get the thread index within the warp (0 to 31)
    thread_idx = tl.program_id(0) % 32

    # Initialize accumulators for the result of these 3 rows
    C0_acc = tl.zeros((N // 32,), dtype=tl.float32)
    C1_acc = tl.zeros((N // 32,), dtype=tl.float32)
    C2_acc = tl.zeros((N // 32,), dtype=tl.float32)
    
    for k in range(K):
        # Calculate offsets of A and B values
        A0_offset = (row_idx * stride_am + k * stride_ak)
        A1_offset = ((row_idx + 1) * stride_am + k * stride_ak)
        A2_offset = ((row_idx + 2) * stride_am + k * stride_ak)

        B_num_cols = tl.constexpr(N // 32)
        B_start = tl.constexpr(thread_idx * B_num_cols)
        B_end = tl.constexpr(B_start + B_num_cols)
        B_indices = tl.arange(0, N)
        B_mask = (B_indices >= B_start) & (B_indices < B_end)
        B_offset = (k * stride_bk + B_indices * stride_bn)

        # Each thread loads three of A's values and two of B's values
        A0 = tl.load(A_ptr + A0_offset)
        A1 = tl.load(A_ptr + A1_offset)
        A2 = tl.load(A_ptr + A2_offset)
        B_row = tl.load(B_ptr + B_offset, mask=B_mask, other=0.0)

        # Each thread accumulates partial results for its portion of the row
        C0_acc += A0 * B_row
        C1_acc += A1 * B_row
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
def launch_3row_warp_unroll(A, B):
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

    # Define grid - M warps, each warp handles three rows
    grid = (M//3 * 32,)

    # Launch the kernel with one warp per row
    _3row_warp_unroll_kernel[grid](
        A_ptr, B_ptr, C_ptr, 
        N, K, 
        A.strides[0] // A.itemsize, A.strides[1] // A.itemsize,
        B.strides[0] // B.itemsize, B.strides[1] // B.itemsize,
        C.strides[0] // C.itemsize, C.strides[1] // C.itemsize
    )

    # Convert C_ptr back to NumPy array
    C = C_ptr.cpu().numpy()
    print(C)
    return C

# Test the kernel
def main():

    # Define matrix dimensions
    M, K, N = 96, 16, 16

    # Create random matrices with numPy
    A_sp = sp.random(M, N, 0.2, format='csr')
    A = A_sp.toarray()
    B = np.random.rand(N, K).astype(np.float32)

    # Launch the kernel
    C = launch_3row_warp_unroll(A, B)

    # Compare with built-in implementation
    C_ref = A @ B
    #print(C)
    #print(C_ref)
    print(np.allclose(C, C_ref))


if __name__ == '__main__':
    main()