import torch
import triton
import triton.language as tl

# Triton kernel for SpMM using Gustavson's Algorithm (CSR format)
@triton.jit
def spmm_csr_kernel(
    data_ptr,          # Pointer to non-zero values of sparse matrix A
    indices_ptr,       # Pointer to column indices for non-zero values of sparse matrix A
    indptr_ptr,        # Pointer to row pointers of sparse matrix A
    B_ptr,             # Pointer to dense matrix B
    C_ptr,             # Pointer to result matrix C
    N: tl.constexpr,   # Feature size for dense matrix, should be a compile-time constant
    stride_B,          # Stride for dense matrix B (row-major)
    stride_C,          # Stride for dense matrix C (row-major)
):
    # Each Triton program computes one row of C
    row_idx = tl.program_id(0)  # Get the row this program is responsible for

    # Fetch the row start and end from the CSR indptr
    row_start = tl.load(indptr_ptr + row_idx)
    row_end = tl.load(indptr_ptr + row_idx + 1)

    # Initialize the result row for C (dense row)
    result = tl.zeros((N,), dtype=tl.float32)  # Ensure N is compile-time constant

    # Loop over non-zero elements in row
    for i in range(row_start, row_end):
        val_A = tl.load(data_ptr + i)
        col_A = tl.load(indices_ptr + i)

        # Parallelize over columns of matrix B corresponding to col_A
        j = tl.arange(0, N)  # Create a parallelized index array for columns
        val_B = tl.load(B_ptr + col_A * stride_B + j)
        result += val_A * val_B

    # Store the result row back into the dense matrix C
    tl.store(C_ptr + row_idx * stride_C + tl.arange(0, N), result)

def spmm_csr(A_csr, B, C):
    # A_csr: sparse matrix in CSR format
    # B: dense matrix
    # C: output matrix

    # Extract components of the CSR matrix A
    data = A_csr.data  # Non-zero values
    indices = A_csr.indices  # Column indices of non-zero values
    indptr = A_csr.indptr  # Row pointers

    M, _ = A_csr.shape  # A is MxK
    _, N = B.shape  # B is KxN

    # Allocate GPU memory using PyTorch and copy data
    device = torch.device('cuda')

    # Convert NumPy arrays to PyTorch tensors and move them to GPU
    data_ptr = torch.tensor(data, dtype=torch.float32, device=device)
    indices_ptr = torch.tensor(indices, dtype=torch.int32, device=device)
    indptr_ptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.tensor(C, dtype=torch.float32, device=device)

    # Call Triton kernel using Gustavson's algorithm
    grid = lambda meta: (M,)

    def kernel_launch():
        spmm_csr_kernel[grid](
            data_ptr, indices_ptr, indptr_ptr,
            B_ptr, C_ptr,
            N,
            stride_B=B.shape[1],
            stride_C=C.shape[1]  # Removed stride_A, since it's not used
        )
        torch.cuda.synchronize()  # Ensure GPU operations complete
    return kernel_launch, C_ptr