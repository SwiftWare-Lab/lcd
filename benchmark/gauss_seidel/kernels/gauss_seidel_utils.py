import numpy as np
from numba import cuda

def extract_diagonal_info(A_csr):
    """
    Extract diagonal positions and values from a CSR matrix.
    
    Args:
        A_csr: scipy.sparse.csr_matrix
    
    Returns:
        diag_pos: np.ndarray[int32] - position of diagonal in each row's data array
        diag_vals: np.ndarray[float64] - diagonal values for each row
    """
    n = A_csr.shape[0]
    diag_pos = np.zeros(n, dtype=np.int32)
    diag_vals = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        
        # Find diagonal position in this row
        found = False
        for j in range(row_start, row_end):
            if A_csr.indices[j] == i:
                diag_pos[i] = j
                diag_vals[i] = A_csr.data[j]
                found = True
                break
        
        if not found:
            # If no diagonal found, use a default position and value
            # This should not happen for well-formed matrices
            diag_pos[i] = row_start
            diag_vals[i] = 1.0
    
    return diag_pos, diag_vals


def extract_diagonal_info_gpu(A_csr):
    """
    Extract diagonal positions and values and transfer to GPU.
    
    Args:
        A_csr: scipy.sparse.csr_matrix
    
    Returns:
        d_diag_pos: cuda device array - position of diagonal in each row
        d_diag_vals: cuda device array - diagonal values for each row
    """
    diag_pos, diag_vals = extract_diagonal_info(A_csr)
    d_diag_pos = cuda.to_device(diag_pos)
    d_diag_vals = cuda.to_device(diag_vals)
    return d_diag_pos, d_diag_vals

