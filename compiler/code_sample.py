import numpy as np


def tensor_times_matrix(X, M):
    """
    Multiplies a 3D tensor X by a matrix M along the second dimension.

    Parameters:
        X (ndarray): Input tensor of shape (I, J, K).
        M (ndarray): Input matrix of shape (J, L).

    Returns:
        Y (ndarray): Resulting tensor of shape (I, K, L).
        
    Schedule:
        Unroll (i,2), (k,2), (l,2), 
        jam(j)
        Duplicate sparse conditions
        coarsen and map
         
    """
    I, J, K = X.shape
    J_M, L = M.shape
    if J != J_M:
        raise ValueError("The second dimension of tensor X must match the first dimension of matrix M.")

    # Initialize the result tensor Y with zeros
    Y = np.zeros((I, K, L))

    # Perform the tensor-matrix multiplication
    for i in range(I):  # Loop over the first dimension of X
        for k in range(K):  # Loop over the third dimension of X
            for l in range(L):  # Loop over the second dimension of M
                current_sum = 0
                for j in range(J):  # Loop over the second dimension of X and first of M (contracted dimension)
                    if X[i, j, k] != 0:  # Only compute if both elements are non-zero
                        current_sum += X[i, j, k] * M[j, l]
                Y[i, k, l] = current_sum

    return Y

# tensor code tensor times vector
def tensor_times_vector(X, v):
    """
    Multiplies a 3D tensor X by a vector v along the second dimension.

    Parameters:
        X (ndarray): Input tensor of shape (I, J, K).
        v (ndarray): Input vector of shape (J,).

    Returns:
        Y (ndarray): Resulting tensor of shape (I, K).
    Schedule:
        Unroll (i,2), (k,2), (l,2), 
        jam(j)
        Duplicate sparse conditions
        coarsen and map
    """
    I, J, K = X.shape
    if v.shape[0] != J:
        raise ValueError("The length of vector v must match the second dimension of tensor X.")

    # Initialize the result tensor Y with zeros
    Y = np.zeros((I, K))

    # Perform the tensor-vector multiplication
    for i in range(I):  # Loop over the first dimension of X
        for k in range(K):  # Loop over the third dimension of X
            current_sum = 0
            for j in range(J):  # Loop over the second dimension of X (contracted dimension)
                if X[i, j, k] != 0:  # Only compute if the tensor element is non-zero
                    current_sum += X[i, j, k] * v[j]
            Y[i, k] = current_sum

    return Y


def sampled_ddm(A, B, Mask):
    """
    Perform sampled dense-dense matrix multiplication using a mask.

    Parameters:
        A (ndarray): First input matrix of shape (m, n).
        B (ndarray): Second input matrix of shape (n, p).
        Mask (ndarray): Boolean mask matrix of shape (m, p) indicating which elements to compute.
    Returns:
        C (ndarray): Result matrix of shape (m, p) with computed elements where Mask is True.
        Schedule:
        Unroll (i,2), (j,2) 
        jam(k)
        Duplicate sparse conditions (Mask)
        coarsen and map
    """
    m, n = A.shape
    n_B, p = B.shape
    if n != n_B:
        raise ValueError("Inner dimensions of A and B must match.")
    if Mask.shape != (m, p):
        raise ValueError("Mask shape must match the output matrix shape (m, p).")

    # Initialize the result matrix C with zeros
    C = np.zeros((m, p))

    # Perform the masked multiplication
    for i in range(m):  # Loop over rows of A
        for j in range(p):  # Loop over columns of B
            if Mask[i, j]:  # Only compute if the mask is True
                current_sum = 0
                for k in range(n):  # Loop over the shared dimension
                    current_sum += A[i, k] * B[k, j]
                C[i, j] = current_sum

    return C
