import os
import sys

import numpy as np
from scipy.io import mmread, mmwrite
from sksparse.cholmod import cholesky_AAt # pip install scikit-sparse
from scipy.sparse import issparse


def compute_cholesky_lfactor(input_mtx_file: str, output_mtx_file: str):
    """
    Loads a matrix from a Matrix Market (.mtx) file, computes its
    Cholesky L-factor using sparse factorization (with default reordering),
    and saves the L-factor to a new .mtx file.

    Args:
        input_mtx_file (str): Path to the input Matrix Market file.
        output_mtx_file (str): Path to save the output L-factor Matrix Market file.
    """
    print(f"Attempting to load matrix from: {input_mtx_file}")

    try:
        # 1. Load the Matrix Market file
        # mmread loads the matrix, typically as a CSR or COO sparse matrix
        A = mmread(input_mtx_file)

        if not issparse(A):
            # Convert to sparse if it was loaded as a dense array
            print("Warning: Input matrix was loaded as dense. Converting to CSR sparse format.")
            A = A.tocsc()

        print(f"Matrix loaded successfully. Shape: {A.shape}, Type: {A.dtype}, Sparse format: {type(A)}")

        # if A is not in csc format, convert it
        if not A.getformat() == 'csc':
            print("Converting matrix to CSC format for Cholesky factorization...")
            A = A.tocsc()

        # 2. Compute the Cholesky L-factor
        # cholesky_AAt computes the Cholesky factorization of A^T * A.
        # This function is used because standard sparse Cholesky requires the
        # input matrix to be symmetric positive definite. It also
        # automatically handles sparse reordering (e.g., AMD) for efficiency.
        print("Starting Cholesky factorization (L^T * L = A^T * A)...")
        L = cholesky_AAt(A)

        # extract the L factor as a sparse matrix
        L = L.L()
        print(f"Cholesky L-factor computed successfully. Shape: {L.shape}")

        # 3. Store the L-factor as a Matrix Market file
        # The L factor is an Upper Triangular matrix in this case (L^T * L = A^T * A)
        # We can store it directly. mmwrite handles sparse matrices.
        output_mtx_folder = "/".join(output_mtx_file.split("/")[:-1])
        print(output_mtx_folder)
        os.makedirs(output_mtx_folder, exist_ok=True)
        print(f"Saving L-factor to: {output_mtx_file}")
        mmwrite(output_mtx_file, L, comment='Cholesky L-factor')

        print("Operation complete.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_mtx_file}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_lfactors.py <input_matrix.mtx> <output_lfactor_name.mtx>")
        sys.exit(1)
    # Define file paths
    list_file = sys.argv[1]
    matrix_folder = sys.argv[2]
    output_folder = sys.argv[3]
    with open(list_file) as f:
        lines = f.readlines()
        mat_list = map(str.strip, lines)
        for mat_name in mat_list:
            input_file = os.path.join(matrix_folder, mat_name)
            output_file = os.path.join(output_folder, mat_name)
            compute_cholesky_lfactor(input_file, output_file)
    # input_file = sys.argv[1] # e.g., "input_matrix.mtx"
    # output_file = sys.argv[2] # e.g., "l_factor.mtx"
    # compute_cholesky_lfactor(input_file, output_file)

