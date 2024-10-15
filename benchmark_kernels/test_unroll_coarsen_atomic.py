from scipy.io import mmread
import os
import sys
import numpy as np
from benchmark_unroll_coarsen3 import test_unroll_coarsen
import cupy as cp
import scipy.sparse as sp

sys.path.insert(1, '../utils')

from read_compressed_format import read_compressed_matrix

# tests for benchmark_unroll_coarsen2.py

cols = [32, 64, 128, 256, 512]
row_n = 768
densities = np.arange(0.1, 0.8, 0.1)

passed = 0
total = 0

for col in cols:
    for density in densities:
        A_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.mtx"
        B_path = f"../matrices/B_sparse_512x{col}_1.0.mtx"

        # Load A and B as np arrays
        A_sp = mmread(A_path)
        A_sp = A_sp.tocsr()
        B_sp = mmread(B_path)

        # Convert scipy arrays to numpy arrays
        A = A_sp.toarray().astype(np.float32)
        B = B_sp.toarray().astype(np.float32)
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

        os.system(f"python3 ../utils/compressed_format_generator.py {A_path} 3 4 > /dev/null 2>&1")

        A_cf_path = f"../matrices/A_sparse_{row_n}x512_{density:.2f}.cfmtx"
        A_cf = read_compressed_matrix(A_cf_path)

        exec_time, C = test_unroll_coarsen(A_cf, B)

        C_ref = A @ B

        # cuBLAS timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        c_cuBLAS = cp.matmul(cp.asarray(A), cp.asarray(B))

        end_event.record()

        # Wait for all operations to complete
        end_event.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)

        c_cuBLAS_ref = c_cuBLAS.get()

        # cuSPARSE timing
        # Convert NumPy arrays to CuPy arrays
        A_csr = cp.sparse.csr_matrix(cp.asarray(A))  # Convert sparse matrix to CuPy CSR
        B_dense = cp.asarray(B)  # Convert dense matrix to CuPy array

        # Initialize cuSPARSE
        cusparse_handle = cp.cuda.cusparse.create()

        # Sparse matrix descriptor (SpMat)
        spmat_descr = cp.cuda.cusparse.createCsr(
            A_csr.shape[0],  # number of rows
            A_csr.shape[1],  # number of columns
            A_csr.nnz,  # number of non-zero elements (nnz)
            A_csr.indptr.data.ptr,  # row pointers
            A_csr.indices.data.ptr,  # column indices
            A_csr.data.data.ptr,  # non-zero values
            cp.cuda.runtime.CUDA_R_64F,  # data type (double precision)
            cp.cuda.cusparse.CUSPARSE_INDEX_32I,  # index type for row pointers
            cp.cuda.cusparse.CUSPARSE_INDEX_32I,  # index type for column indices
            cp.cuda.cusparse.CUSPARSE_INDEX_BASE_ZERO  # index base (0-based indexing)
        )

        # Dense matrix descriptor (DnMat)
        dense_descr = cp.cuda.cusparse.createDnMat(
            B_dense.shape[0],  # rows of B
            B_dense.shape[1],  # cols of B
            B_dense.shape[1],  # leading dimension (B's major dimension)
            B_dense.data.ptr,  # pointer to the dense data
            cp.cuda.runtime.CUDA_R_64F,  # data type
            cp.cuda.cusparse.CUSPARSE_ORDER_ROW  # memory layout (row-major)
        )

        # Output dense matrix
        output_matrix = cp.zeros((A_csr.shape[0], B_dense.shape[1]), dtype=cp.float64)
        output_descr = cp.cuda.cusparse.createDnMat(
            output_matrix.shape[0], output_matrix.shape[1], output_matrix.shape[1],
            output_matrix.data.ptr,
            cp.cuda.runtime.CUDA_R_64F,  # data type
            cp.cuda.cusparse.CUSPARSE_ORDER_ROW  # memory layout (row-major)
        )

        # Scalars for SpMM
        alpha = cp.array(1.0, dtype=cp.float64)
        beta = cp.array(0.0, dtype=cp.float64)

        # Timing the operation using cp.cuda.Event
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Start timing
        start_event.record()

        # Perform the multiplication using cusparseSpMM
        cp.cuda.cusparse.spMM(
            cusparse_handle,
            cp.cuda.cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,  # operation on A
            cp.cuda.cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,  # operation on B
            alpha.data.ptr,  # scalar alpha
            spmat_descr,  # sparse matrix A descriptor
            dense_descr,  # dense matrix B descriptor
            beta.data.ptr,  # scalar beta
            output_descr,  # output matrix C descriptor
            cp.cuda.runtime.CUDA_R_64F,  # computation type
            cp.cuda.cusparse.CUSPARSE_MM_ALG_DEFAULT,  # algorithm choice
            cp.cuda.cusparse.CUSPARSE_MM_ALG_DEFAULT  # additional parameter if needed (set to default)
        )

        # Stop timing
        end_event.record()

        # Synchronize to ensure completion
        cp.cuda.Device().synchronize()

        output_numpy = cp.asnumpy(output_matrix)
        # Measure time
        elapsed_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        print("SpMM completed in {:.3f} ms".format(elapsed_time_ms))
        print("Result matrix shape:", output_matrix.shape)



        if np.allclose(C, C_ref, atol=1e-3):
            print(f"Test passed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx")
            passed += 1

        else:
            print(f"Test failed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx")
            print(f"Expected: {C_ref}")
            print(f"Got: {C}")


        if np.allclose(C, c_cuBLAS_ref, atol=1e-3):
            print(
                f"Test passed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx for cuBLAS with time {elapsed_time} ms")
            passed += 1

        else:
            print(
                f"Test failed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx for cuBLAS")
            print(f"Expected: {c_cuBLAS_ref}")
            print(f"Got: {C}")

        if np.allclose(C, output_numpy, atol=1e-3):
            print(
                f"Test passed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx for cuSPARSE with time {elapsed_time_ms} ms")
        else:
            print(
                f"Test failed for matrix A_sparse_{row_n}x512_{density:.2f}.mtx and B_sparse_512x{col}_1.0.mtx for cuSPARSE")
            print(f"Expected: {output_numpy}")
            print(f"Got: {C}")

        total += 1

print(f"Passed {passed}/{total} tests")

                                       