#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "CUDA_wrapper.h"
#include <iomanip>

// A wrapper for CUDA APIs
// run using nvcc -o cuda_lib.so. CUDA_wrapper.cu -shared -lcusparse -lcublas -lcudart -Xcompiler -fPIC

void columnMajorToRowMajor(float* rowMajor, const float* colMajor, int rows, int cols) {
    // Loop through the matrix elements in column-major order
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            // Calculate the index for column-major storage
            int colMajorIndex = col * rows + row;
            // Calculate the index for row-major storage
            int rowMajorIndex = row * cols + col;
            // Copy the element from column-major to row-major order
            rowMajor[rowMajorIndex] = colMajor[colMajorIndex];
        }
    }
}


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA API failed at line " << __LINE__ << " with error: " \
                  << cudaGetErrorString(status) << " (" << status << ")\n";    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "CUSPARSE API failed at line " << __LINE__ << " with error: " \
                  << cusparseGetErrorString(status) << " (" << status << ")\n";\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS API failed at line " << __LINE__ << " with error: " \
                  << status << "\n";                                           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

extern "C" float benchmark_cusparseSpMMCSR(int A_num_rows, int A_num_cols, int A_nnz, int B_num_cols,
                                const int* hA_csrOffsets, const int* hA_columns,
                               const float* hA_values, const float* hB,
                               float* hC)
{
    int   B_num_rows      = A_num_cols;
    int   ldb             = B_num_cols; // leading dimension of B
    int   ldc             = B_num_cols; // leading dimension of C
    int   B_size          = A_num_cols * B_num_cols;
    int   C_size          = A_num_rows * B_num_cols;
    float alpha           = 1.0f;
    float beta            = 0.0f;


    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )

    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,        
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Get matrix C values and copy them back to CPU
    int64_t C_num_rows_attr, C_num_cols_attr, C_ld_attr;
    void* C_values_attr;
    cudaDataType C_type_attr;
    cusparseOrder_t C_order_attr;
    CHECK_CUSPARSE( cusparseDnMatGet(matC, &C_num_rows_attr, &C_num_cols_attr, &C_ld_attr, &C_values_attr, &C_type_attr, &C_order_attr) )

    // Copy C values to from GPU to CPU
    CHECK_CUDA( cudaMemcpy(hC, C_values_attr, C_size * sizeof(float), cudaMemcpyDeviceToHost) )

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    
    return milliseconds;
}

extern "C" float* test_cusparseSpMMCSR(int A_num_rows, int A_num_cols, int A_nnz, int B_num_cols,
                                const int* hA_csrOffsets, const int* hA_columns,
                               const float* hA_values, const float* hB,
                               float* hC)
{
    int   B_num_rows      = A_num_cols;
    int   ldb             = B_num_cols; // leading dimension of B
    int   ldc             = B_num_cols; // leading dimension of C
    int   B_size          = A_num_cols * B_num_cols;
    int   C_size          = A_num_rows * B_num_cols;
    float alpha           = 1.0f;
    float beta            = 0.0f;


    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )

    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,        
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Get matrix C values and copy them back to CPU
    int64_t C_num_rows_attr, C_num_cols_attr, C_ld_attr;
    void* C_values_attr;
    cudaDataType C_type_attr;
    cusparseOrder_t C_order_attr;
    CHECK_CUSPARSE( cusparseDnMatGet(matC, &C_num_rows_attr, &C_num_cols_attr, &C_ld_attr, &C_values_attr, &C_type_attr, &C_order_attr) )

    // Copy C values to from GPU to CPU
    CHECK_CUDA( cudaMemcpy(hC, C_values_attr, C_size * sizeof(float), cudaMemcpyDeviceToHost) )

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    
    return hC;
}

extern "C" float benchmark_cublasSgemm(int A_num_rows, int A_num_cols, int B_num_cols,
                                       const float* hA, const float* hB, float* hC)
{
    // A: (A_num_rows x A_num_cols) matrix
    // B: (A_num_cols x B_num_cols) matrix
    // C: (A_num_rows x B_num_cols) matrix (result)

    int   lda = A_num_cols; // leading dimension of A
    int   ldb = B_num_cols; // leading dimension of B
    int   ldc = A_num_rows; // leading dimension of C
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Device memory management
    float *dA, *dB, *dC;
    size_t size_A = A_num_rows * A_num_cols * sizeof(float);
    size_t size_B = A_num_cols * B_num_cols * sizeof(float);
    size_t size_C = A_num_rows * B_num_cols * sizeof(float);

    CHECK_CUDA( cudaMalloc((void**)&dA, size_A) )
    CHECK_CUDA( cudaMalloc((void**)&dB, size_B) )
    CHECK_CUDA( cudaMalloc((void**)&dC, size_C) )

    CHECK_CUDA( cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, size_C, cudaMemcpyHostToDevice) )

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS( cublasCreate(&handle) )

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Perform SGEMM using cuBLAS
    // C = alpha * A * B + beta * C
    CHECK_CUBLAS( cublasSgemm(handle,
                              CUBLAS_OP_T, CUBLAS_OP_T,   // No transpose for A and B
                              A_num_rows,                 // number of rows in A and C
                              B_num_cols,                 // number of columns in B and C
                              A_num_cols,                 // number of columns in A and rows in B
                              &alpha,                     // scalar alpha
                              dA, lda,                    // A matrix and leading dimension
                              dB, ldb,                    // B matrix and leading dimension
                              &beta,                      // scalar beta
                              dC, ldc))                   // C matrix and leading dimension

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Convert the result from column-major to row-major
    float* hC_result = new float[A_num_rows * B_num_cols];
    float* hC_temp = new float[A_num_rows * B_num_cols];
    CHECK_CUDA( cudaMemcpy(hC_result, dC, size_C, cudaMemcpyDeviceToHost) )
    columnMajorToRowMajor(hC_temp, hC_result, A_num_rows, B_num_cols);

    // Copy hC_temp to hC
    for (int i = 0; i < A_num_rows * B_num_cols; ++i) {
        hC[i] = hC_temp[i];
    }

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy the result back to the host
    CHECK_CUDA( cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost) )

    // Free device memory
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )

    // Destroy cuBLAS handle and events
    CHECK_CUBLAS( cublasDestroy(handle) )
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds;
}

// A simple matrix multiplication function on the CPU for verification purposes
void cpu_matrix_multiply(int A_num_rows, int A_num_cols, int B_num_cols, const float* A, const float* B, float* C)
{
    for (int i = 0; i < A_num_rows; ++i) {
        for (int j = 0; j < B_num_cols; ++j) {
            C[i * B_num_cols + j] = 0;
            for (int k = 0; k < A_num_cols; ++k) {
                C[i * B_num_cols + j] += A[i * A_num_cols + k] * B[k * B_num_cols + j];
            }
        }
    }
}

// Function to compare two matrices and check if they are close enough
bool compare_matrices(int rows, int cols, const float* A, const float* B, float tolerance)
{
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " != " << B[i] << std::endl;
            std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1) << A[i] << std::endl;
            std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1) << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Function to print a matrix
void print_matrix(int rows, int cols, const float* A)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << A[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

// Tester function that calls the cuBLAS benchmark and verifies the result against the CPU version
// compile using nvcc CUDA_wrapper.cu -o out -lcusparse -lcublas