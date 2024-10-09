#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

// A wrapper for CUDA APIs
// run using nvcc -o cuda_lib.so. CUDA_wrapper.cu -shared -lcusparse -lcublas -lcudart -Xcompiler -fPIC


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
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
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
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
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

extern "C" float benchmark_cublasSgemm(int A_num_rows, int A_num_cols, int B_num_cols,
                                       const float* hA, const float* hB, float* hC)
{
    // A: (A_num_rows x A_num_cols) matrix
    // B: (A_num_cols x B_num_cols) matrix
    // C: (A_num_rows x B_num_cols) matrix (result)

    int   lda = A_num_rows; // leading dimension of A
    int   ldb = A_num_cols; // leading dimension of B
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
                              CUBLAS_OP_N, CUBLAS_OP_N,   // No transpose for A and B
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
            std::cerr << "Mismatch at element " << i << ": " << A[i] << " != " << B[i] << "\n";
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
int main()
{
    const int A_num_rows = 3;
    const int A_num_cols = 3;
    const int B_num_cols = 3;

    // Allocate host memory for A, B, and C
    std::vector<float> hA(A_num_rows * A_num_cols);
    std::vector<float> hB(A_num_cols * B_num_cols);
    std::vector<float> hC(A_num_rows * B_num_cols);     // Result from cuBLAS
    std::vector<float> hC_cpu(A_num_rows * B_num_cols); // Result from CPU

    // Initialize matrices A and B with random values
    for (int i = 0; i < A_num_rows * A_num_cols; ++i) {
        hA[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < A_num_cols * B_num_cols; ++i) {
        hB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Print the input matrices
    std::cout << "Matrix A:\n";
    print_matrix(A_num_rows, A_num_cols, hA.data());
    std::cout << "Matrix B:\n";
    print_matrix(A_num_cols, B_num_cols, hB.data());

    // Run the cuBLAS SGEMM benchmark
    float time_cublas = benchmark_cublasSgemm(A_num_rows, A_num_cols, B_num_cols, hA.data(), hB.data(), hC.data());
    std::cout << "cuBLAS SGEMM execution time: " << time_cublas << " ms\n";

    // Compute the expected result using the CPU
    cpu_matrix_multiply(A_num_rows, A_num_cols, B_num_cols, hA.data(), hB.data(), hC_cpu.data());

    // Print the output matrices
    std::cout << "Matrix C (cuBLAS result):\n";
    print_matrix(A_num_rows, B_num_cols, hC.data());
    std::cout << "Matrix C (CPU result):\n";
    print_matrix(A_num_rows, B_num_cols, hC_cpu.data());

    // Compare the result from cuBLAS with the CPU result
    if (compare_matrices(A_num_rows, B_num_cols, hC.data(), hC_cpu.data(), 1e-5f)) {
        std::cout << "Test passed! The cuBLAS result matches the CPU result.\n";
    } else {
        std::cerr << "Test failed! The cuBLAS result does not match the CPU result.\n";
    }
}

