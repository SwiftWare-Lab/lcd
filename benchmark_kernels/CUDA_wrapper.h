#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

void cpu_matrix_multiply(int A_num_rows, int A_num_cols, int B_num_cols, const float* A, const float* B, float* C);
void columnMajorToRowMajor(float* rowMajor, const float* colMajor, int rows, int cols);
extern "C" float benchmark_cublasSgemm(int A_num_rows, int A_num_cols, int B_num_cols,
                                       const float* hA, const float* hB, float* hC);
extern "C" float benchmark_cusparseSpMMCSR(int A_num_rows, int A_num_cols, int A_nnz, int B_num_cols,
                                const int* hA_csrOffsets, const int* hA_columns,
                               const float* hA_values, const float* hB,
                               float* hC);
bool compare_matrices(int rows, int cols, const float* A, const float* B, float tolerance);

#endif // CUDA_WRAPPER_H