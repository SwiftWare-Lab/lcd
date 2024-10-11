#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "CUDA_wrapper.h"
#include <iomanip>

using namespace std;

// Function to generate a sparse n by m matrix with a specific density, returned as a 1D vector
vector<float> generateSpMatrix(int n, int m, double density, float min_value = 1.0, float max_value = 10.0) {
    // Initialize a 1D vector to store the matrix in row-major order
    vector<float> matrix(n * m, 0);

    // Calculate the number of non-zero elements based on the density
    int total_elements = n * m;
    int non_zero_elements = static_cast<int>(density * total_elements);

    // Seed for randomness
    srand(static_cast<unsigned>(time(0)));

    // Randomly assign non-zero values to random positions in the 1D matrix
    for (int i = 0; i < non_zero_elements; ++i) {
        int row = rand() % n;
        int col = rand() % m;

        // Map 2D (row, col) to 1D index
        int index = row * m + col;

        // Generate a random value between min_value and max_value
        int value_i = rand() % 10 + 1;
        float value = static_cast<float>(value_i);

        // Ensure the chosen position is not already filled
        while (matrix[index] != 0) {
            row = rand() % n;
            col = rand() % m;
            index = row * m + col;
        }

        // Assign the random value to the 1D position
        matrix[index] = value;
    }

    return matrix;
}

// Function to generate a dense n by m matrix returned as a 1D vector
vector<float> generateDnMatrix(int n, int m, float min_value = 1.0, float max_value = 10.0) {
    // Initialize a 1D vector to store the matrix in row-major order
    vector<float> matrix(n * m);

    // Seed for randomness
    srand(static_cast<unsigned>(time(0)));

    // Fill the matrix with random values between min_value and max_value
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int value_i = rand() % 10 + 1;
            float value = static_cast<float>(value_i);
            matrix[i * m + j] = value;
        }
    }

    return matrix;
}

// Function to read .mtx files
void read_mtx_file(const std::string& filename, std::vector<int>& row_indices, std::vector<int>& col_indices, std::vector<float>& values, int& num_rows, int& num_cols, int& num_nonzeros) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    // Skip header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Read matrix dimensions and number of non-zero elements
    std::istringstream iss(line);
    iss >> num_rows >> num_cols >> num_nonzeros;

    // Resize vectors to hold the matrix data
    row_indices.resize(num_nonzeros);
    col_indices.resize(num_nonzeros);
    values.resize(num_nonzeros);

    // Read the matrix entries
    for (int i = 0; i < num_nonzeros; ++i) {
        int row, col;
        float value;
        file >> row >> col >> value;
        row_indices[i] = row - 1; // Convert to zero-based index
        col_indices[i] = col - 1; // Convert to zero-based index
        values[i] = value;
    }

    file.close();
}

// Function to write to .mtx files TODO: needs testing and fixing
void write_mtx_file(const std::string& filename, const std::vector<int>& row_indices, const std::vector<int>& col_indices, const std::vector<float>& values, int num_rows, int num_cols, int num_nonzeros) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << num_rows << " " << num_cols << " " << num_nonzeros << "\n";

    // Write the matrix entries
    for (int i = 0; i < num_nonzeros; ++i) {
        file << row_indices[i] + 1 << " " << col_indices[i] + 1 << " " << values[i] << "\n";
    }

    file.close();
}

// Function to compare the results TODO: needs testing and fixing
bool compare_results(const std::vector<float>& cpu_result, const std::vector<float>& gpu_result, float tolerance = 1e-5) {
    if (cpu_result.size() != gpu_result.size()) {
        return false;
    }

    for (size_t i = 0; i < cpu_result.size(); ++i) {
        if (std::fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

int main() {
    int passed = 0;
    int total = 0;
    int row_n = 768;
    std::vector<int> cols = {32, 64, 128, 256, 512};
    std::vector<float> densities = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    for (float density : densities) {
        for (int col : cols) {
            int A_num_rows = row_n;
            int A_num_cols = 512;

            int B_num_rows = 512;
            int B_num_cols = col;

            // Generate a sparse matrix
            vector<float> A = generateSpMatrix(row_n, 512, density);
            // Generate a dense matrix
            vector<float> B = generateDnMatrix(512, col);
            // Generate matrices to hold results from cuBLAS, cuSPARSE, and CPU
            std::vector<float> hC1(A_num_rows * B_num_cols);
            std::vector<float> hC2(A_num_rows * B_num_cols);
            std::vector<float> hCPU(A_num_rows * B_num_cols);


            std::vector<int> hA_csrOffsets(A_num_rows + 1);
            std::vector<int> hA_columns;
            std::vector<float> hA_values;

            int nnz = 0;
            for (int i = 0; i < A_num_rows; ++i) {
                for (int j = 0; j < A_num_cols; ++j) {
                    if (A[i * A_num_cols + j] != 0.0f) {
                        hA_columns.push_back(j);
                        hA_values.push_back(A[i * A_num_cols + j]);
                        nnz++;
                    }
                }
                hA_csrOffsets[i + 1] = nnz;
            }


            // Perform matrix multiplication on the CPU
            cpu_matrix_multiply(A_num_rows, A_num_cols, B_num_cols, A.data(), B.data(), hCPU.data());

            // Perform GPU computations
            float cuSPARSE_time = benchmark_cusparseSpMMCSR(A_num_rows, A_num_cols, nnz, B_num_cols,
                                                            hA_csrOffsets.data(), hA_columns.data(),
                                                            hA_values.data(), B.data(), hC1.data());

            float cuBLAS_time = benchmark_cublasSgemm(A_num_rows, A_num_cols, B_num_cols, A.data(), B.data(), hC2.data());

            // Convert cuBLAS result from column-major to row-major
            float* hC_temp = new float[A_num_rows * B_num_cols];
            columnMajorToRowMajor(hC_temp, hC2.data(), A_num_rows, B_num_cols);

            // Compare the result from cuSPARSE with the CPU result
            if (compare_matrices(A_num_rows, B_num_cols, hC1.data(), hCPU.data(), 1e-5f)) {
                std::cout << "Test passed! The cuSPARSE result matches the CPU result.\n";
                passed += 1;
            } else {
                std::cerr << "Test failed! The cuSPARSE result does not match the CPU result.\n";
            }
            
            // Compare the result from cuBLAS with the CPU result
            if (compare_matrices(A_num_rows, B_num_cols, hC_temp, hCPU.data(), 1e-2f)) {
                std::cout << "Test passed! The cuBLAS result matches the CPU result.\n";
                passed += 1;
            } else {
                std::cerr << "Test failed! The cuBLAS result does not match the CPU result.\n";
            }

            total += 2;
        }
    }
    std::cout << "Passed " << passed << " out of " << total << " tests.\n";
    return 0;
}

//nvcc -c -o CUDA_wrapper.o CUDA_wrapper.cu -lcusparse -lcublas -lcudart
//nvcc -c -o test_cuda_kernels.o test_cuda_kernels.cu -lcusparse -lcublas -lcudart
//nvcc -o test_cuda_kernels CUDA_wrapper.o test_cuda_kernels.o -lcusparse -lcublas -lcudart