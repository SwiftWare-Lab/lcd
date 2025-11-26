#include "cuda_dag_builder_viz.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <iostream>
#include <fstream>
#include <vector>

// Simple error helper
void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    // 1. Setup Resources
    CudaDagBuilder builder;
    int N = 8192; 

    // --- Data Allocation ---
    float *d_DenseA, *d_VecX, *d_VecY;
    check(cudaMalloc(&d_DenseA, N * N * sizeof(float)), "Malloc A");
    check(cudaMalloc(&d_VecX, N * sizeof(float)), "Malloc X");
    check(cudaMalloc(&d_VecY, N * sizeof(float)), "Malloc Y");

    // Initialize with dummy data (0s and 1s)
    cudaMemset(d_DenseA, 0, N*N*sizeof(float));
    cudaMemset(d_VecX, 1, N*sizeof(float));
    cudaMemset(d_VecY, 0, N*sizeof(float));

    // --- Library Handles ---
    cublasHandle_t h_blas;
    cublasCreate(&h_blas);
    
    cusparseHandle_t h_sparse;
    cusparseCreate(&h_sparse);

    // --- Sparse Matrix Setup (Diagonal Identity in CSR) ---
    // We need this to make the cusparse node actually runnable
    std::vector<int> h_rowOffsets(N + 1);
    std::vector<int> h_colIndices(N);
    std::vector<float> h_values(N, 1.0f);
    
    for (int i = 0; i <= N; i++) h_rowOffsets[i] = i;
    for (int i = 0; i < N; i++) h_colIndices[i] = i;

    int *d_rowOffsets, *d_colIndices;
    float *d_values;
    cudaMalloc(&d_rowOffsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_colIndices, N * sizeof(int));
    cudaMalloc(&d_values, N * sizeof(float));

    cudaMemcpy(d_rowOffsets, h_rowOffsets.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndices, h_colIndices.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Create Descriptors
    cusparseSpMatDescr_t matDescr;
    cusparseCreateCsr(&matDescr, N, N, N, 
                      d_rowOffsets, d_colIndices, d_values, 
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, d_VecX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, d_VecY, CUDA_R_32F);

    // Workspace for SpMV
    size_t bufferSize = 0;
    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMV_bufferSize(h_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matDescr, vecX, &beta, vecY, 
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    void* d_spmvBuffer;
    cudaMalloc(&d_spmvBuffer, bufferSize);


    // --- 2. Define Node Operations ---

    // Op A: Dense Matrix-Matrix Mul (using cuBLAS)
    auto denseGemmOp = [&](cudaStream_t s){ 
        cublasSetStream(h_blas, s);
        // C = A * B (Using VecX and VecY as dummy matrices for demo)
        float a = 1.0f, b = 0.0f;
        cublasSgemm(h_blas, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, 1, N, // Treating as Nx1 matrix
                    &a, d_DenseA, N, d_VecX, N, &b, d_VecY, N);
    };

    // Op B: Sparse Matrix-Vector Mul (using cuSPARSE)
    auto sparseSpMVOp = [&](cudaStream_t s){ 
        cusparseSetStream(h_sparse, s);
        float a = 1.0f, b = 1.0f;
        cusparseSpMV(h_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                     &a, matDescr, vecX, &b, vecY, 
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuffer);
    };

    // --- 3. Build the Mixed DAG ---
    // Structure:
    //       /--> Sparse_B --\
    // GEMM_A                 |--> GEMM_D
    //       \--> GEMM_C ----/
    
    int nA = builder.addNode("Cublas_GEMM_Start", denseGemmOp, {});
    
    int nB = builder.addNode("Cusparse_SpMV", sparseSpMVOp, {nA});
    int nC = builder.addNode("Cublas_GEMM_Branch", denseGemmOp, {nA});
    
    int nD = builder.addNode("Cublas_GEMM_Merge", denseGemmOp, {nB, nC});

    // Independent Parallel Work
    int nE = builder.addNode("Indep_Sparse_Head", sparseSpMVOp, {});
    int nF = builder.addNode("Indep_Dense_Tail", denseGemmOp, {nE});


    // --- 4. Capture & Visualize ---
    std::cout << "Capturing Graph..." << std::endl;
    cudaGraph_t graph = builder.captureGraph();
    std::cout << "Capture Complete." << std::endl;

    std::ofstream out("graph.dot");
    out << builder.exportDot();
    out.close();
    std::cout << "DOT file generated as 'graph.dot'.\n" << std::endl;
    
    // --- 5. Run Loop ---
    std::cout << "Instantiating Graph..." << std::endl;
    cudaGraphExec_t graphExec;
    check(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "Instantiate");

    cudaStream_t runStream;
    cudaStreamCreate(&runStream);

    std::cout << "Launching Graph 100 times..." << std::endl;
    for(int i = 0; i < 100; i++) {
        check(cudaGraphLaunch(graphExec, runStream), "Launch");
    }

    // Wait for all 100 iterations to finish and check for execution errors
    check(cudaStreamSynchronize(runStream), "Graph Execution / Stream Sync");
    std::cout << "Execution Finished Successfully." << std::endl;

    // --- Cleanup ---
    cudaStreamDestroy(runStream);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cublasDestroy(h_blas);
    cusparseDestroy(h_sparse);
    return 0;
}
