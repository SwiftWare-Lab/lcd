//
// Created by salehm32 on 2025-11-18.
//

#include "mm_loader.h"
#include "supernodal_utils.h"
#include "sptrsv_supernodal_cuda_graph.cuh"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

// Macro for checking cuSPARSE errors
#define CUSPARSE_CHECK(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define RUN_BENCHMARK_AVG(iters, CODE_BLOCK, avg_ms_out)                     \
{                                                                         \
    float __total_ms = 0.0f;                                                 \
                                                                              \
    cudaEvent_t __start, __stop;                                             \
    CUDA_CHECK(cudaEventCreate(&__start));                                   \
    CUDA_CHECK(cudaEventCreate(&__stop));                                    \
                                                                              \
    for (int __i = 0; __i < (iters); ++__i) {                                \
        CUDA_CHECK(cudaEventRecord(__start));                                \
                                                                              \
        CODE_BLOCK                                                            \
                                                                              \
        CUDA_CHECK(cudaEventRecord(__stop));                                 \
        CUDA_CHECK(cudaEventSynchronize(__stop));                            \
                                                                              \
        float __ms = 0.0f;                                                   \
        CUDA_CHECK(cudaEventElapsedTime(&__ms, __start, __stop));            \
        __total_ms += __ms;                                                  \
    }                                                                        \
                                                                              \
    CUDA_CHECK(cudaEventDestroy(__start));                                   \
    CUDA_CHECK(cudaEventDestroy(__stop));                                    \
                                                                              \
    (avg_ms_out) = __total_ms / (iters);                                     \
}


int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "Usage: ./benchmark <matrix_market_file>" << std::endl;
        return 1;
    }
    std::string file_name = argv[1];

    // --- 1. Load and Prepare Matrix (Host) ---
    Eigen::SparseMatrix<float> spMat = load_matrix_market(file_name);
    auto triLMat = spMat.triangularView<Eigen::Lower>();
    Eigen::SparseMatrix<float, Eigen::ColMajor> spTriLMat = triLMat;
    spTriLMat.makeCompressed();

    auto rowPtr = const_cast<int*>(spTriLMat.innerIndexPtr()); // Row indices
    auto colInd = const_cast<int*>(spTriLMat.outerIndexPtr()); // Col pointers
    auto values = const_cast<float*>(spTriLMat.valuePtr());    // Values
    int n = spTriLMat.rows();
    int nnz_total = spTriLMat.nonZeros();

    auto* cscMatrix = new CSCMatrix<float>();
    cscMatrix->n      = n;
    cscMatrix->indptr = colInd;
    cscMatrix->indices= rowPtr;
    cscMatrix->data   = values;

    auto *b = new float[n];
    auto *x = new float[n];
    for (int i = 0; i < n; i++){
        b[i] = 1.0f;
        x[i] = 1.0f;
    }

    auto snMat = convert_csc_to_supernodal<float>(*cscMatrix);
    auto deps  = build_dependency_graph(snMat);
    int nnz_sn = snMat.x.size();

    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_Lx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Lx, sizeof(float) * nnz_sn));
    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz_sn * sizeof(float), cudaMemcpyHostToDevice));

    int* d_Li = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Li, snMat.i.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_Li, snMat.i.data(), snMat.i.size() * sizeof(int), cudaMemcpyHostToDevice));

    auto cudaGraph = create_cuda_graph_from_super_nodes(
            snMat.supernodes.data(),
            snMat.ip.data(),
            snMat.i.data(),
            d_Li,
            snMat.p.data(),
            d_Lx,
            b,
            d_x,
            n,
            deps,
            snMat.nSuper,
            nnz_sn
    );

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, cudaGraph, NULL, NULL, 0));

    // --- 3. Benchmark Custom Graph ---
    cudaEvent_t start_custom, stop_custom;
    CUDA_CHECK(cudaEventCreate(&start_custom));
    CUDA_CHECK(cudaEventCreate(&stop_custom));

//    std::cout << "Launching Custom Graph..." << std::endl;

    auto avg_cudagraph = 0.f;
    RUN_BENCHMARK_AVG(
            10,
            {
                CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
            },
            avg_cudagraph
    );

    // Retrieve Custom Results
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cudagraph," << avg_cudagraph << "," << file_name << "," << cscMatrix->n << "," << nnz_total << std::endl;

    // --- 4. Benchmark cuSPARSE ---
 //   std::cout << "Preparing cuSPARSE..." << std::endl;

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matU_CSR;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Allocate Device Memory for matrix + vectors
    int   *d_cscColPtr, *d_cscRowIdx;
    float *d_cscVal, *d_y_cusparse, *d_x_cusparse;

    CUDA_CHECK(cudaMalloc((void**)&d_cscColPtr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cscRowIdx, nnz_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cscVal,    nnz_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y_cusparse, n * sizeof(float))); // will be solution or RHS depending on mapping
    CUDA_CHECK(cudaMalloc((void**)&d_x_cusparse, n * sizeof(float))); // will be solution (by our choice)

    CUDA_CHECK(cudaMemcpy(d_cscColPtr, colInd,  (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cscRowIdx, rowPtr,  nnz_total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cscVal,    values,  nnz_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_cusparse, b, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x_cusparse, 0, n * sizeof(float)));

    CUSPARSE_CHECK(cusparseCreateCsr(&matU_CSR,
                                     n,              // rows
                                     n,              // cols
                                     nnz_total,      // nnz
                                     d_cscColPtr,    // rowOffsets (CSC colPtr)
                                     d_cscRowIdx,    // colIndices (CSC rowIdx)
                                     d_cscVal,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    // Tell cuSPARSE that this is structurally UPPER (since L^T = U)
    cusparseFillMode_t fill_upper   = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit= CUSPARSE_DIAG_TYPE_NON_UNIT;

    CUSPARSE_CHECK(cusparseSpMatSetAttribute(
            matU_CSR, CUSPARSE_SPMAT_FILL_MODE,
            &fill_upper, sizeof(fill_upper)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(
            matU_CSR, CUSPARSE_SPMAT_DIAG_TYPE,
            &diag_non_unit, sizeof(diag_non_unit)));

    float alpha = 1.0f;
    cusparseSpSVDescr_t spsvDescr;
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsvDescr));

    // We solve L x = b by solving U^T x = b  (U = L^T).
    // SpSV semantics: op(A) * y = alpha * x
    // We want op(A)=U^T, y = solution x, x = RHS b.
    cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;

    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_y_cusparse, CUDA_R_32F)); // x = RHS = b
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n, d_x_cusparse, CUDA_R_32F)); // y = solution

    // Get Buffer Size
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            handle, op,
            &alpha, matU_CSR, vecX, vecY,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr,
            &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Analysis
    CUSPARSE_CHECK(cusparseSpSV_analysis(
            handle, op,
            &alpha, matU_CSR, vecX, vecY,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr,
            dBuffer));

    // Solve
    float avg_cusparse = 0.0f;
    RUN_BENCHMARK_AVG(
            50,
            {
                CUSPARSE_CHECK(cusparseSpSV_solve(
                        handle, op,
                        &alpha, matU_CSR, vecX, vecY,
                        CUDA_R_32F,
                        CUSPARSE_SPSV_ALG_DEFAULT,
                        spsvDescr));
            },
            avg_cusparse
    );
    std::cout << "cusparse,"<< avg_cusparse << "," << file_name << "," << cscMatrix->n << "," << nnz_total << std::endl;

//    std::cout << "------------------------------------------" << std::endl;
//    std::cout << "Performance Results:" << std::endl;
//    std::cout << "Custom Graph Time: " << ms_custom << " ms" << std::endl;
//    std::cout << "cuSPARSE Time:     " << ms_cusp   << " ms" << std::endl;
//    std::cout << "Speedup (vs cuSPARSE): " << ms_cusp / ms_custom << "x" << std::endl;
//    std::cout << "------------------------------------------" << std::endl;


    // --- Cleanup ---
    cusparseDestroySpMat(matU_CSR);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseSpSV_destroyDescr(spsvDescr);
    cusparseDestroy(handle);

    cudaFree(d_x);
    cudaFree(d_Lx);
    cudaFree(d_Li);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);
    cudaFree(d_y_cusparse);
    cudaFree(d_x_cusparse);
    cudaFree(dBuffer);

    delete[] b;
    delete[] x;

    return 0;
}

