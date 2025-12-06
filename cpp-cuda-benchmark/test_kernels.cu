//
// Created by salehm32 on 2025-11-18.
//

#include "mm_loader.h"
#include "supernodal_utils.h"
#include "sptrsv_supernodal_cuda_graph.cuh"
#include <cusparse.h>
#include <cuda_runtime.h>

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

void sptrsv_csc_cpu(int* Ap, int* Ai, float* Ax, float *b, float* x, int n){
    for (int i = 0; i < n; i++){
        x[i] = (b[i] - x[i]) / Ax[Ap[i]];
        for (int j = Ap[i] + 1; j < Ap[i+1]; j++){
            x[Ai[j]] += Ax[j] * x[i];
        }
    }
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
    Eigen::SparseMatrix<float, Eigen::ColMajor> spTriLMat= triLMat;
    spTriLMat.makeCompressed();

    auto rowPtr = const_cast<int*>(spTriLMat.innerIndexPtr()); // Row indices
    auto colInd = const_cast<int*>(spTriLMat.outerIndexPtr()); // Col pointers
    auto values = const_cast<float*>(spTriLMat.valuePtr());    // Values
    int n = spTriLMat.rows();
    int nnz_total = spTriLMat.nonZeros();

    auto* cscMatrix =  new CSCMatrix<float>();
    cscMatrix->n = n;
    cscMatrix->indptr = colInd;
    cscMatrix->indices = rowPtr;
    cscMatrix->data = values;
    
    auto *b = new float[n];
    auto *x = new float[n];
    for (int i = 0; i < n; i++){
        b[i] = 1;
        x[i] = 1;
    }
    auto snMat = convert_csc_to_supernodal<float>(*cscMatrix);
    auto deps = build_dependency_graph(snMat);
    int nnz_sn = snMat.x.size();
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_Lx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Lx, sizeof(float) * nnz_sn));
    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz_sn * sizeof(float), cudaMemcpyHostToDevice));
    auto cudaGraph = create_cuda_graph_from_super_nodes(
            snMat.supernodes.data(),
            snMat.ip.data(),
            snMat.i.data(),
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

    std::cout << "Launching Custom Graph..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start_custom));
    CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    CUDA_CHECK(cudaEventRecord(stop_custom));
    CUDA_CHECK(cudaEventSynchronize(stop_custom));

    float ms_custom = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_custom, start_custom, stop_custom));

    // Retrieve Custom Results
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));


    // --- 4. Benchmark cuSPARSE ---
    std::cout << "Preparing cuSPARSE..." << std::endl;
    
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matL;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Allocate Device Memory for standard CSC format (cuSPARSE needs this)
    int *d_cscColPtr, *d_cscRowIdx;
    float *d_cscVal, *d_y_cusparse, *d_x_cusparse;

    CUDA_CHECK(cudaMalloc((void**)&d_cscColPtr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cscRowIdx, nnz_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cscVal, nnz_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y_cusparse, n * sizeof(float))); // Input b
    CUDA_CHECK(cudaMalloc((void**)&d_x_cusparse, n * sizeof(float))); // Output x

    CUDA_CHECK(cudaMemcpy(d_cscColPtr, colInd, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cscRowIdx, rowPtr, nnz_total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cscVal, values, nnz_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_cusparse, b, n * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_x_cusparse, 0, n * sizeof(float))); 

    // Create Sparse Matrix Descriptor (CSC)
    CUSPARSE_CHECK(cusparseCreateCsc(&matL, n, n, nnz_total,
                                     d_cscColPtr, d_cscRowIdx, d_cscVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Specify Fill Mode (Lower) and Unit/Non-Unit Diagonal
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit)));

    // Create Dense Vector Descriptors
    // SpSV solves: op(A) * x = alpha * y
    // Here: L * x_out = b_in
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n, d_y_cusparse, CUDA_R_32F)); // b (input)
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x_cusparse, CUDA_R_32F)); // x (output)

    float alpha = 1.0f;
    cusparseSpSVDescr_t spsvDescr;
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsvDescr));

    // Get Buffer Size
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecX, vecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Analysis Phase
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecX, vecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer));

    // Solve Phase (Timed)
    cudaEvent_t start_cusp, stop_cusp;
    CUDA_CHECK(cudaEventCreate(&start_cusp));
    CUDA_CHECK(cudaEventCreate(&stop_cusp));

    std::cout << "Launching cuSPARSE SpSV..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start_cusp));
    CUSPARSE_CHECK(cusparseSpSV_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecX, vecY, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));
    CUDA_CHECK(cudaEventRecord(stop_cusp));
    CUDA_CHECK(cudaEventSynchronize(stop_cusp));

    float ms_cusp = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cusp, start_cusp, stop_cusp));

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "Custom Graph Time: " << ms_custom << " ms" << std::endl;
    std::cout << "cuSPARSE Time:     " << ms_cusp << " ms" << std::endl;
    std::cout << "Speedup (vs cuSPARSE): " << ms_cusp / ms_custom << "x" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // --- 5. CPU Validation (Existing) ---
    auto *xCpu = new float[n];
    for (int i = 0; i < n; i++){
        xCpu[i] = 0;
    }

    sptrsv_csc_cpu(cscMatrix->indptr, cscMatrix->indices, cscMatrix->data, b, xCpu, n);
    auto atol = 1e-5;
    int count  = 0;
    for(int i = 0; i < n; i++){
        if (abs(x[i] - xCpu[i]) > atol){
            count++;
        }
    }

    std::cout << "Validation (Custom vs CPU):" << std::endl;
    std::cout << "Total rows: " << spTriLMat.rows() << "\nIncorrect count: " << count << std::endl;

    // --- Cleanup ---
    cusparseDestroySpMat(matL);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseSpSV_destroyDescr(spsvDescr);
    
    cusparseDestroy(handle);
    cudaFree(d_x); cudaFree(d_Lx);
    cudaFree(d_cscColPtr); cudaFree(d_cscRowIdx); cudaFree(d_cscVal);
    cudaFree(d_y_cusparse); cudaFree(d_x_cusparse); cudaFree(dBuffer);
    delete[] b; delete[] x; delete[] xCpu;

    return 0;
}