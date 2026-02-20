//
// Created by salehm32 on 2026-02-18.
//

#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>
#include "mm_loader.h"
#include "sptrsv_multi_gpu.cuh"
#include <chrono>

int main(int argc, char* argv[]) {

    cudaSetDevice(0);

    // --- Data Loading (All ranks load same file for simplicity) ---
    if (argc < 2) {
        std::cerr << "Usage: ./multi_gpu_test <matrix.mtx>" << std::endl;
    }
    std::string file_name = argv[1];

    // Load Matrix on host
    Eigen::SparseMatrix<float> spMat = load_matrix_market(file_name);
    auto triLMat = spMat.triangularView<Eigen::Lower>();
    Eigen::SparseMatrix<float, Eigen::ColMajor> spTriLMat = triLMat;
    spTriLMat.makeCompressed();

    int n = spTriLMat.rows();
    int nnz = spTriLMat.nonZeros();

    auto *csc = new CSCMatrix<float>();
    csc->n = n;
    csc->indptr = const_cast<int*>(spTriLMat.outerIndexPtr());
    csc->indices = const_cast<int*>(spTriLMat.innerIndexPtr());
    csc->data = const_cast<float*>(spTriLMat.valuePtr());

    // Supernode Conversion
    auto snMat = convert_csc_to_supernodal<float>(*csc);
    auto deps  = build_dependency_graph(snMat);
    int nnz_sn = snMat.x.size();

    // Prepare Device Memory
    float *d_x, *d_b;

    // Allocate Full Size on all ranks (simplification)
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    // Initialize X with 1.0 (RHS B implicit as 1.0 logic from main.cu)
    std::vector<float> h_x(n, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    //TODO: get num parts from arguments later.
    // Create Graph
    int num_parts = 4;
    cudaGraph_t graph = create_single_gpu_sptrsv_graph_for_part(
            snMat.supernodes.data(),
            snMat.ip.data(),
            snMat.i.data(),
            snMat.p.data(),
            snMat.x.data(),
            nullptr, // b is implicit in logic or unused in favor of x init
            d_x,
            n,
            deps,
            snMat.nSuper,
            nnz_sn,
            num_parts
    );

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Warmup & Run
    for(int i=0; i<3; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        // Simple barrier via NCCL just to keep steps aligned roughly (optional)
        // ncclAllReduce(...) usage prevents race if we loop fast?
        // Actually cudaGraphLaunch is async.
    }

    // Timer
    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<10; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Avg Time: " << (elapsed.count() / 10.0) * 1000.0 << " ms" << std::endl;

    // Validation (Gather X to Rank 0)
    // We need to copy d_x to host, then valid.
    std::vector<float> h_result(n);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));



    float sum = 0;
    for(float v : h_result) sum += v;
    std::cout << "Sum X: " << sum << std::endl;

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_x);
    return 0;
}
