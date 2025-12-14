//
// Created by salehm32 on 2025-11-18.
//


#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <map>
#include "supernodal_utils.h"

// tiny macro for CUDA error checking
#define CUDA_CHECK(cmd) do {                                                \
    cudaError_t e = (cmd);                                                  \
    if (e != cudaSuccess)                                                   \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(e)); \
} while(0)

#define CUBLAS_CHECK(cmd) do {                                              \
    cublasStatus_t s = (cmd);                                               \
    if (s != CUBLAS_STATUS_SUCCESS)                                         \
        throw std::runtime_error(std::string("cuBLAS Error: ") + std::to_string(s)); \
} while(0)

__global__
void scatter_kernel(const float* __restrict__ update,
                    const int* __restrict__ row_idx,
                    float* __restrict__ x,
                    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d\n", tid);
    if (tid >= n) return;
    int r = row_idx[tid];
    atomicAdd(x + r, update[tid]);
}

cudaGraph_t sptrsv_csc_supernodal_wavefront_cudagraph(
        const int* sn_begin,
        const int* Lip,
        const int* Li,
        const int* Lp,
        const float* d_Lx,
        const float* d_b,
        float* d_x,
        const std::vector<int> wave_ptr,
        const std::vector<int> waves,
        int n,
        int nnz
        ){

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> joinEvents;
    for (int i = 0; i < wave_ptr[(int)wave_ptr.size()]; i++){
        cudaStream_t newStream;
        CUDA_CHECK(cudaStreamCreate(&newStream));
        streams.push_back(newStream);
        cudaEvent_t newEvent;
        CUDA_CHECK(cudaEventCreate(&newEvent));
        joinEvents.push_back(newEvent);
    }
    cudaEvent_t forkEvent;
    CUDA_CHECK(cudaEventCreate(&forkEvent));
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for (int w0 = wave_ptr[0]; w0 < (int)wave_ptr.size() - 1; ++w0) {
        CUDA_CHECK(cudaEventRecord(forkEvent));
        for (int w1 = wave_ptr[w0]; w1 < wave_ptr[w0 + 1]; ++w1) {
            CUDA_CHECK(cudaStreamWaitEvent(streams[w1], forkEvent));
            int sn_id = waves[w1];
            int col_b = sn_begin[sn_id];
            int col_e = sn_begin[sn_id + 1];
            int sup_width = col_e - col_b;
            int sup_len = Lip[sn_id + 1] - Lip[sn_id];
            int val_beg = Lp[col_b];

            int row_idx_start = Lip[sn_id] + sup_width;
            int row_idx_end   = Lip[sn_id] + 1;
            int n_scatter     = row_idx_end - row_idx_start;

            const float* diag_block_ptr = d_Lx + val_beg; // column-major block with lda = sup_len
            float* x_block_ptr = d_x + col_b;             // pointer to x[col_b]

            if (sup_width > 0) {

                CUBLAS_CHECK(cublasSetStream(cublas, streams[w1]));
                CUBLAS_CHECK(cublasStrsv(
                        cublas,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        sup_width,
                        diag_block_ptr,    // points to (0,0) of block
                        sup_len,           // lda: number of rows in full block (sup_len)
                        x_block_ptr,       // vector to be overwritten in-place
                        1));
            }


            if (sup_len > sup_width && n_scatter > 0) {

                const float* off_diag_ptr = diag_block_ptr + sup_width; // element (row=sup_width, col=0)


                float* d_update = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_update, sizeof(float) * n_scatter, stream));

                const float alpha = -1.0;
                const float beta  = 0.0;


                int m = n_scatter;
                int n = sup_width;

                CUBLAS_CHECK(cublasSetStream(cublas, streams[w1]));
                CUBLAS_CHECK(cublasSgemv(
                        cublas,
                        CUBLAS_OP_N,
                        m,        // rows of A
                        n,        // cols of A
                        &alpha,
                        off_diag_ptr,
                        sup_len,  // lda
                        x_block_ptr,
                        1,
                        &beta,
                        d_update,
                        1));


                const int* host_row_idx_begin = nullptr;

                int* d_row_idx = nullptr;
                if (n_scatter > 0) {
                    CUDA_CHECK(cudaMallocAsync(&d_row_idx, sizeof(int) * n_scatter, stream));
                    CUDA_CHECK(cudaMemcpyAsync(d_row_idx,
                                               &Li[row_idx_start],
                                               sizeof(int) * n_scatter,
                                               cudaMemcpyHostToDevice,
                                               stream));
                }


                const int threads_per_block = 256;
                const int blocks = (n_scatter + threads_per_block - 1) / threads_per_block;
                if (n_scatter > 0) {
                    scatter_kernel<<<blocks, threads_per_block, 0, stream>>>(d_update, d_row_idx, d_x, n_scatter);

                }

                if (d_row_idx) CUDA_CHECK(cudaFreeAsync(d_row_idx, stream));
                if (d_update)  CUDA_CHECK(cudaFreeAsync(d_update, stream));
            }
            CUDA_CHECK(cudaEventRecord(joinEvents[w1]));
        }
        for (int w1 = wave_ptr[w0]; w1 < wave_ptr[w0 + 1]; ++w1) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, joinEvents[w1]));
        }
    }
    cudaGraph_t g = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(stream, &g));
    CUBLAS_CHECK(cublasDestroy(cublas));
    for (auto s: streams){
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto je: joinEvents){
        CUDA_CHECK(cudaEventDestroy(je));
    }
    CUDA_CHECK(cudaEventDestroy(forkEvent));
    return g;
}

cudaGraph_t create_cuda_graph_from_super_nodes(
        const int* sn_begin,
        const int* Lip,
        const int* Li,
        const int* d_Li,
        const int* Lp,
        const float* d_Lx,
        const float* d_b,
        float* d_x,
        int n,
        std::vector<std::vector<int>> deps,
        int n_super,
        int nnz){
    std::map<int,int> sn_to_node_id;
    CudaDagBuilder builder;
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    for (int sn_id = 0; sn_id < n_super; sn_id++) {
        int col_b = sn_begin[sn_id];
        int col_e = sn_begin[sn_id + 1];
        int sup_width = col_e - col_b;
        int sup_len = Lip[sn_id + 1] - Lip[sn_id];
        int val_beg = Lp[col_b];

        int row_idx_start = Lip[sn_id] + sup_width;
        int row_idx_end   = Lip[sn_id + 1]; 
        int n_scatter     = row_idx_end - row_idx_start;

        const float* diag_block_ptr = d_Lx + val_beg;
        float* x_block_ptr = d_x + col_b;
        // Pointer to the specific indices for this supernode ALREADY ON DEVICE
        const int* d_current_row_indices = d_Li + row_idx_start;

        auto supernode_op = [=](cudaStream_t s) {
            cublasSetStream(cublas, s);
            // 1. Solve Diagonal Block
            if (sup_width > 0) {
                CUBLAS_CHECK(cublasStrsv(
                        cublas,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        sup_width,
                        diag_block_ptr,
                        sup_len,
                        x_block_ptr,
                        1));
            }

            // 2. Off-diagonal Update & Scatter
            if (sup_len > sup_width && n_scatter > 0) {
                const float *off_diag_ptr = diag_block_ptr + sup_width;
                float *d_update = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_update, sizeof(float) * n_scatter, s));

                const float alpha = -1.0f; 
                const float beta = 0.0f;

                // A * x_diag
                CUBLAS_CHECK(cublasSgemv(
                        cublas,
                        CUBLAS_OP_N,
                        n_scatter, // m
                        sup_width, // n
                        &alpha,
                        off_diag_ptr,
                        sup_len,   // lda
                        x_block_ptr,
                        1,
                        &beta,
                        d_update,
                        1));

                const int threads_per_block = 256;
                const int blocks = (n_scatter + threads_per_block - 1) / threads_per_block;
                
                // NO MEMCPY HERE. We use d_current_row_indices directly.
                scatter_kernel<<<blocks, threads_per_block, 0, s>>>(d_update, d_current_row_indices, d_x, n_scatter);
                
                CUDA_CHECK(cudaFreeAsync(d_update, s));
            }
        };
        int cuda_graph_node = builder.addNode("SuperNode_" + std::to_string(sn_id), supernode_op, {});
        sn_to_node_id[sn_id] = cuda_graph_node; 
    }
    for (int sn_id = 0; sn_id < n_super; sn_id++){
        std::vector<int>  deps_cg_ids;
        for (auto dep_sn_id: deps[sn_id]){
            if (dep_sn_id != sn_id) {
                deps_cg_ids.push_back(sn_to_node_id[dep_sn_id]);
            }
        }
        builder.addDependencies(sn_to_node_id[sn_id], deps_cg_ids);
    }
//    std::cout << "Capturing Graph..." << std::endl;
    cudaGraph_t graph = builder.captureGraph();
//    std::cout << "Capture Complete." << std::endl;
    std::ofstream out("graph.dot");
    out << builder.exportDot();
    out.close();
//    std::cout << "DOT file generated as 'graph.dot'.\n" << std::endl;
    return graph;
}
