#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "cuda_dag_builder_viz.h"
#include "sptrsv_supernodal_cuda_graph.cuh"

// Error handle for NCCL
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Function to partition supernodes among ranks
// For simplicity, we use 1D block partitioning.
inline void get_partition_range(int n_super, int rank, int n_ranks, int& start, int& end) {
    int chunk = (n_super + n_ranks - 1) / n_ranks;
    start = std::min(rank * chunk, n_super);
    end = std::min((rank + 1) * chunk, n_super);
}

// Multi-GPU SpTRSV Graph Construction
cudaGraph_t create_multi_gpu_sptrsv_graph(
    const int* sn_begin,
    const int* Lip,
    const int* Li,
    const int* d_Li,  // Device pointer to row indices
    const int* Lp,
    const float* d_Lx,
    const float* d_b,
    float* d_x,       // Solution vector (device)
    int n,            // Total rows
    std::vector<std::vector<int>>& deps, // Dependencies (host)
    int n_super,
    int nnz_sn,
    int rank,
    int n_ranks,
    ncclComm_t comm,
    cudaStream_t stream // Stream for NCCL setup if needed
) {
    CudaDagBuilder builder;
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    int my_start_sn, my_end_sn;
    get_partition_range(n_super, rank, n_ranks, my_start_sn, my_end_sn);
    
    // Determine dependencies from previous ranks
    // Ideally, we find exactly which *values* we need.
    // For this prototype: 
    // - Rank K receives the Full X vector updates from Rank 0...K-1.
    // - Rank K computes.
    // - Rank K sends its updates to Rank K+1...N-1.
    // This serializes the blocks but uses DAG locally.
    
    // Note: To make this "Graph compliant", we encapsulate NCCL in nodes.
    
    // 1. Receive Node (from previous rank)
    int recv_node_id = -1;
    if (rank > 0) {
        auto recv_op = [=](cudaStream_t s) {
            // Receive the entire X vector (or relevant part) from rank-1
            // Simple chain: Rank 0 -> Rank 1 -> Rank 2
            // Receive from Rank - 1
            NCCLCHECK(ncclRecv(d_x, n, ncclFloat, rank - 1, comm, s));
        };
        recv_node_id = builder.addNode("NCCL_Recv", recv_op, {});
    }

    // 2. Compute Nodes (Local Supernodes)
    std::map<int, int> sn_to_node_id;
    
    for (int sn_id = my_start_sn; sn_id < my_end_sn; sn_id++) {
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
        const int* d_current_row_indices = d_Li + row_idx_start;

        // Same logic as single GPU
        auto supernode_op = [=](cudaStream_t s) {
            cublasSetStream(cublas, s);
            if (sup_width > 0) {
                // Diagonal solve
                CUBLAS_CHECK(cublasStrsv(
                    cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    sup_width, diag_block_ptr, sup_len, x_block_ptr, 1));
            }
            if (sup_len > sup_width && n_scatter > 0) {
                // Scatter update
                const float *off_diag_ptr = diag_block_ptr + sup_width;
                float *d_update = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_update, sizeof(float) * n_scatter, s)); 
                const float alpha = -1.0f; const float beta = 0.0f;
                
                CUBLAS_CHECK(cublasSgemv(
                    cublas, CUBLAS_OP_N, n_scatter, sup_width, &alpha,
                    off_diag_ptr, sup_len, x_block_ptr, 1, &beta, d_update, 1));

                const int threads_per_block = 256;
                const int blocks = (n_scatter + threads_per_block - 1) / threads_per_block;
                scatter_kernel<<<blocks, threads_per_block, 0, s>>>(d_update, d_current_row_indices, d_x, n_scatter);
                CUDA_CHECK(cudaFreeAsync(d_update, s));
            }
        };

        // Determine dependencies
        std::vector<int> graph_deps;
        if (recv_node_id != -1) {
            // Implicitly depend on Recv if we depend on any REMOTE node.
            // For block partition, ANY dependency < my_start_sn implies remote dependency.
            bool depends_on_remote = false;
            for (int dep_sn : deps[sn_id]) {
                if (dep_sn < my_start_sn) {
                    depends_on_remote = true;
                    break;
                }
            }
            if (depends_on_remote) {
                graph_deps.push_back(recv_node_id);
            }
        }
        
        // Add Local Dependencies
        for (int dep_sn : deps[sn_id]) {
             if (dep_sn >= my_start_sn && dep_sn < my_end_sn) {
                 if (sn_to_node_id.count(dep_sn)) {
                     graph_deps.push_back(sn_to_node_id[dep_sn]);
                 }
             }
        }

        int node_id = builder.addNode("SuperNode_" + std::to_string(sn_id), supernode_op, graph_deps);
        sn_to_node_id[sn_id] = node_id;
    }

    // 3. Send Node (to next rank)
    if (rank < n_ranks - 1) {
        auto send_op = [=](cudaStream_t s) {
            // Send entire X (updated) to next rank
            NCCLCHECK(ncclSend(d_x, n, ncclFloat, rank + 1, comm, s));
        };
        
        // Send depends on ALL local computations
        // We can create a "Barrier" node or just add dependencies from all leaves.
        // Or simpler: Make it depend on the last few nodes?
        // Correctness: Must depend on ANY node that writes to X.
        // For simplicity: Add explicit dependencies from ALL local nodes? That's too many edges.
        // Better: CudaDagBuilder doesn't have a "Global Barrier".
        // Alternative: Create a dummy node that depends on everything?
        // Let's iterate and find nodes with 0 children? Or just all nodes?
        // Optimization: Just ensure it runs after the LAST scheduled logical node.
        
        std::vector<int> all_local_nodes;
        for(auto& p : sn_to_node_id) all_local_nodes.push_back(p.second);
        
        // Ideally we only depend on sinks. But finding sinks requires parsing deps. 
        // Let's try adding all. CudaGraph can handle it.
        builder.addNode("NCCL_Send", send_op, all_local_nodes);
    }
    
    return builder.captureGraph();
}


cudaGraph_t create_multi_gpu_sptrsv_graph_for_part(
        const int* sn_begin,
        const int* Lip,
        const int* d_Li,  // Device pointer to row indices
        const int* Lp,
        const float* d_Lx,
        const float* d_b,
        float* d_x,       // Solution vector (device)
        int n,            // Total rows
        std::vector<std::vector<int>>& deps, // Dependencies (host)
        int n_super,
        int nnz_sn,
        int rank,
        int n_ranks,
        ncclComm_t comm,
        cudaStream_t stream, // Stream for NCCL setup if needed
        int my_start_sn,
        int my_end_sn,
        int part_val_beg,
        int part_row_idx_start
) {
    CudaDagBuilder builder;
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));


    // Determine dependencies from previous ranks
    // Ideally, we find exactly which *values* we need.
    // For this prototype:
    // - Rank K receives the Full X vector updates from Rank 0...K-1.
    // - Rank K computes.
    // - Rank K sends its updates to Rank K+1...N-1.
    // This serializes the blocks but uses DAG locally.

    // Note: To make this "Graph compliant", we encapsulate NCCL in nodes.

    // 1. Receive Node (from previous rank)
    int recv_node_id = -1;
    if (rank > 0) {
        auto recv_op = [=](cudaStream_t s) {
            // Receive the entire X vector (or relevant part) from rank-1
            // Simple chain: Rank 0 -> Rank 1 -> Rank 2
            // Receive from Rank - 1
            NCCLCHECK(ncclRecv(d_x, n, ncclFloat, rank - 1, comm, s));
        };
        recv_node_id = builder.addNode("NCCL_Recv", recv_op, {});
    }

    // 2. Compute Nodes (Local Supernodes)
    std::map<int, int> sn_to_node_id;

    for (int sn_id = my_start_sn; sn_id < my_end_sn; sn_id++) {
        int col_b = sn_begin[sn_id];
        int col_e = sn_begin[sn_id + 1];
        int sup_width = col_e - col_b;
        int sup_len = Lip[sn_id + 1] - Lip[sn_id];
        int val_beg = Lp[col_b] - part_val_beg;

        int row_idx_start = Lip[sn_id] + sup_width - part_row_idx_start;
        int row_idx_end   = Lip[sn_id + 1] - part_row_idx_start;
        int n_scatter     = row_idx_end - row_idx_start;

        const float* diag_block_ptr = d_Lx + val_beg;
        float* x_block_ptr = d_x + col_b;
        const int* d_current_row_indices = d_Li + row_idx_start;

        // Same logic as single GPU
        auto supernode_op = [=](cudaStream_t s) {
            cublasSetStream(cublas, s);
            if (sup_width > 0) {
                // Diagonal solve
                CUBLAS_CHECK(cublasStrsv(
                        cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                        sup_width, diag_block_ptr, sup_len, x_block_ptr, 1));
            }
            if (sup_len > sup_width && n_scatter > 0) {
                // Scatter update
                const float *off_diag_ptr = diag_block_ptr + sup_width;
                float *d_update = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_update, sizeof(float) * n_scatter, s));
                const float alpha = -1.0f; const float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemv(
                        cublas, CUBLAS_OP_N, n_scatter, sup_width, &alpha,
                        off_diag_ptr, sup_len, x_block_ptr, 1, &beta, d_update, 1));

                const int threads_per_block = 256;
                const int blocks = (n_scatter + threads_per_block - 1) / threads_per_block;
                scatter_kernel<<<blocks, threads_per_block, 0, s>>>(d_update, d_current_row_indices, d_x, n_scatter);
                CUDA_CHECK(cudaFreeAsync(d_update, s));
            }
        };

        // Determine dependencies
        std::vector<int> graph_deps;
        if (recv_node_id != -1) {
            // Implicitly depend on Recv if we depend on any REMOTE node.
            // For block partition, ANY dependency < my_start_sn implies remote dependency.
            bool depends_on_remote = false;
            for (int dep_sn : deps[sn_id]) {
                if (dep_sn < my_start_sn) {
                    depends_on_remote = true;
                    break;
                }
            }
            if (depends_on_remote) {
                graph_deps.push_back(recv_node_id);
            }
        }

        // Add Local Dependencies
        for (int dep_sn : deps[sn_id]) {
            if (dep_sn >= my_start_sn && dep_sn < my_end_sn) {
                if (sn_to_node_id.count(dep_sn)) {
                    graph_deps.push_back(sn_to_node_id[dep_sn]);
                }
            }
        }

        int node_id = builder.addNode("SuperNode_" + std::to_string(sn_id), supernode_op, graph_deps);
        sn_to_node_id[sn_id] = node_id;
    }

    // 3. Send Node (to next rank)
    if (rank < n_ranks - 1) {
        auto send_op = [=](cudaStream_t s) {
            // Send entire X (updated) to next rank
            NCCLCHECK(ncclSend(d_x, n, ncclFloat, rank + 1, comm, s));
        };

        // Send depends on ALL local computations
        // We can create a "Barrier" node or just add dependencies from all leaves.
        // Or simpler: Make it depend on the last few nodes?
        // Correctness: Must depend on ANY node that writes to X.
        // For simplicity: Add explicit dependencies from ALL local nodes? That's too many edges.
        // Better: CudaDagBuilder doesn't have a "Global Barrier".
        // Alternative: Create a dummy node that depends on everything?
        // Let's iterate and find nodes with 0 children? Or just all nodes?
        // Optimization: Just ensure it runs after the LAST scheduled logical node.

        std::vector<int> all_local_nodes;
        for(auto& p : sn_to_node_id) all_local_nodes.push_back(p.second);

        // Ideally we only depend on sinks. But finding sinks requires parsing deps.
        // Let's try adding all. CudaGraph can handle it.
        builder.addNode("NCCL_Send", send_op, all_local_nodes);
    }

    return builder.captureGraph();
}


cudaGraph_t create_single_gpu_sptrsv_graph_for_part(
        const int* sn_begin,
        const int* Lip,
        const int* Li,
        const float* Lx,
        const int* Lp,
        int* d_Li,
        float* d_Lx,
        const float* d_b,
        float* d_x,       // Solution vector (device)
        int n,            // Total rows
        std::vector<std::vector<int>>& deps, // Dependencies (host)
        int n_super,
        int nnz_sn,
        int num_parts
) {

    CudaDagBuilder builder;
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    int end_node_id = -1;
    for (int part = 0; part < num_parts; part++) {
        int my_start_sn, my_end_sn;
        get_partition_range(n_super, part, num_parts, my_start_sn, my_end_sn);

        int part_col_b = sn_begin[my_start_sn];
        int part_col_e = sn_begin[my_end_sn];

        int part_val_beg = Lp[part_col_b];
        int part_val_end = Lp[part_col_e];
        int part_nnz_count = part_val_end - part_val_beg;

        int part_row_idx_start = Lip[my_start_sn];
        int part_row_idx_end = Lip[my_end_sn];
        int part_Li_count = part_row_idx_end - part_row_idx_start;

        // Determine dependencies from previous ranks
        // Ideally, we find exactly which *values* we need.
        // For this prototype:
        // - Rank K receives the Full X vector updates from Rank 0...K-1.
        // - Rank K computes.
        // - Rank K sends its updates to Rank K+1...N-1.
        // This serializes the blocks but uses DAG locally.

        // Note: To make this "Graph compliant", we encapsulate NCCL in nodes.

        // 1. Receive Node (from previous rank)
        int mem_load_node_id = -1;
        if (part > 0) {
            auto mem_load = [=](cudaStream_t s) {
                CUDA_CHECK(cudaMemcpyAsync(d_Lx, Lx + part_val_beg, part_nnz_count * sizeof(float), cudaMemcpyHostToDevice,
                                           s));
                CUDA_CHECK(cudaMemcpyAsync(d_Li, Li + part_row_idx_start, part_Li_count * sizeof(int),
                                           cudaMemcpyHostToDevice, s));
            };
            mem_load_node_id = builder.addNode("Part_start", mem_load, {end_node_id});
        }

        // 2. Compute Nodes (Local Supernodes)
        std::map<int, int> sn_to_node_id;

        for (int sn_id = my_start_sn; sn_id < my_end_sn; sn_id++) {
            int col_b = sn_begin[sn_id];
            int col_e = sn_begin[sn_id + 1];
            int sup_width = col_e - col_b;
            int sup_len = Lip[sn_id + 1] - Lip[sn_id];
            int val_beg = Lp[col_b] - part_val_beg;

            int row_idx_start = Lip[sn_id] + sup_width - part_row_idx_start;
            int row_idx_end = Lip[sn_id + 1] - part_row_idx_start;
            int n_scatter = row_idx_end - row_idx_start;

            const float *diag_block_ptr = d_Lx + val_beg;
            float *x_block_ptr = d_x + col_b;
            const int *d_current_row_indices = d_Li + row_idx_start;

            // Same logic as single GPU
            auto supernode_op = [=](cudaStream_t s) {
                cublasSetStream(cublas, s);
                if (sup_width > 0) {
                    // Diagonal solve
                    CUBLAS_CHECK(cublasStrsv(
                            cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            sup_width, diag_block_ptr, sup_len, x_block_ptr, 1));
                }
                if (sup_len > sup_width && n_scatter > 0) {
                    // Scatter update
                    const float *off_diag_ptr = diag_block_ptr + sup_width;
                    float *d_update = nullptr;
                    CUDA_CHECK(cudaMallocAsync(&d_update, sizeof(float) * n_scatter, s));
                    const float alpha = -1.0f;
                    const float beta = 0.0f;

                    CUBLAS_CHECK(cublasSgemv(
                            cublas, CUBLAS_OP_N, n_scatter, sup_width, &alpha,
                            off_diag_ptr, sup_len, x_block_ptr, 1, &beta, d_update, 1));

                    const int threads_per_block = 256;
                    const int blocks = (n_scatter + threads_per_block - 1) / threads_per_block;
                    scatter_kernel<<<blocks, threads_per_block, 0, s>>>(d_update, d_current_row_indices, d_x,
                                                                        n_scatter);
                    CUDA_CHECK(cudaFreeAsync(d_update, s));
                }
            };

            // Determine dependencies
            std::vector<int> graph_deps;
            if (part > 0)
                graph_deps.push_back(mem_load_node_id);

            // Add Local Dependencies
            for (int dep_sn: deps[sn_id]) {
                if (dep_sn >= my_start_sn && dep_sn < my_end_sn) {
                    if (sn_to_node_id.count(dep_sn)) {
                        graph_deps.push_back(sn_to_node_id[dep_sn]);
                    }
                }
            }

            int node_id = builder.addNode("SuperNode_" + std::to_string(sn_id), supernode_op, graph_deps);
            sn_to_node_id[sn_id] = node_id;
        }

        // 3. finish Node (to next part)
        if (part < num_parts - 1) {
            auto send_op = [=](cudaStream_t s) {};
            // dummy graph node to indicate a partition finish
            std::vector<int> all_local_nodes;
            for (auto &p: sn_to_node_id) all_local_nodes.push_back(p.second);

            // Ideally we only depend on sinks. But finding sinks requires parsing deps.
            // Let's try adding all. CudaGraph can handle it.
            end_node_id = builder.addNode("Part_finish", send_op, all_local_nodes);
        }
    }

    return builder.captureGraph();
}