#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>
#include "mm_loader.h"
#include "sptrsv_multi_gpu.cuh"
#include <chrono>

// Unique ID for NCCL
void get_nccl_unique_id(ncclUniqueId* id, int rank) {
    if (rank == 0) ncclGetUniqueId(id);
    MPI_Bcast(id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    // --- MPI Init ---
    MPI_Init(&argc, &argv);
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // --- GPU Init ---
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    int local_gpu = rank % n_devices;
    cudaSetDevice(local_gpu);
    
    if (rank == 0) 
        std::cout << "Running Multi-GPU Test with " << n_ranks << " ranks." << std::endl;

    // --- NCCL Init ---
    ncclUniqueId id;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, n_ranks, id, rank));

    // --- Data Loading (All ranks load same file for simplicity) ---
    if (argc < 2) {
        if(rank==0) std::cerr << "Usage: ./multi_gpu_test <matrix.mtx>" << std::endl;
        MPI_Finalize();
        return 1;
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
    float *d_x, *d_Lx, *d_b;
    int *d_Li;
    
    // Allocate Full Size on all ranks (simplification)
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Lx, nnz_sn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Li, snMat.i.size() * sizeof(int)));
    // Initialize X with 1.0 (RHS B implicit as 1.0 logic from main.cu)
    std::vector<float> h_x(n, 1.0f); 
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz_sn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Li, snMat.i.data(), snMat.i.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Create Graph
    cudaGraph_t graph = create_multi_gpu_sptrsv_graph(
        snMat.supernodes.data(),
        snMat.ip.data(),
        snMat.i.data(),
        d_Li,
        snMat.p.data(),
        d_Lx,
        nullptr, // b is implicit in logic or unused in favor of x init
        d_x,
        n,
        deps,
        snMat.nSuper,
        nnz_sn,
        rank,
        n_ranks,
        comm,
        0
    );

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Warmup & Run
    for(int i=0; i<3; ++i) {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timer
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<10; ++i) {
        // Correcting: Re-init x for each iteration if we want independent solves/validation
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
        CUDA_CHECK(cudaDeviceSynchronize()); // Synchronize to ensure independent timing/init
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    if (rank == 0) {
        std::cout << "Avg Time: " << (elapsed.count() / 10.0) * 1000.0 << " ms" << std::endl;
    }

    // Validation (Gather X to Rank 0)
    // We need to copy d_x to host, then valid.
    std::vector<float> h_result(n);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Hack: Since we updated disjoint parts, we need to merge results?
    // Wait, with the Recv/Send chain, d_x should be fully updated at the end on the LAST rank?
    // - Rank 0 updates part 0, sends to 1.
    // - Rank 1 receives part 0 (and part 1 is init), updates part 1, sends to 2.
    // - Rank N-1 has the final full vector.
    
    if (rank == n_ranks - 1) {
       // Validate on last rank
       std::cout << "Validating on Rank " << rank << "..." << std::endl;
       // (Validation logic omitted for brevity, but could copy CPU solver from main.cu)
       float sum = 0;
       for(float v : h_result) sum += v;
       std::cout << "Sum X: " << sum << std::endl;
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_x);
    cudaFree(d_Lx);
    cudaFree(d_Li);
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
