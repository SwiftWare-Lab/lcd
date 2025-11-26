//
// Created by salehm32 on 2025-11-17.
//
#include <nvbench/nvbench.cuh>
#include "kernels.cuh"
#include "sptrsv_supernodal_cuda_graph.cuh"
#include "mm_loader.h"
#include "supernodal_utils.h"

auto base_path = "/home/salehm32/projects/fused-gnn/fusion/data/SPD/";
//void bench_sn_sptrsv_cudagraph_incorrect(nvbench::state &state)
//{
//
//    auto matrix_name = state.get_string("matrix_names");
//    auto mat_path = base_path + matrix_name;
//
//
//    Eigen::SparseMatrix<float> spMat = load_matrix_market(mat_path);
//    auto triLMat = spMat.triangularView<Eigen::Lower>();
//    Eigen::SparseMatrix<float, Eigen::ColMajor> spTriLMat= triLMat;
//    spTriLMat.makeCompressed();
//
//    auto rowPtr = const_cast<int*>(spTriLMat.innerIndexPtr());
//    auto colInd = const_cast<int*>(spTriLMat.outerIndexPtr());
//    auto values = const_cast<float*>(spTriLMat.valuePtr());
//    int n = spTriLMat.rows();
//    auto* cscMatrix =  new CSCMatrix<float>();
//    cscMatrix->n = n;
//    cscMatrix->indptr = colInd;
//    cscMatrix->indices = rowPtr;
//    cscMatrix->data = values;
//    auto *b = new float[n];
//    auto *x = new float[n];
//    for (int i = 0; i < n; i++){
//        b[i] = 1;
//        x[i] = 1;
//    }
//    auto snMat = convert_csc_to_supernodal<float>(*cscMatrix);
//    auto snDag = build_supernodal_dag_csc(snMat);
//    auto waveFronts = compute_wavefront(*snDag);
//    auto compact2DList = compact_2d_list(waveFronts);
//    int nnz = snMat.x.size();
//    float* d_x = nullptr;
//    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * n));
//    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
//
//    float* d_Lx = nullptr;
//    CUDA_CHECK(cudaMalloc(&d_Lx, sizeof(float) * nnz));
//    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
//    cudaStream_t stream;
//    CUDA_CHECK(cudaStreamCreate(&stream));
//
//    cublasHandle_t cublas;
//    CUBLAS_CHECK(cublasCreate(&cublas));
//    CUBLAS_CHECK(cublasSetStream(cublas, stream));
//
//    auto graphs = sptrsv_csc_supernodal_wavefront_cuda_graph_incorrect(
//            snMat.supernodes,
//            snMat.ip,
//            snMat.i,
//            snMat.p,
//            d_Lx,
//            b,
//            d_x,
//            compact2DList.wave_start_loc,
//            compact2DList.waves,
//            n,
//            nnz,
//            stream,
//            cublas
//    );
//    std::vector<cudaGraphExec_t> graphExecs;
//    for (auto g : graphs) {
//        cudaGraphExec_t gexec = nullptr;
//        CUDA_CHECK(cudaGraphInstantiate(&gexec, g, nullptr, nullptr, 0));
//        graphExecs.push_back(gexec);
//    }
//
//    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
//        for (auto gexec : graphExecs) {
//            CUDA_CHECK(cudaGraphLaunch(gexec, stream));
//            CUDA_CHECK(cudaStreamSynchronize(stream));
//        }
//    });
//
//
//    for (auto gexec : graphExecs) {
//        CUDA_CHECK(cudaGraphExecDestroy(gexec));
//    }
//    for (auto g : graphs) {
//        CUDA_CHECK(cudaGraphDestroy(g));
//    }
//
//
//    CUBLAS_CHECK(cublasDestroy(cublas));
//    CUDA_CHECK(cudaStreamDestroy(stream));
//    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
//    CUDA_CHECK(cudaFree(d_x));
//    CUDA_CHECK(cudaFree(d_Lx));
//
//
//    delete[] b;
//    delete[] x;
//    delete snDag;
//
//}


//void bench_sn_sptrsv(nvbench::state &state)
//{
//
//    auto matrix_name = state.get_string("matrix_names");
//    auto mat_path = base_path + matrix_name;
//
//
//    Eigen::SparseMatrix<float> spMat = load_matrix_market(mat_path);
//    auto triLMat = spMat.triangularView<Eigen::Lower>();
//    Eigen::SparseMatrix<float, Eigen::ColMajor> spTriLMat= triLMat;
//    spTriLMat.makeCompressed();
//
//    auto rowPtr = const_cast<int*>(spTriLMat.innerIndexPtr());
//    auto colInd = const_cast<int*>(spTriLMat.outerIndexPtr());
//    auto values = const_cast<float*>(spTriLMat.valuePtr());
//    int n = spTriLMat.rows();
//    auto* cscMatrix =  new CSCMatrix<float>();
//    cscMatrix->n = n;
//    cscMatrix->indptr = colInd;
//    cscMatrix->indices = rowPtr;
//    cscMatrix->data = values;
//    auto *b = new float[n];
//    auto *x = new float[n];
//    for (int i = 0; i < n; i++){
//        b[i] = 1;
//        x[i] = 1;
//    }
//    auto snMat = convert_csc_to_supernodal<float>(*cscMatrix);
//    auto snDag = build_supernodal_dag_csc(snMat);
//    auto waveFronts = compute_wavefront(*snDag);
//    auto compact2DList = compact_2d_list(waveFronts);
//    int nnz = snMat.x.size();
//    float* d_x = nullptr;
//    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * n));
//    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
//
//    float* d_Lx = nullptr;
//    CUDA_CHECK(cudaMalloc(&d_Lx, sizeof(float) * nnz));
//    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
//
//    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
//        sptrsv_csc_supernodal_wavefront_incorrect(
//                snMat.supernodes,
//                snMat.ip,
//                snMat.i,
//                snMat.p,
//                d_Lx,
//                b,
//                d_x,
//                compact2DList.wave_start_loc,
//                compact2DList.waves,
//                n,
//                nnz
//        );
//    });
//
//
//    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
//    CUDA_CHECK(cudaFree(d_x));
//    CUDA_CHECK(cudaFree(d_Lx));
//
//
//    delete[] b;
//    delete[] x;
//    delete snDag;
//}

void bench_sn_sptrsv_cudagraph(nvbench::state &state)
{

    auto matrix_name = state.get_string("matrix_names");
    auto mat_path = base_path + matrix_name;


    Eigen::SparseMatrix<float> spMat = load_matrix_market(mat_path);
    auto triLMat = spMat.triangularView<Eigen::Lower>();
    Eigen::SparseMatrix<float, Eigen::ColMajor> spTriLMat= triLMat;
    spTriLMat.makeCompressed();

    auto rowPtr = const_cast<int*>(spTriLMat.innerIndexPtr());
    auto colInd = const_cast<int*>(spTriLMat.outerIndexPtr());
    auto values = const_cast<float*>(spTriLMat.valuePtr());
    int n = spTriLMat.rows();
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
    auto snDag = build_supernodal_dag_csc(snMat);
    auto waveFronts = compute_wavefront(*snDag);
    auto compact2DList = compact_2d_list(waveFronts);
    int nnz = snMat.x.size();
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_Lx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Lx, sizeof(float) * nnz));
    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    auto g = sptrsv_csc_supernodal_wavefront_cudagraph(
            snMat.supernodes.data(),
            snMat.ip.data(),
            snMat.i.data(),
            snMat.p.data(),
            d_Lx,
            b,
            d_x,
            compact2DList.wave_start_loc,
            compact2DList.waves,
            n,
            nnz
    );
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaGraphExec_t gexec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&gexec, g, nullptr, nullptr, 0));
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        CUDA_CHECK(cudaGraphLaunch(gexec, stream));
        cudaStreamSynchronize(stream);
    });

    CUDA_CHECK(cudaGraphExecDestroy(gexec));
    CUDA_CHECK(cudaGraphDestroy(g));
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_Lx));


    delete[] b;
    delete[] x;
    delete snDag;
}

std::vector<std::string> matrix_names = {"Andrews/Andrews.mtx",
                     "bmw7st_1/bmw7st_1.mtx",
                     "bmwcra_1/bmwcra_1.mtx",
//                     "crankseg_1/crankseg_1.mtx",
//                     "crankseg_2/crankseg_2.mtx",
                     "hood/hood.mtx"};


//NVBENCH_BENCH(bench_sn_sptrsv_cudagraph_incorrect).set_name("super_nodal_cudagraph_incorrect")
//    .add_string_axis("matrix_names", matrix_names);
//NVBENCH_BENCH(bench_sn_sptrsv).set_name("super_nodal")
//        .add_string_axis("matrix_names", matrix_names);
NVBENCH_BENCH(bench_sn_sptrsv_cudagraph).set_name("super_nodal_cudagraph")
        .add_string_axis("matrix_names", matrix_names);
