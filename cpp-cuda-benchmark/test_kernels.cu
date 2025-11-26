//
// Created by salehm32 on 2025-11-18.
//

#include "mm_loader.h"
#include "supernodal_utils.h"
#include "sptrsv_supernodal_cuda_graph.cuh"
//#include <Eigen/Sparse>

void sptrsv_csc_cpu(int* Ap, int* Ai, float* Ax, float *b, float* x, int n){
    for (int i = 0; i < n; i++){
        x[i] = (b[i] - x[i]) / Ax[Ap[i]];
        for (int j = Ap[i] + 1; j < Ap[i+1]; j++){
            x[Ai[j]] += Ax[j] * x[i];
        }
    }
}


int main(int argv, char** argc){
    std::string file_name = argc[1];
    Eigen::SparseMatrix<float> spMat = load_matrix_market(file_name);
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
    auto deps = build_dependency_graph(snMat);
    int nnz = snMat.x.size();
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_Lx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Lx, sizeof(float) * nnz));
    CUDA_CHECK(cudaMemcpy(d_Lx, snMat.x.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
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
            nnz
    );

    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_Lx));

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

    std::cout << "total count: " << spTriLMat.rows() << "\nincorrect count:" << count << std::endl;

    delete[] b;
    delete[] x;
}
