#include <cuda_runtime.h>
#include <cusparse.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <stdexcept>
#include <sstream>

namespace py = pybind11;

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::ostringstream oss;                                             \
            oss << "CUDA error: " << cudaGetErrorString(err__);                 \
            throw std::runtime_error(oss.str());                                \
        }                                                                       \
    } while (0)

#define CHECK_CUSPARSE(call)                                                    \
    do {                                                                        \
        cusparseStatus_t st__ = (call);                                         \
        if (st__ != CUSPARSE_STATUS_SUCCESS) {                                  \
            std::ostringstream oss;                                             \
            oss << "cuSPARSE error code: " << static_cast<int>(st__);           \
            throw std::runtime_error(oss.str());                                \
        }                                                                       \
    } while (0)

class CuSparseSpTRSV {
public:
    CuSparseSpTRSV(py::array_t<int, py::array::c_style | py::array::forcecast> rowptr,
                   py::array_t<int, py::array::c_style | py::array::forcecast> colind,
                   py::array_t<double, py::array::c_style | py::array::forcecast> values,
                   bool lower = true,
                   bool unit_diag = false)
            : lower_(lower), unit_diag_(unit_diag) {
        auto rowptr_buf = rowptr.request();
        auto colind_buf = colind.request();
        auto values_buf = values.request();

        if (rowptr_buf.ndim != 1 || colind_buf.ndim != 1 || values_buf.ndim != 1) {
            throw std::runtime_error("rowptr, colind, and values must be 1D arrays.");
        }

        if (colind_buf.shape[0] != values_buf.shape[0]) {
            throw std::runtime_error("colind and values must have the same length.");
        }

        n_ = static_cast<int>(rowptr_buf.shape[0]) - 1;
        nnz_ = static_cast<int>(colind_buf.shape[0]);

        if (n_ <= 0) {
            throw std::runtime_error("Invalid rowptr length.");
        }

        const int* h_rowptr = static_cast<const int*>(rowptr_buf.ptr);
        const int* h_colind = static_cast<const int*>(colind_buf.ptr);
        const double* h_values = static_cast<const double*>(values_buf.ptr);

        CHECK_CUSPARSE(cusparseCreate(&handle_));

        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_rowptr_), (n_ + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_colind_), nnz_ * sizeof(int)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_values_), nnz_ * sizeof(double)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_rhs_), n_ * sizeof(double)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_sol_), n_ * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(d_rowptr_, h_rowptr, (n_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_colind_, h_colind, nnz_ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_values_, h_values, nnz_ * sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUSPARSE(cusparseCreateCsr(&matA_,
                                         n_, n_, nnz_,
                                         d_rowptr_, d_colind_, d_values_,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_64F));

        // vecB_ is the RHS input, vecX_ is the solution output
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecB_, n_, d_rhs_, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX_, n_, d_sol_, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsv_descr_));

        cusparseFillMode_t fill_mode = lower_ ? CUSPARSE_FILL_MODE_LOWER
                                              : CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag_type = unit_diag_ ? CUSPARSE_DIAG_TYPE_UNIT
                                                  : CUSPARSE_DIAG_TYPE_NON_UNIT;

        CHECK_CUSPARSE(cusparseSpMatSetAttribute(
                matA_, CUSPARSE_SPMAT_FILL_MODE, &fill_mode, sizeof(fill_mode)));
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(
                matA_, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type, sizeof(diag_type)));
    }

    ~CuSparseSpTRSV() {
        try {
            if (buffer_) cudaFree(buffer_);
            if (spsv_descr_) cusparseSpSV_destroyDescr(spsv_descr_);
            if (vecB_) cusparseDestroyDnVec(vecB_);
            if (vecX_) cusparseDestroyDnVec(vecX_);
            if (matA_) cusparseDestroySpMat(matA_);
            if (handle_) cusparseDestroy(handle_);
            if (d_rowptr_) cudaFree(d_rowptr_);
            if (d_colind_) cudaFree(d_colind_);
            if (d_values_) cudaFree(d_values_);
            if (d_rhs_) cudaFree(d_rhs_);
            if (d_sol_) cudaFree(d_sol_);
        } catch (...) {
        }
    }

    void analyze() {
        if (analyzed_) return;

        CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle_,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha_,
                                               matA_,
                                               vecB_,
                                               vecX_,
                                               CUDA_R_64F,
                                               CUSPARSE_SPSV_ALG_DEFAULT,
                                               spsv_descr_,
                                               &buffer_size_));

        CHECK_CUDA(cudaMalloc(&buffer_, buffer_size_));

        CHECK_CUSPARSE(cusparseSpSV_analysis(handle_,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha_,
                                             matA_,
                                             vecB_,
                                             vecX_,
                                             CUDA_R_64F,
                                             CUSPARSE_SPSV_ALG_DEFAULT,
                                             spsv_descr_,
                                             buffer_));

        CHECK_CUDA(cudaDeviceSynchronize());
        analyzed_ = true;
    }

    py::array_t<double> solve(
            py::array_t<double, py::array::c_style | py::array::forcecast> b) {
        ensure_analyzed();

        auto b_buf = b.request();
        if (b_buf.ndim != 1) {
            throw std::runtime_error("b must be a 1D array.");
        }
        if (static_cast<int>(b_buf.shape[0]) != n_) {
            throw std::runtime_error("b has incorrect size.");
        }

        const double* h_b = static_cast<const double*>(b_buf.ptr);

        CHECK_CUDA(cudaMemcpy(d_rhs_, h_b, n_ * sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUSPARSE(cusparseSpSV_solve(handle_,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha_,
                                          matA_,
                                          vecB_,
                                          vecX_,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          spsv_descr_));

        CHECK_CUDA(cudaDeviceSynchronize());

        py::array_t<double> out(n_);
        auto out_buf = out.request();
        double* h_x = static_cast<double*>(out_buf.ptr);

        CHECK_CUDA(cudaMemcpy(h_x, d_sol_, n_ * sizeof(double), cudaMemcpyDeviceToHost));
        return out;
    }

    int n() const { return n_; }
    int nnz() const { return nnz_; }

private:
    void ensure_analyzed() {
        if (!analyzed_) {
            analyze();
        }
    }

    int n_ = 0;
    int nnz_ = 0;
    bool lower_ = true;
    bool unit_diag_ = false;

    double alpha_ = 1.0;

    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t matA_ = nullptr;
    cusparseDnVecDescr_t vecB_ = nullptr;
    cusparseDnVecDescr_t vecX_ = nullptr;
    cusparseSpSVDescr_t spsv_descr_ = nullptr;

    int* d_rowptr_ = nullptr;
    int* d_colind_ = nullptr;
    double* d_values_ = nullptr;
    double* d_rhs_ = nullptr;
    double* d_sol_ = nullptr;

    void* buffer_ = nullptr;
    size_t buffer_size_ = 0;
    bool analyzed_ = false;
};

PYBIND11_MODULE(cusparse_sptrsv, m) {
    m.doc() = "pybind11 wrapper for cuSPARSE SpSV";

    py::class_<CuSparseSpTRSV>(m, "CuSparseSpTRSV")
            .def(py::init<
                         py::array_t<int, py::array::c_style | py::array::forcecast>,
                         py::array_t<int, py::array::c_style | py::array::forcecast>,
                         py::array_t<double, py::array::c_style | py::array::forcecast>,
                         bool,
                         bool>(),
                 py::arg("rowptr"),
                 py::arg("colind"),
                 py::arg("values"),
                 py::arg("lower") = true,
                 py::arg("unit_diag") = false)
            .def("analyze", &CuSparseSpTRSV::analyze)
            .def("solve", &CuSparseSpTRSV::solve, py::arg("b"))
            .def_property_readonly("n", &CuSparseSpTRSV::n)
            .def_property_readonly("nnz", &CuSparseSpTRSV::nnz);
}