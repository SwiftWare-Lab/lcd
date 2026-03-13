// sptrsv_cusparse_bench.cu
//
// Benchmark cuSPARSE SpSV on the LOWER-TRIANGULAR part of a Matrix Market matrix.
// The behavior is similar to scipy.sparse.tril(A): keep entries with col <= row.
//
// Build:
//   nvcc -O3 -std=c++17 sptrsv_cusparse_bench.cu -lcusparse -o sptrsv_bench
//
// Run:
//   ./sptrsv_bench matrix.mtx [iters] [warmup]
//
// Notes:
// - Supports coordinate Matrix Market files.
// - Supports real/integer/pattern fields.
// - Converts input to CSR, then applies tril() semantics.
// - Benchmarks solve phase only (analysis done once before timing).
// - Approximate GFLOP/s is reported as 2*nnz / time.

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t status_ = (call);                                            \
        if (status_ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " -> " << cudaGetErrorString(status_) << std::endl;     \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUSPARSE(call)                                                     \
    do {                                                                         \
        cusparseStatus_t status_ = (call);                                       \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__     \
                      << " -> status code " << static_cast<int>(status_)         \
                      << std::endl;                                              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

struct COOEntry {
    int row;
    int col;
    double val;
};

struct CSRMatrix {
    int nrows = 0;
    int ncols = 0;
    int nnz   = 0;
    std::vector<int> rowPtr;
    std::vector<int> colInd;
    std::vector<double> values;
};

struct MMHeader {
    bool coordinate = false;
    bool array = false;
    bool real = false;
    bool integer = false;
    bool pattern = false;
    bool symmetric = false;
    bool hermitian = false;
    bool skew = false;
};

static MMHeader parse_mm_banner(const std::string& line) {
    std::istringstream iss(line);
    std::string banner, mtx, format, field, symmetry;
    iss >> banner >> mtx >> format >> field >> symmetry;

    if (banner != "%%MatrixMarket" || mtx != "matrix") {
        throw std::runtime_error("Invalid Matrix Market banner.");
    }

    MMHeader h;
    h.coordinate = (format == "coordinate");
    h.array      = (format == "array");
    h.real       = (field == "real");
    h.integer    = (field == "integer");
    h.pattern    = (field == "pattern");
    h.symmetric  = (symmetry == "symmetric");
    h.hermitian  = (symmetry == "hermitian");
    h.skew       = (symmetry == "skew-symmetric");

    if (!h.coordinate) {
        throw std::runtime_error("Only coordinate Matrix Market format is supported.");
    }
    if (!(h.real || h.integer || h.pattern)) {
        throw std::runtime_error("Only real/integer/pattern Matrix Market fields are supported.");
    }
    if (h.hermitian || h.skew) {
        throw std::runtime_error("Hermitian/skew-symmetric inputs are not supported.");
    }

    return h;
}

static CSRMatrix coo_to_csr(int m, int n, std::vector<COOEntry>& entries) {
    std::sort(entries.begin(), entries.end(),
              [](const COOEntry& a, const COOEntry& b) {
                  if (a.row != b.row) return a.row < b.row;
                  return a.col < b.col;
              });

    CSRMatrix A;
    A.nrows = m;
    A.ncols = n;
    A.nnz   = static_cast<int>(entries.size());
    A.rowPtr.assign(m + 1, 0);
    A.colInd.resize(A.nnz);
    A.values.resize(A.nnz);

    for (const auto& e : entries) {
        if (e.row < 0 || e.row >= m || e.col < 0 || e.col >= n) {
            throw std::runtime_error("Entry out of bounds after index conversion.");
        }
        A.rowPtr[e.row + 1]++;
    }

    for (int i = 0; i < m; ++i) {
        A.rowPtr[i + 1] += A.rowPtr[i];
    }

    std::vector<int> offset = A.rowPtr;
    for (const auto& e : entries) {
        int pos = offset[e.row]++;
        A.colInd[pos] = e.col;
        A.values[pos] = e.val;
    }

    return A;
}

static CSRMatrix tril_csr(const CSRMatrix& A) {
    CSRMatrix L;
    L.nrows = A.nrows;
    L.ncols = A.ncols;
    L.rowPtr.assign(L.nrows + 1, 0);

    std::vector<int> cols;
    std::vector<double> vals;
    cols.reserve(A.nnz);
    vals.reserve(A.nnz);

    for (int i = 0; i < A.nrows; ++i) {
        for (int p = A.rowPtr[i]; p < A.rowPtr[i + 1]; ++p) {
            int j = A.colInd[p];
            if (j <= i) {
                cols.push_back(j);
                vals.push_back(A.values[p]);
                L.rowPtr[i + 1]++;
            }
        }
    }

    for (int i = 0; i < L.nrows; ++i) {
        L.rowPtr[i + 1] += L.rowPtr[i];
    }

    L.nnz = static_cast<int>(cols.size());
    L.colInd = std::move(cols);
    L.values = std::move(vals);

    return L;
}

static CSRMatrix read_matrix_market_to_csr(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    if (!std::getline(fin, line)) {
        throw std::runtime_error("Empty file.");
    }

    MMHeader header = parse_mm_banner(line);

    do {
        if (!std::getline(fin, line)) {
            throw std::runtime_error("Unexpected EOF while reading comments.");
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream dims(line);
    int m, n, nnz_input;
    dims >> m >> n >> nnz_input;
    if (!dims) {
        throw std::runtime_error("Failed to read matrix dimensions.");
    }

    std::vector<COOEntry> entries;
    entries.reserve(header.symmetric ? 2 * nnz_input : nnz_input);

    for (int k = 0; k < nnz_input; ++k) {
        if (!std::getline(fin, line)) {
            throw std::runtime_error("Unexpected EOF while reading entries.");
        }
        if (line.empty()) {
            --k;
            continue;
        }

        std::istringstream iss(line);
        int i, j;
        double v = 1.0;

        if (header.pattern) {
            iss >> i >> j;
        } else {
            iss >> i >> j >> v;
        }

        if (!iss) {
            throw std::runtime_error("Malformed Matrix Market entry.");
        }

        --i;  // convert 1-based to 0-based
        --j;

        entries.push_back({i, j, v});

        // Expand symmetric storage to full matrix first,
        // then tril() will keep the lower triangular half.
        if (header.symmetric && i != j) {
            entries.push_back({j, i, v});
        }
    }

    CSRMatrix A = coo_to_csr(m, n, entries);
    CSRMatrix L = tril_csr(A);
    return L;
}

static bool has_full_diagonal(const CSRMatrix& A) {
    for (int i = 0; i < A.nrows; ++i) {
        bool found = false;
        for (int p = A.rowPtr[i]; p < A.rowPtr[i + 1]; ++p) {
            if (A.colInd[p] == i) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

static double vec_mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = std::accumulate(x.begin(), x.end(), 0.0);
    return s / static_cast<double>(x.size());
}

static double vec_stddev(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double mu = vec_mean(x);
    double s2 = 0.0;
    for (double v : x) {
        double d = v - mu;
        s2 += d * d;
    }
    return std::sqrt(s2 / static_cast<double>(x.size()));
}

struct BenchResult {
    std::string name;
    double avg_ms = 0.0;
    double std_ms = 0.0;
    double gflops = 0.0;
};

static BenchResult benchmark_spsv(const CSRMatrix& A, bool unit_diag,
                                  int iters, int warmup) {
    if (A.nrows != A.ncols) {
        throw std::runtime_error("SpSV requires a square matrix.");
    }

    const int n = A.nrows;
    const int nnz = A.nnz;
    const double alpha = 1.0;

    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    double *d_vals = nullptr, *d_x = nullptr, *d_y = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, A.rowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, A.colInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, A.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> hx(n, 1.0);
    std::vector<double> hy(n, 0.0);
    CHECK_CUDA(cudaMemcpy(d_x, hx.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, hy.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseSpSVDescr_t spsvDescr;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                     n, n, nnz,
                                     d_rowPtr, d_colInd, d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr));

    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = unit_diag ? CUSPARSE_DIAG_TYPE_UNIT
                                        : CUSPARSE_DIAG_TYPE_NON_UNIT;

    CHECK_CUSPARSE(cusparseSpMatSetAttribute(
            matA, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(
            matA, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            vecY,
            CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr,
            &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpSV_analysis(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            vecY,
            CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr,
            dBuffer));

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUSPARSE(cusparseSpSV_solve(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                matA,
                vecX,
                vecY,
                CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT,
                spsvDescr));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<double> times_ms;
    times_ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUSPARSE(cusparseSpSV_solve(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                matA,
                vecX,
                vecY,
                CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT,
                spsvDescr));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(static_cast<double>(ms));
    }

    BenchResult result;
    result.name = std::string("cusparseSpSV(default, lower, ")
                  + (unit_diag ? "unit-diag" : "non-unit-diag") + ")";
    result.avg_ms = vec_mean(times_ms);
    result.std_ms = vec_stddev(times_ms);

    double avg_s = result.avg_ms * 1e-3;
    result.gflops = (avg_s > 0.0)
                    ? (2.0 * static_cast<double>(nnz) / avg_s / 1e9)
                    : 0.0;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [iters=100] [warmup=10]\n";
        return EXIT_FAILURE;
    }

    const std::string filename = argv[1];
    const int iters  = (argc >= 3) ? std::stoi(argv[2]) : 100;
    const int warmup = (argc >= 4) ? std::stoi(argv[3]) : 10;

    try {
        CSRMatrix A = read_matrix_market_to_csr(filename);

        if (A.nrows != A.ncols) {
            throw std::runtime_error("Input must be square for triangular solve.");
        }

        bool full_diag = has_full_diagonal(A);

//        std::cout << "Loaded matrix: " << filename << "\n";
//        std::cout << "Shape after tril(): " << A.nrows << " x " << A.ncols << "\n";
//        std::cout << "NNZ after tril():   " << A.nnz << "\n";
//        std::cout << "Diagonal detected:  "
//                  << (full_diag ? "present" : "missing in at least one row") << "\n";
//        std::cout << "Iterations: " << iters
//                  << ", warmup: " << warmup << "\n\n";

        BenchResult result = benchmark_spsv(
                A,
                !full_diag,   // if diagonal missing, treat as unit diagonal
                iters,
                warmup);

//        std::cout << std::left
//                  << std::setw(45) << "Implementation"
//                  << std::setw(15) << "Avg ms"
//                  << std::setw(15) << "Std ms"
//                  << std::setw(15) << "Approx GF/s"
//                  << "\n";
//        std::cout << std::string(90, '-') << "\n";
//
//        std::cout << std::left
//                  << std::setw(45) << result.name
//                  << std::setw(15) << std::fixed << std::setprecision(6) << result.avg_ms
//                  << std::setw(15) << std::fixed << std::setprecision(6) << result.std_ms
//                  << std::setw(15) << std::fixed << std::setprecision(6) << result.gflops
//                  << "\n";
        std::cout << filename << ","
                  << A.nnz << ","
                  << result.avg_ms << ","
                  << result.gflops
                  << std::endl;
//        std::cout << "\nDone.\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}