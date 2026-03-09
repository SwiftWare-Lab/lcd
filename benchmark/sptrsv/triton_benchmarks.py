

from kernels.sptrsv_triton import *
import scipy.io as scio
from benchmark.spmv import *
from benchmark.bench_utils import *
from kernels.sptrsv_baseline import *


mtx_list = ["/home/salehm32/projects/fused-gnn/fusion/data/gpu-backend-data/1138_bus/1138_bus.mtx"]
if len(sys.argv) > 1:
    mtx_file = sys.argv[1]
    mtx_folder = sys.argv[2]
    with open(mtx_file) as f:
        mtx_list = f.readlines()
        mtx_list = list(map(lambda x: os.path.join(mtx_folder,x.strip()), mtx_list))
method_list = ["tr_csr_syncfree_baseline", "tr_csc_syncfree_baseline", "tr_csr_wavefront_baseline", "tr_csc_wavefront_baseline"]
configs = []


def correctness_test():
    for mtx in mtx_list:
        print(f"Testing {mtx}...")
        matrix = scio.mmread(mtx)
        n = matrix.shape[0]
        A = matrix.tocsc()
        A = tril(A, format="csc")
        if A.data.dtype != np.float64:
            print(f"Matrix {mtx} is not double precision")
            A.data = A.data.astype(np.float64)
        b = np.ones(n)
        A_csr = A.tocsr()
        val_f32_csr = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        indices_i32_csr = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32_csr = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        val_f32_csc = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        indices_i32_csc = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32_csc = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        dep_count = torch.tensor(incoming_edge(A), dtype=torch.int32, device=DEVICE)
        succ_offsets_csr, succ_indices_csr = build_csr_successors(A_csr)
        succ_offsets_csr_f32 = torch.tensor(succ_offsets_csr, dtype=torch.int32, device=DEVICE)
        succ_indices_csr_i32 = torch.tensor(succ_indices_csr, dtype=torch.int32, device=DEVICE)
        y_syncfree_csr = sptrsv_sync_free_csr_baseline(
            val_f32_csr, indices_i32_csr, indptr_i32_csr, b_f32, n, n, A_csr.nnz,
            dep_count, succ_offsets_csr_f32, succ_indices_csr_i32
        )
        y_syncfree_csc = sptrsv_sync_free_csc_baseline(
            val_f32_csc, indices_i32_csc, indptr_i32_csc, b_f32, n, n, A.nnz, dep_count
        )
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        y_wavefront_csr = sptrsv_wavefront_csr_baseline(
            val_f32_csr, indices_i32_csr, indptr_i32_csr, b_f32, n, n, A_csr.nnz, wave_start_loc, waves_gpu
        )
        y_wavefront_csc = sptrsv_wavefront_csc_baseline(
            val_f32_csc, indices_i32_csc, indptr_i32_csc, b_f32, n, n, A.nnz, wave_start_loc, waves_gpu
        )
        y_syncfree_csr = y_syncfree_csr.cpu().numpy()
        y_syncfree_csc = y_syncfree_csc.cpu().numpy()
        y_wavefront_csr = y_wavefront_csr.cpu().numpy()
        y_wavefront_csc = y_wavefront_csc.cpu().numpy()
        print(y_syncfree_csr)
        assert np.allclose(y_syncfree_csr, y_syncfree_csc, rtol=1e-5, atol=1e-8), f"Sync-free CSR and CSC results do not match for {mtx}"
        assert np.allclose(y_syncfree_csr, y_wavefront_csr, rtol=1e-5, atol=1e-8), f"Sync-free CSR and wavefront CSR results do not match for {mtx}"
        assert np.allclose(y_syncfree_csc, y_wavefront_csc, rtol=1e-5, atol=1e-8), f"Sync-free CSC and wavefront CSC results do not match for {mtx}"
        print(f"Matrix {mtx} passed correctness test!")

configs.append(
    triton.testing.Benchmark(
        x_names=["matrices"],
        x_vals=[mtx_list[i] for i in range(0, len(mtx_list))],
        line_arg="provider",
        line_vals=method_list,
        line_names=method_list,
        ylabel="GFLOP/S",
        plot_name="sptrsv-performance",
        args={}
    ))


@triton.testing.perf_report(configs)
def benchmark(matrices, provider):
    # read matrix from file
    matrix = scio.mmread(matrices)
    n = matrix.shape[0]
    # convert matrix to csc
    A = matrix.tocsc()
    # makke it sparse lower triangular
    A = tril(A, format="csc")
    if A.data.dtype != np.float64:
        print(f"Matrix {matrices} is not double precision")
        A.data = A.data.astype(np.float64)

    # add diagonals with 100
    A.setdiag(A.diagonal())
    A.sort_indices() # Ensure sorted indices
    b = np.ones(n)  # Dense vector b
    quantiles = [0.5, 0.2, 0.8]
    rep, warmpup = 100, 25
    if provider == "tr_csr_syncfree_baseline":
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        in_edge = incoming_edge(A)
        dep_count = torch.tensor(in_edge, dtype=torch.int32, device=DEVICE)
        succ_offsets_csr, succ_indices_csr = build_csr_successors(A_csr)
        succ_offsets_csr_f32 = torch.tensor(succ_offsets_csr, dtype=torch.int32, device=DEVICE)
        succ_indices_csr_i32 = torch.tensor(succ_indices_csr, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda:sptrsv_sync_free_csr_baseline(
            val_f32, indices_i32, indptr_i32, b_f32, n, n, A_csr.nnz, dep_count, succ_offsets_csr_f32, succ_indices_csr_i32
        ), quantiles=quantiles, warmup=warmpup, rep=rep)
    elif provider == "tr_csc_syncfree_baseline":
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        in_edge = incoming_edge(A)
        dep_count = torch.tensor(in_edge, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda:sptrsv_sync_free_csc_baseline(
            val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz, dep_count
        ), quantiles=quantiles, warmup=warmpup, rep=rep)
    elif provider == "tr_csr_wavefront_baseline":
        A_csr = A.tocsr()
        val_f32 = torch.tensor(A_csr.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A_csr.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A_csr.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_wavefront_csr_baseline(
            val_f32, indices_i32, indptr_i32, b_f32, n, n, A_csr.nnz, wave_start_loc, waves_gpu
        ), quantiles=quantiles, warmup=warmpup, rep=rep)
    elif provider == "tr_csc_wavefront_baseline":
        val_f32 = torch.tensor(A.data, dtype=REAL_TYPE, device=DEVICE)
        b_f32 = torch.tensor(b, dtype=REAL_TYPE, device=DEVICE)
        indices_i32 = torch.tensor(A.indices, dtype=torch.int32, device=DEVICE)
        indptr_i32 = torch.tensor(A.indptr, dtype=torch.int32, device=DEVICE)
        wavefronts = compute_wavefront(A)
        wave_start_loc, waves = compact_2d_list(wavefronts)
        waves_gpu = torch.tensor(waves, dtype=torch.int32, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sptrsv_wavefront_csc_baseline(
            val_f32, indices_i32, indptr_i32, b_f32, n, n, A.nnz, wave_start_loc, waves_gpu
        ), quantiles=quantiles, warmup=warmpup, rep=rep)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    perf = lambda ms: (A.shape[0] + 2 * A.nnz) / (ms * 1e-3) / 1e9
    return perf(ms), perf(min_ms), perf(max_ms)




# correctness_test()

benchmark.run(print_data=True, save_path=".", show_plots=False)