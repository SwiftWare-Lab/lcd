

from kernels.sptrsv_triton import *
import scipy.io as scio
from benchmark.spmv import *
from benchmark.bench_utils import *
from kernels.sptrsv_baseline import *


DEFAULT_MTX_LIST = ["/home/salehm32/projects/fused-gnn/fusion/data/gpu-backend-data/1138_bus/1138_bus.mtx"]
if len(sys.argv) > 1:
    mtx_file = sys.argv[1]
    mtx_folder = sys.argv[2]
    with open(mtx_file) as f:
        mtx_list = f.readlines()
        mtx_list = list(map(lambda x: os.path.join(mtx_folder,x.strip()), mtx_list))
method_list = ["tr_csr_syncfree_baseline", "tr_csc_syncfree_baseline", "tr_csr_wavefront_baseline", "tr_csc_wavefront_baseline"]
configs = []

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





benchmark.run(print_data=True, save_path=".", show_plots=False)