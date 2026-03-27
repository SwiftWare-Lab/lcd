from __future__ import annotations

import math
import time

import cupy as cp
import numpy as np

WARP_SIZE = 32
DEFAULT_BORDER = 10
DEFAULT_WARPS_PER_BLOCK = 32

_YY_KERNEL = None

_YY_KERNEL_SRC = r"""
extern "C" __global__
void yysptrsv_csr_kernel(const int *__restrict__ d_csrRowPtr,
                         const int *__restrict__ d_csrColIdx,
                         const double *__restrict__ d_csrVal,
                         volatile int *d_get_value,
                         const int m,
                         const int nnz,
                         const double *__restrict__ d_b,
                         double *d_x,
                         const int begin,
                         const int *__restrict__ d_warp_num,
                         const int Len,
                         int *d_id_extractor) {
    const int global_id = atomicAdd(d_id_extractor, 1);
    const int warp_id = global_id / 32;
    int row;

    if (warp_id >= (Len - 1))
        return;

    const int lane_id = 31 & threadIdx.x;
    (void)nnz;
    (void)begin;

    if (d_warp_num[warp_id + 1] > (d_warp_num[warp_id] + 1)) {
        row = d_warp_num[warp_id] + lane_id;
        if (row >= m)
            return;

        int col, j, i;
        double xi;
        double left_sum = 0.0;
        i = row;
        j = d_csrRowPtr[i];

        while (j < d_csrRowPtr[i + 1]) {
            col = d_csrColIdx[j];
            while (d_get_value[col] == 1) {
                left_sum += d_csrVal[j] * d_x[col];
                j++;
                if (j >= d_csrRowPtr[i + 1])
                    break;
                col = d_csrColIdx[j];
            }
            if (j >= d_csrRowPtr[i + 1])
                break;
            if (i == col) {
                xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i + 1] - 1];
                d_x[i] = xi;
                __threadfence();
                d_get_value[i] = 1;
                j++;
            }
        }
    } else {
        row = d_warp_num[warp_id];
        if (row >= m)
            return;

        int col, j;
        double xi, sum = 0.0;
        for (j = d_csrRowPtr[row] + lane_id; j < d_csrRowPtr[row + 1] - 1; j += 32) {
            col = d_csrColIdx[j];
            while (d_get_value[col] != 1) {
                __threadfence_block();
            }
            sum += d_x[col] * d_csrVal[j];
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }

        if (!lane_id) {
            xi = (d_b[row] - sum) / d_csrVal[d_csrRowPtr[row + 1] - 1];
            d_x[row] = xi;
            __threadfence();
            d_get_value[row] = 1;
        }
    }
}
"""


def _get_yy_kernel():
    global _YY_KERNEL
    if _YY_KERNEL is None:
        _YY_KERNEL = cp.RawKernel(_YY_KERNEL_SRC, "yysptrsv_csr_kernel")
    return _YY_KERNEL


def _to_host_array(arr, dtype=None) -> np.ndarray:
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr).astype(dtype) if dtype is not None else cp.asnumpy(arr)
    return np.asarray(arr, dtype=dtype)


def _to_cupy_array(arr, dtype=None):
    if isinstance(arr, cp.ndarray):
        return arr.astype(dtype, copy=False) if dtype is not None else arr
    return cp.asarray(arr, dtype=dtype)


def build_yy_warp_schedule(
    indptr,
    m: int | None = None,
    border: int = DEFAULT_BORDER,
    warp_size: int = WARP_SIZE,
) -> np.ndarray:
    indptr_h = _to_host_array(indptr, dtype=np.int64)
    if m is None:
        m = int(indptr_h.shape[0] - 1)
    if warp_size != WARP_SIZE:
        raise ValueError(f"YY replication requires warp_size == {WARP_SIZE}")
    if border < 0:
        raise ValueError("border must be >= 0")

    warp_num = np.zeros(m + 1, dtype=np.int32)
    row = 0
    k = 1

    while row < m:
        block_end = min(row + warp_size, m)
        rows_in_block = block_end - row
        element_n = int(indptr_h[block_end] - indptr_h[row])
        avg_element_n = element_n / rows_in_block

        if avg_element_n >= border:
            while row < block_end:
                row += 1
                warp_num[k] = row
                k += 1
        else:
            row = block_end
            warp_num[k] = row
            k += 1

    return warp_num[:k]


def upload_yy_warp_schedule(
    indptr,
    m: int | None = None,
    border: int = DEFAULT_BORDER,
):
    return cp.asarray(build_yy_warp_schedule(indptr, m=m, border=border))


def _exact_num_blocks(
    len_add: int,
    warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,
) -> int:
    if warps_per_block < 1:
        raise ValueError("warps_per_block must be >= 1")
    if len_add <= 1:
        return 1
    num_threads = warps_per_block * WARP_SIZE
    return int(math.ceil(((len_add - 1) * WARP_SIZE) / num_threads))


class SptrsvWarpspecializedCudaWorkspace:
    __slots__ = (
        "M",
        "N",
        "nnz",
        "len_add",
        "num_threads",
        "num_blocks",
        "kernel",
        "d_data",
        "d_indices",
        "d_indptr",
        "d_b",
        "d_x",
        "d_get_value",
        "d_warp_num",
        "d_id_extractor",
    )

    def __init__(
        self,
        data,
        indices,
        indptr,
        b,
        M: int | None = None,
        N: int | None = None,
        nnz: int | None = None,
        border: int = DEFAULT_BORDER,
        warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,
        *,
        warp_num=None,
    ):
        if M is None:
            M = int(_to_host_array(indptr, dtype=np.int64).shape[0] - 1)
        if N is None:
            N = M
        if M != N:
            raise ValueError("YY SpTRSV requires a square triangular matrix")
        if nnz is None:
            nnz = int(_to_host_array(indptr, dtype=np.int64)[M])

        self.M = int(M)
        self.N = int(N)
        self.nnz = int(nnz)

        self.d_data = _to_cupy_array(data, dtype=cp.float64)
        self.d_indices = _to_cupy_array(indices, dtype=cp.int32)
        self.d_indptr = _to_cupy_array(indptr, dtype=cp.int32)
        self.d_b = _to_cupy_array(b, dtype=cp.float64)

        if warp_num is None:
            warp_num_host = build_yy_warp_schedule(indptr, m=M, border=border)
            self.d_warp_num = cp.asarray(warp_num_host, dtype=cp.int32)
            self.len_add = int(warp_num_host.shape[0])
        else:
            self.d_warp_num = _to_cupy_array(warp_num, dtype=cp.int32)
            self.len_add = int(self.d_warp_num.shape[0])

        self.d_x = cp.empty(self.M, dtype=cp.float64)
        self.d_get_value = cp.zeros(self.M, dtype=cp.int32)
        self.d_id_extractor = cp.empty(1, dtype=cp.int32)

        self.num_threads = int(warps_per_block * WARP_SIZE)
        self.num_blocks = _exact_num_blocks(self.len_add, warps_per_block)
        self.kernel = _get_yy_kernel()

    def reset(self) -> None:
        self.d_x.fill(0)
        self.d_get_value.fill(0)
        self.d_id_extractor.fill(0)

    def run(self):
        self.reset()
        self.kernel(
            (self.num_blocks,),
            (self.num_threads,),
            (
                self.d_indptr,
                self.d_indices,
                self.d_data,
                self.d_get_value,
                np.int32(self.M),
                np.int32(self.nnz),
                self.d_b,
                self.d_x,
                np.int32(0),
                self.d_warp_num,
                np.int32(self.len_add),
                self.d_id_extractor,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        return self.d_x


def sptrsv_warpspecialized_cuda(
    data,
    indices,
    indptr,
    b,
    M: int | None = None,
    N: int | None = None,
    nnz: int | None = None,
    border: int = DEFAULT_BORDER,
    warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,
    *,
    warp_num=None,
):
    workspace = SptrsvWarpspecializedCudaWorkspace(
        data,
        indices,
        indptr,
        b,
        M=M,
        N=N,
        nnz=nnz,
        border=border,
        warps_per_block=warps_per_block,
        warp_num=warp_num,
    )
    return workspace.run()


def bench_sptrsv_warpspecialized_cuda(
    A_csr,
    rep: int = 100,
    warmup: int = 25,
    border: int = DEFAULT_BORDER,
    warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,
):
    m = int(A_csr.shape[0])
    b = np.ones(m, dtype=np.float64)

    workspace = SptrsvWarpspecializedCudaWorkspace(
        A_csr.data,
        A_csr.indices,
        A_csr.indptr,
        b,
        M=A_csr.shape[0],
        N=A_csr.shape[1],
        nnz=A_csr.nnz,
        border=border,
        warps_per_block=warps_per_block,
    )

    for _ in range(warmup):
        workspace.run()

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        workspace.run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times = np.asarray(times)
    return np.median(times), np.min(times), np.max(times)


__all__ = [
    "WARP_SIZE",
    "DEFAULT_BORDER",
    "DEFAULT_WARPS_PER_BLOCK",
    "SptrsvWarpspecializedCudaWorkspace",
    "bench_sptrsv_warpspecialized_cuda",
    "build_yy_warp_schedule",
    "upload_yy_warp_schedule",
    "sptrsv_warpspecialized_cuda",
]
