from __future__ import annotations

import math
import time

import numpy as np
from numba import config, cuda

from .sptrsv_warpspecialized import (
    DEFAULT_BORDER,
    DEFAULT_WARPS_PER_BLOCK,
    WARP_SIZE,
    build_yy_warp_schedule,
)

config.CUDA_ENABLE_PYNVJITLINK = 1

DEFAULT_RESET_THREADS = 256


def _to_host_array(arr, dtype=None) -> np.ndarray:
    if cuda.is_cuda_array(arr):
        return np.asarray(cuda.as_cuda_array(arr).copy_to_host(), dtype=dtype)
    return np.asarray(arr, dtype=dtype)


def _to_device_array(arr, dtype=None):
    if cuda.is_cuda_array(arr):
        return arr
    return cuda.to_device(np.asarray(arr, dtype=dtype))


def _infer_value_dtype(data, b):
    if cuda.is_cuda_array(b):
        return cuda.as_cuda_array(b).dtype
    if cuda.is_cuda_array(data):
        return cuda.as_cuda_array(data).dtype
    return np.asarray(b if b is not None else data).dtype


def _exact_num_blocks(
    len_add: int,
    warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,
) -> int:
    if warps_per_block < 1:
        raise ValueError("warps_per_block must be >= 1")
    max_tpb = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    if warps_per_block * WARP_SIZE > max_tpb:
        raise ValueError(
            f"warps_per_block * {WARP_SIZE} exceeds MAX_THREADS_PER_BLOCK"
        )
    if len_add <= 1:
        return 1
    num_threads = warps_per_block * WARP_SIZE
    return int(math.ceil(((len_add - 1) * WARP_SIZE) / num_threads))


@cuda.jit
def _fill_kernel(arr, value):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = value


@cuda.jit
def sptrsv_yy_replica_numba_kernel(
    d_csrRowPtr,
    d_csrColIdx,
    d_csrVal,
    d_get_value,
    m,
    nnz,
    d_b,
    d_x,
    begin,
    d_warp_num,
    Len,
    d_id_extractor,
):
    _ = nnz
    _ = begin

    global_id = cuda.atomic.add(d_id_extractor, 0, 1)
    warp_id = global_id // WARP_SIZE
    if warp_id >= (Len - 1):
        return

    lane_id = cuda.threadIdx.x & (WARP_SIZE - 1)

    if d_warp_num[warp_id + 1] > (d_warp_num[warp_id] + 1):
        row = d_warp_num[warp_id] + lane_id
        if row >= m:
            return

        left_sum = 0.0
        j = d_csrRowPtr[row]
        stop = d_csrRowPtr[row + 1]

        while j < stop:
            col = d_csrColIdx[j]
            while cuda.atomic.add(d_get_value, col, 0) == 1:
                left_sum += d_csrVal[j] * d_x[col]
                j += 1
                if j >= stop:
                    break
                col = d_csrColIdx[j]

            if j >= stop:
                break

            if row == col:
                xi = (d_b[row] - left_sum) / d_csrVal[d_csrRowPtr[row + 1] - 1]
                d_x[row] = xi
                cuda.threadfence()
                d_get_value[row] = 1
                j += 1
    else:
        row = d_warp_num[warp_id]
        if row >= m:
            return

        sum_val = 0.0
        j = d_csrRowPtr[row] + lane_id
        stop = d_csrRowPtr[row + 1] - 1

        while j < stop:
            col = d_csrColIdx[j]
            while cuda.atomic.add(d_get_value, col, 0) != 1:
                cuda.threadfence_block()
            sum_val += d_x[col] * d_csrVal[j]
            j += WARP_SIZE

        offset = WARP_SIZE // 2
        mask = cuda.activemask()
        while offset > 0:
            sum_val += cuda.shfl_down_sync(mask, sum_val, offset)
            offset //= 2

        if lane_id == 0:
            xi = (d_b[row] - sum_val) / d_csrVal[d_csrRowPtr[row + 1] - 1]
            d_x[row] = xi
            cuda.threadfence()
            d_get_value[row] = 1


class SptrsvWarpspecializedNumbaWorkspace:
    __slots__ = (
        "M",
        "N",
        "nnz",
        "len_add",
        "num_threads",
        "num_blocks",
        "reset_blocks",
        "zero_value",
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
        self.d_data = _to_device_array(data)
        self.d_indices = _to_device_array(indices, dtype=np.int32)
        self.d_indptr = _to_device_array(indptr, dtype=np.int32)
        self.d_b = _to_device_array(b)

        if warp_num is None:
            warp_num_host = build_yy_warp_schedule(indptr, m=M, border=border)
            self.d_warp_num = cuda.to_device(warp_num_host)
            self.len_add = int(warp_num_host.shape[0])
        elif cuda.is_cuda_array(warp_num):
            self.d_warp_num = warp_num
            self.len_add = int(cuda.as_cuda_array(warp_num).shape[0])
        else:
            warp_num_host = np.asarray(warp_num, dtype=np.int32)
            self.d_warp_num = cuda.to_device(warp_num_host)
            self.len_add = int(warp_num_host.shape[0])

        value_dtype = _infer_value_dtype(data, b)
        self.zero_value = value_dtype.type(0)
        self.d_x = cuda.device_array(self.M, dtype=value_dtype)
        self.d_get_value = cuda.device_array(self.M, dtype=np.int32)
        self.d_id_extractor = cuda.device_array(1, dtype=np.int32)

        self.num_threads = warps_per_block * WARP_SIZE
        self.num_blocks = _exact_num_blocks(self.len_add, warps_per_block)
        self.reset_blocks = max(
            1, (self.M + DEFAULT_RESET_THREADS - 1) // DEFAULT_RESET_THREADS
        )

    def reset(self) -> None:
        _fill_kernel[self.reset_blocks, DEFAULT_RESET_THREADS](
            self.d_x, self.zero_value
        )
        _fill_kernel[self.reset_blocks, DEFAULT_RESET_THREADS](
            self.d_get_value, np.int32(0)
        )
        _fill_kernel[1, 1](self.d_id_extractor, np.int32(0))

    def run(self):
        self.reset()
        sptrsv_yy_replica_numba_kernel[self.num_blocks, self.num_threads](
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
        )
        cuda.synchronize()
        return self.d_x


def sptrsv_warpspecialized_numba(
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
    workspace = SptrsvWarpspecializedNumbaWorkspace(
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


def bench_sptrsv_warpspecialized_numba(
    A_csr,
    rep: int = 100,
    warmup: int = 25,
    border: int = DEFAULT_BORDER,
    warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,
):
    m = int(A_csr.shape[0])
    b = np.ones(m, dtype=np.float64)

    workspace = SptrsvWarpspecializedNumbaWorkspace(
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
    "SptrsvWarpspecializedNumbaWorkspace",
    "bench_sptrsv_warpspecialized_numba",
    "sptrsv_warpspecialized_numba",
    "sptrsv_yy_replica_numba_kernel",
]
