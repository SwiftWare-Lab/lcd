from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import cupy as cp
from numba import cuda, config
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
from numba.core.errors import NumbaPerformanceWarning

config.CUDA_ENABLE_PYNVJITLINK = 1
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

WARP_SIZE = 32
DEFAULT_NNZ_THRESHOLD = 10
DEFAULT_MAX_CHUNK = 32


def _to_device_int32(arr):
    if isinstance(arr, DeviceNDArrayBase):
        return arr
    return cuda.to_device(np.asarray(arr, dtype=np.int32))


def _validate_warps_per_block(warps_per_block: int) -> None:
    if warps_per_block < 1:
        raise ValueError("warps_per_block must be >= 1")
    max_tpb = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    if warps_per_block * WARP_SIZE > max_tpb:
        raise ValueError(f"warps_per_block * {WARP_SIZE} exceeds MAX_THREADS_PER_BLOCK")


# =============================================================================
# Task 1a — CSR sync-free: split into heavy rows vs light row-bundles + ordered runs
# =============================================================================


def build_csr_syncfree_task_split(
    indptr,
    m: int | None = None,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_chunk: int = DEFAULT_MAX_CHUNK,
) -> dict[str, np.ndarray | list]:
    """
    Classify CSR rows in natural order into:

    - **heavy_tasks**: row indices with nnz > ``nnz_threshold`` (one warp task each).
    - **light_bundle_tasks**: each entry is a half-open row range ``[lo, hi)`` of up to
      ``max_chunk`` consecutive **short** rows (one thread task each).

    Also returns **runs**: a list of ``("heavy", rows_np)`` or ``("light", lo_np, hi_np)``
    where consecutive segments of the same kind are merged. Launch order must follow
    ``runs`` for sync-free correctness.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    if m is None:
        m = int(indptr.shape[0] - 1)
    if max_chunk > WARP_SIZE:
        max_chunk = WARP_SIZE
    if nnz_threshold < 0 or max_chunk < 1:
        raise ValueError("invalid nnz_threshold or max_chunk")

    segments: list[tuple[str, ...]] = []
    row = 0
    while row < m:
        row_nnz = int(indptr[row + 1] - indptr[row])
        if row_nnz > nnz_threshold:
            segments.append(("heavy", int(row)))
            row += 1
        else:
            rb = row
            cnt = 0
            while row < m and cnt < max_chunk:
                rn = int(indptr[row + 1] - indptr[row])
                if rn > nnz_threshold:
                    break
                row += 1
                cnt += 1
            segments.append(("light", int(rb), int(row)))

    runs: list[tuple[str, ...]] = []
    for seg in segments:
        kind = seg[0]
        if not runs or runs[-1][0] != kind:
            if kind == "heavy":
                runs.append(("heavy", [seg[1]]))
            else:
                runs.append(("light", [(seg[1], seg[2])]))
        else:
            if kind == "heavy":
                runs[-1][1].append(seg[1])
            else:
                runs[-1][1].append((seg[1], seg[2]))

    heavy_list: list[int] = []
    light_lo: list[int] = []
    light_hi: list[int] = []
    run_desc: list[tuple[str, np.ndarray, ...]] = []

    for run in runs:
        if run[0] == "heavy":
            rows = np.array(run[1], dtype=np.int32)
            heavy_list.extend(int(x) for x in rows.tolist())
            run_desc.append(("heavy", rows))
        else:
            pairs = run[1]
            lo = np.array([p[0] for p in pairs], dtype=np.int32)
            hi = np.array([p[1] for p in pairs], dtype=np.int32)
            light_lo.extend(lo.tolist())
            light_hi.extend(hi.tolist())
            run_desc.append(("light", lo, hi))

    return {
        "heavy_tasks": np.array(heavy_list, dtype=np.int32),
        "light_bundle_lo": np.array(light_lo, dtype=np.int32),
        "light_bundle_hi": np.array(light_hi, dtype=np.int32),
        "runs_host": run_desc,
    }


class CsrSfUploadedSchedule:
    """Device-resident CSR sync-free run schedule (from :func:`upload_csr_syncfree_schedule`)."""

    __slots__ = ("runs",)

    def __init__(self, runs):
        self.runs = runs  # list of ('heavy', d_rows) or ('light', d_lo, d_hi)


def upload_csr_syncfree_schedule(split: dict) -> CsrSfUploadedSchedule:
    """Upload ``runs_host`` from :func:`build_csr_syncfree_task_split` to the GPU."""
    dev_runs = []
    for item in split["runs_host"]:
        if item[0] == "heavy":
            dev_runs.append(("heavy", _to_device_int32(item[1])))
        else:
            dev_runs.append(("light", _to_device_int32(item[1]), _to_device_int32(item[2])))
    return CsrSfUploadedSchedule(dev_runs)


def prepare_csr_syncfree_hybrid_persistent_assets(
    indptr,
    m: int | None = None,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_chunk: int = DEFAULT_MAX_CHUNK,
):
    """
    Build and upload the CSR sync-free split schedule (replaces monolithic task_starts).

    Returns :class:`CsrSfUploadedSchedule` for use with :func:`sptrsv_csr_syncfree_hybrid_persistent`.
    """
    indptr_h = np.asarray(indptr, dtype=np.int64)
    if m is None:
        m = int(indptr_h.shape[0] - 1)
    split = build_csr_syncfree_task_split(
        indptr_h, m=m, nnz_threshold=nnz_threshold, max_chunk=max_chunk
    )
    return upload_csr_syncfree_schedule(split)


# Kept for external callers that only need host arrays (e.g. inspection / tests).
def build_contiguous_csr_task_schedule(
    indptr,
    m: int | None = None,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_chunk: int = DEFAULT_MAX_CHUNK,
) -> np.ndarray:
    """
    Deprecated cumulative layout; prefer :func:`build_csr_syncfree_task_split`.

    Reconstructs a YY-style ``task_starts`` boundary array from the split for compatibility.
    """
    sp = build_csr_syncfree_task_split(indptr, m, nnz_threshold, max_chunk)
    boundaries = [0]
    for run in sp["runs_host"]:
        if run[0] == "heavy":
            for r in np.asarray(run[1], dtype=np.int64):
                boundaries.append(int(r) + 1)
        else:
            hi = np.asarray(run[2], dtype=np.int64)
            for h in hi:
                boundaries.append(int(h))
    return np.array(boundaries, dtype=np.int32)


# =============================================================================
# Task 2a — CSR sync-free kernels (heavy warp / light warp, lane-per-row bundles)
# =============================================================================


@cuda.jit
def _csr_sf_heavy_rows_warp_kernel(
    data,
    indices,
    indptr,
    b,
    y,
    M,
    heavy_rows,
    n_heavy,
    dep_counts,
    succ_offsets,
    succ_indices,
):
    lane_id = cuda.threadIdx.x & (WARP_SIZE - 1)
    wid = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // WARP_SIZE
    if wid >= n_heavy:
        return
    row = heavy_rows[wid]
    if row >= M:
        return
    if lane_id == 0:
        while cuda.atomic.add(dep_counts, row, 0) > 0:
            pass
    cuda.syncwarp()

    row_start = indptr[row]
    row_end_r = indptr[row + 1]
    partial = 0.0
    for j in range(row_start + lane_id, row_end_r - 1, WARP_SIZE):
        partial += data[j] * y[indices[j]]
    for offset in (16, 8, 4, 2, 1):
        partial += cuda.shfl_down_sync(0xFFFFFFFF, partial, offset)
    if lane_id == 0:
        y[row] = (b[row] - partial) / data[row_end_r - 1]
        cuda.threadfence()
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            cuda.atomic.add(dep_counts, succ_indices[p], -1)


@cuda.jit
def _csr_sf_light_bundles_warp_kernel(
    data,
    indices,
    indptr,
    b,
    y,
    M,
    bundle_lo,
    bundle_hi,
    n_bundles,
    dep_counts,
    succ_offsets,
    succ_indices,
):
    lane_id = cuda.threadIdx.x & (WARP_SIZE - 1)
    wid = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // WARP_SIZE
    if wid >= n_bundles:
        return
    row_begin = bundle_lo[wid]
    row_end = bundle_hi[wid]
    row = row_begin + lane_id
    if row < row_end and row < M:
        while cuda.atomic.add(dep_counts, row, 0) > 0:
            pass
        row_start = indptr[row]
        row_end_r = indptr[row + 1]
        acc = 0.0
        for j in range(row_start, row_end_r - 1):
            acc += data[j] * y[indices[j]]
        y[row] = (b[row] - acc) / data[row_end_r - 1]
        cuda.threadfence()
        s0 = succ_offsets[row]
        s1 = succ_offsets[row + 1]
        for p in range(s0, s1):
            cuda.atomic.add(dep_counts, succ_indices[p], -1)


def _csr_sf_max_blocks(num_sms: int | None) -> int:
    if num_sms is None:
        num_sms = getattr(cuda.get_current_device(), "MULTIPROCESSOR_COUNT", 80)
    return max(1, min(1024, num_sms * 4))


def _launch_csr_sf_schedule_runs(
    schedule: CsrSfUploadedSchedule,
    data,
    indices,
    indptr,
    b,
    d_y,
    M: int,
    d_dep,
    d_succ_off,
    d_succ_idx,
    warps_per_block: int,
    max_blocks: int,
    nb_stream,
):
    """Dispatch heavy-warp / light-warp kernels for each run (optionally on ``nb_stream``)."""
    threads_warp = warps_per_block * WARP_SIZE
    m_i32 = np.int32(M)

    for run in schedule.runs:
        if run[0] == "heavy":
            d_rows = run[1]
            nh = int(d_rows.shape[0])
            if nh <= 0:
                continue
            blocks = min(max_blocks, (nh + warps_per_block - 1) // warps_per_block)
            blocks = max(1, blocks)
            if nb_stream is None:
                _csr_sf_heavy_rows_warp_kernel[blocks, threads_warp](
                    data,
                    indices,
                    indptr,
                    b,
                    d_y,
                    m_i32,
                    d_rows,
                    np.int32(nh),
                    d_dep,
                    d_succ_off,
                    d_succ_idx,
                )
            else:
                _csr_sf_heavy_rows_warp_kernel[blocks, threads_warp, nb_stream](
                    data,
                    indices,
                    indptr,
                    b,
                    d_y,
                    m_i32,
                    d_rows,
                    np.int32(nh),
                    d_dep,
                    d_succ_off,
                    d_succ_idx,
                )
        else:
            d_lo, d_hi = run[1], run[2]
            nb = int(d_lo.shape[0])
            if nb <= 0:
                continue
            tpb = warps_per_block * WARP_SIZE
            blocks = max(1, min(max_blocks, (nb + warps_per_block - 1) // warps_per_block))
            if nb_stream is None:
                _csr_sf_light_bundles_warp_kernel[blocks, tpb](
                    data,
                    indices,
                    indptr,
                    b,
                    d_y,
                    m_i32,
                    d_lo,
                    d_hi,
                    np.int32(nb),
                    d_dep,
                    d_succ_off,
                    d_succ_idx,
                )
            else:
                _csr_sf_light_bundles_warp_kernel[blocks, tpb, nb_stream](
                    data,
                    indices,
                    indptr,
                    b,
                    d_y,
                    m_i32,
                    d_lo,
                    d_hi,
                    np.int32(nb),
                    d_dep,
                    d_succ_off,
                    d_succ_idx,
                )


class CsrSfLoadbalancedGraphSolver:
    """
    Capture the full CSR load-balanced multi-launch sequence once (CuPy CUDA graph),
    then replay it with cheap H2D resets for ``y`` and dependency counts.

    Use this for benchmarking or repeated solves with fixed matrix structure; it avoids
    thousands of Python/Numba launch overheads per call.
    """

    __slots__ = (
        "M",
        "d_y",
        "d_dep",
        "_graph",
        "_stream",
        "_nb_stream",
        "_schedule",
        "_warps_per_block",
        "_max_blocks",
        "_d_data",
        "_d_indices",
        "_d_indptr",
        "_d_b",
        "_d_succ_off",
        "_d_succ_idx",
    )

    def __init__(
        self,
        schedule: CsrSfUploadedSchedule,
        d_data,
        d_indices,
        d_indptr,
        d_b,
        M: int,
        in_edges_template,
        succ_offsets,
        succ_indices,
        warps_per_block: int = 8,
        num_sms: int | None = None,
    ):
        _validate_warps_per_block(warps_per_block)
        self.M = int(M)
        self._schedule = schedule
        self._warps_per_block = warps_per_block
        self._max_blocks = _csr_sf_max_blocks(num_sms)
        self._d_data = d_data
        self._d_indices = d_indices
        self._d_indptr = d_indptr
        self._d_b = d_b
        self.d_y = cuda.to_device(np.zeros(self.M, dtype=np.float64))
        self.d_dep = _to_device_int32(np.asarray(in_edges_template, dtype=np.int32).copy())
        self._d_succ_off = _to_device_int32(succ_offsets)
        self._d_succ_idx = _to_device_int32(succ_indices)

        self._stream = cp.cuda.Stream(non_blocking=True)
        self._nb_stream = cuda.external_stream(self._stream.ptr)
        self._stream.begin_capture()
        _launch_csr_sf_schedule_runs(
            schedule,
            d_data,
            d_indices,
            d_indptr,
            d_b,
            self.d_y,
            self.M,
            self.d_dep,
            self._d_succ_off,
            self._d_succ_idx,
            warps_per_block,
            self._max_blocks,
            self._nb_stream,
        )
        self._graph = self._stream.end_capture()

    def run(self, in_edges):
        """Zero ``y``, refresh dependency counts, replay captured graph, synchronize."""
        self.d_y.copy_to_device(
            np.zeros(self.M, dtype=np.float64), stream=self._nb_stream
        )
        self.d_dep.copy_to_device(
            np.asarray(in_edges, dtype=np.int32), stream=self._nb_stream
        )
        self._graph.launch(stream=self._stream)
        self._stream.synchronize()
        return self.d_y


def sptrsv_csr_syncfree_hybrid_persistent(
    data,
    indices,
    indptr,
    b,
    M: int,
    N: int,
    nnz: int,
    in_edges,
    succ_offsets,
    succ_indices,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_chunk: int = DEFAULT_MAX_CHUNK,
    warps_per_block: int = 8,
    num_sms: int | None = None,
    *,
    schedule: CsrSfUploadedSchedule | None = None,
    d_succ_off=None,
    d_succ_idx=None,
):
    """
    Ordered multi-launch CSR sync-free: for each run in ``schedule``, launch the heavy
    (warp) or light (warp, one warp per bundle) kernel. No persistent task counter.

    Pass ``schedule`` from :func:`prepare_csr_syncfree_hybrid_persistent_assets`. If omitted,
    builds and uploads a schedule from ``indptr`` each call.
    """
    _validate_warps_per_block(warps_per_block)
    if schedule is None:
        if hasattr(indptr, "copy_to_host"):
            indptr_h = np.asarray(indptr.copy_to_host(), dtype=np.int64)
        else:
            indptr_h = np.asarray(indptr, dtype=np.int64)
        split = build_csr_syncfree_task_split(
            indptr_h, m=M, nnz_threshold=nnz_threshold, max_chunk=max_chunk
        )
        schedule = upload_csr_syncfree_schedule(split)

    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    d_dep = _to_device_int32(np.asarray(in_edges, dtype=np.int32).copy())
    if d_succ_off is None:
        d_succ_off = _to_device_int32(succ_offsets)
    if d_succ_idx is None:
        d_succ_idx = _to_device_int32(succ_indices)

    max_blocks = _csr_sf_max_blocks(num_sms)
    _launch_csr_sf_schedule_runs(
        schedule,
        data,
        indices,
        indptr,
        b,
        d_y,
        M,
        d_dep,
        d_succ_off,
        d_succ_idx,
        warps_per_block,
        max_blocks,
        None,
    )
    return d_y


# =============================================================================
# Task 1b — H-level: separate heavy chain list vs light chain list per level
# =============================================================================


def build_hlevel_bundle_schedule_cpu(
    indptr: np.ndarray,
    h_task_nodes: np.ndarray,
    task_ptr: np.ndarray,
    h_level_ptr: np.ndarray,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_bundle_chains: int = WARP_SIZE,
    max_light_chain_rows: int = WARP_SIZE,
) -> dict:
    """
    Split each level into:

    - **heavy_tasks**: ``(node_lo, node_hi)`` ranges in ``h_task_nodes`` per heavy chain.
    - **light_bundle_tasks**: one entry per **light** chain (same tuple shape); the
      load-balanced kernel assigns one lane per light chain (coalesced warp scheduling).

    Per-level pointers index into the concatenated heavy / light task tables.
    """
    _ = max_bundle_chains  # retained for callers; each light chain is its own lane task
    indptr = np.asarray(indptr, dtype=np.int64)
    h_task_nodes = np.asarray(h_task_nodes, dtype=np.int32)
    task_ptr = np.asarray(task_ptr, dtype=np.int32)
    h_level_ptr = np.asarray(h_level_ptr, dtype=np.int32)

    heavy_lo: list[int] = []
    heavy_hi: list[int] = []
    light_lo: list[int] = []
    light_hi: list[int] = []

    level_heavy_ptr: list[int] = [0]
    level_light_ptr: list[int] = [0]

    num_levels = len(h_level_ptr) - 1
    for lev in range(num_levels):
        c0 = int(h_level_ptr[lev])
        c1 = int(h_level_ptr[lev + 1])
        for ci in range(c0, c1):
            s = int(task_ptr[ci])
            e = int(task_ptr[ci + 1])
            if _chain_is_light(
                ci, indptr, h_task_nodes, task_ptr,
                nnz_threshold, max_light_chain_rows,
            ):
                light_lo.append(s)
                light_hi.append(e)
            else:
                heavy_lo.append(s)
                heavy_hi.append(e)
        level_heavy_ptr.append(len(heavy_lo))
        level_light_ptr.append(len(light_lo))

    return {
        "heavy_lo": np.array(heavy_lo, dtype=np.int32),
        "heavy_hi": np.array(heavy_hi, dtype=np.int32),
        "light_lo": np.array(light_lo, dtype=np.int32),
        "light_hi": np.array(light_hi, dtype=np.int32),
        "level_heavy_ptr": np.array(level_heavy_ptr, dtype=np.int32),
        "level_light_ptr": np.array(level_light_ptr, dtype=np.int32),
    }


def _chain_is_light(
    chain_idx: int,
    indptr: np.ndarray,
    h_task_nodes: np.ndarray,
    task_ptr: np.ndarray,
    nnz_threshold: int,
    max_light_chain_rows: int,
) -> bool:
    s = int(task_ptr[chain_idx])
    e = int(task_ptr[chain_idx + 1])
    if e - s > max_light_chain_rows:
        return False
    for ni in range(s, e):
        row = int(h_task_nodes[ni])
        nnz_r = int(indptr[row + 1] - indptr[row])
        if nnz_r > nnz_threshold:
            return False
    return True


from .sptrsv_hlevel import build_h_level_structure


def prepare_hlevel_loadbalanced_host(
    indptr,
    indices,
    wavefronts,
    wave_start,
    band_width: int = 64,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_bundle_chains: int = WARP_SIZE,
    max_light_chain_rows: int = WARP_SIZE,
):
    """
    Returns GPU arrays + host level pointers for :func:`sptrsv_csr_hlevel_loadbalanced_solve`:

    ``(d_h_nodes, d_heavy_lo, d_heavy_hi, d_light_lo, d_light_hi,
    level_heavy_ptr, level_light_ptr)``
    """
    indptr_h = np.asarray(indptr, dtype=np.int32)
    indices_h = np.asarray(indices, dtype=np.int32)
    wavefronts_h = np.asarray(wavefronts, dtype=np.int32)
    wave_start_h = np.asarray(wave_start, dtype=np.int32)

    h_nodes, task_ptr, h_level_ptr = build_h_level_structure(
        indptr_h, indices_h, wavefronts_h, wave_start_h, band_width=band_width
    )
    indptr_64 = np.asarray(indptr_h, dtype=np.int64)
    sched = build_hlevel_bundle_schedule_cpu(
        indptr_64,
        h_nodes,
        task_ptr,
        h_level_ptr,
        nnz_threshold=nnz_threshold,
        max_bundle_chains=max_bundle_chains,
        max_light_chain_rows=max_light_chain_rows,
    )

    return (
        _to_device_int32(h_nodes),
        _to_device_int32(sched["heavy_lo"]),
        _to_device_int32(sched["heavy_hi"]),
        _to_device_int32(sched["light_lo"]),
        _to_device_int32(sched["light_hi"]),
        sched["level_heavy_ptr"],
        sched["level_light_ptr"],
    )


def prepare_hlevel_loadbalanced_analysis(
    indptr,
    analysis_data,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    max_bundle_chains: int = WARP_SIZE,
    max_light_chain_rows: int = WARP_SIZE,
):
    """Same return shape as :func:`prepare_hlevel_loadbalanced_host` (split layout)."""
    d_h_nodes, d_task_ptr, h_level_ptr = analysis_data

    if hasattr(indptr, "copy_to_host"):
        indptr_h = np.asarray(indptr.copy_to_host(), dtype=np.int64)
    else:
        indptr_h = np.asarray(indptr, dtype=np.int64)

    h_task_nodes = (
        d_h_nodes.copy_to_host()
        if hasattr(d_h_nodes, "copy_to_host")
        else np.asarray(d_h_nodes, dtype=np.int32)
    )
    task_ptr_h = (
        d_task_ptr.copy_to_host()
        if hasattr(d_task_ptr, "copy_to_host")
        else np.asarray(d_task_ptr, dtype=np.int32)
    )
    h_lp = np.asarray(h_level_ptr, dtype=np.int32)

    sched = build_hlevel_bundle_schedule_cpu(
        indptr_h,
        h_task_nodes,
        task_ptr_h,
        h_lp,
        nnz_threshold=nnz_threshold,
        max_bundle_chains=max_bundle_chains,
        max_light_chain_rows=max_light_chain_rows,
    )

    return (
        d_h_nodes,
        _to_device_int32(sched["heavy_lo"]),
        _to_device_int32(sched["heavy_hi"]),
        _to_device_int32(sched["light_lo"]),
        _to_device_int32(sched["light_hi"]),
        sched["level_heavy_ptr"],
        sched["level_light_ptr"],
    )


# =============================================================================
# Task 2b — H-level kernels (heavy warp / light warp, lane-per-chain tasks)
# =============================================================================


@cuda.jit
def _csr_hlevel_heavy_chains_warp_kernel(
    data,
    indices,
    indptr,
    b,
    y,
    M,
    h_task_nodes,
    heavy_lo,
    heavy_hi,
    n_tasks,
    task_base,
    nnz_threshold,
):
    lane_id = cuda.threadIdx.x & (WARP_SIZE - 1)
    wid = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // WARP_SIZE
    if wid >= n_tasks:
        return
    t = task_base + wid
    node_start = heavy_lo[t]
    node_end = heavy_hi[t]

    for ni in range(node_start, node_end):
        row = h_task_nodes[ni]
        if row >= M:
            continue
        row_start = indptr[row]
        row_end_r = indptr[row + 1]
        row_nnz = row_end_r - row_start

        if row_nnz > nnz_threshold:
            partial = 0.0
            for j in range(row_start + lane_id, row_end_r - 1, WARP_SIZE):
                partial += data[j] * y[indices[j]]
            for offset_r in (16, 8, 4, 2, 1):
                partial += cuda.shfl_down_sync(0xFFFFFFFF, partial, offset_r)
            if lane_id == 0:
                y[row] = (b[row] - partial) / data[row_end_r - 1]
            cuda.syncwarp()
        else:
            if lane_id == 0:
                acc = 0.0
                for j in range(row_start, row_end_r - 1):
                    acc += data[j] * y[indices[j]]
                y[row] = (b[row] - acc) / data[row_end_r - 1]
            cuda.syncwarp()


@cuda.jit
def _csr_hlevel_light_chains_warp_kernel(
    data,
    indices,
    indptr,
    b,
    y,
    M,
    h_task_nodes,
    light_lo,
    light_hi,
    n_tasks,
    task_base,
):
    lane_id = cuda.threadIdx.x & (WARP_SIZE - 1)
    wid = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // WARP_SIZE
    t = task_base + wid * WARP_SIZE + lane_id
    if t >= task_base + n_tasks:
        return
    node_start = light_lo[t]
    node_end = light_hi[t]

    for ni in range(node_start, node_end):
        row = h_task_nodes[ni]
        if row >= M:
            continue
        row_start = indptr[row]
        row_end_r = indptr[row + 1]
        acc = 0.0
        for j in range(row_start, row_end_r - 1):
            acc += data[j] * y[indices[j]]
        y[row] = (b[row] - acc) / data[row_end_r - 1]


def sptrsv_csr_hlevel_loadbalanced_solve(
    data,
    indices,
    indptr,
    b,
    M: int,
    N: int,
    nnz: int,
    loadbalanced_analysis,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    warps_per_block: int = 4,
):
    """
    ``loadbalanced_analysis`` is the 7-tuple from ``prepare_hlevel_loadbalanced_*``:

    ``(d_h_nodes, d_heavy_lo, d_heavy_hi, d_light_lo, d_light_hi,
    level_heavy_ptr, level_light_ptr)``.

    Per level: launch heavy chain kernel (one warp per heavy task), then light chain
    kernel using the same block shape (``threads_warp``), with one lane per light task.
    """
    import cupy as cp

    _validate_warps_per_block(warps_per_block)
    (
        d_h_nodes,
        d_heavy_lo,
        d_heavy_hi,
        d_light_lo,
        d_light_hi,
        level_heavy_ptr,
        level_light_ptr,
    ) = loadbalanced_analysis

    d_y = cuda.to_device(np.zeros(M, dtype=np.float64))
    num_h_levels = len(level_heavy_ptr) - 1

    threads_warp = warps_per_block * WARP_SIZE
    cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    cp_stream.begin_capture()

    for lev in range(num_h_levels):
        ha = int(level_heavy_ptr[lev])
        hb = int(level_heavy_ptr[lev + 1])
        nh = hb - ha
        if nh > 0:
            blocks = (nh + warps_per_block - 1) // warps_per_block
            _csr_hlevel_heavy_chains_warp_kernel[blocks, threads_warp, nb_stream](
                data,
                indices,
                indptr,
                b,
                d_y,
                np.int32(M),
                d_h_nodes,
                d_heavy_lo,
                d_heavy_hi,
                np.int32(nh),
                np.int32(ha),
                np.int32(nnz_threshold),
            )

        la = int(level_light_ptr[lev])
        lb = int(level_light_ptr[lev + 1])
        nl = lb - la
        if nl > 0:
            num_light_warps = (nl + WARP_SIZE - 1) // WARP_SIZE
            blocks = (num_light_warps + warps_per_block - 1) // warps_per_block
            _csr_hlevel_light_chains_warp_kernel[blocks, threads_warp, nb_stream](
                data,
                indices,
                indptr,
                b,
                d_y,
                np.int32(M),
                d_h_nodes,
                d_light_lo,
                d_light_hi,
                np.int32(nl),
                np.int32(la),
            )

    graph = cp_stream.end_capture()
    graph.launch(stream=cp_stream)
    cp_stream.synchronize()
    return d_y


def sptrsv_csr_hlevel_hybrid_solve(
    data,
    indices,
    indptr,
    b,
    M: int,
    N: int,
    nnz: int,
    analysis_data,
    nnz_threshold: int = DEFAULT_NNZ_THRESHOLD,
    warps_per_block: int = 4,
    max_bundle_chains: int = WARP_SIZE,
    max_light_chain_rows: int = WARP_SIZE,
):
    lb = prepare_hlevel_loadbalanced_analysis(
        indptr,
        analysis_data,
        nnz_threshold=nnz_threshold,
        max_bundle_chains=max_bundle_chains,
        max_light_chain_rows=max_light_chain_rows,
    )
    return sptrsv_csr_hlevel_loadbalanced_solve(
        data,
        indices,
        indptr,
        b,
        M,
        N,
        nnz,
        lb,
        nnz_threshold=nnz_threshold,
        warps_per_block=warps_per_block,
    )


__all__ = [
    "WARP_SIZE",
    "DEFAULT_NNZ_THRESHOLD",
    "DEFAULT_MAX_CHUNK",
    "CsrSfUploadedSchedule",
    "CsrSfLoadbalancedGraphSolver",
    "build_csr_syncfree_task_split",
    "build_contiguous_csr_task_schedule",
    "upload_csr_syncfree_schedule",
    "prepare_csr_syncfree_hybrid_persistent_assets",
    "build_hlevel_bundle_schedule_cpu",
    "prepare_hlevel_loadbalanced_host",
    "prepare_hlevel_loadbalanced_analysis",
    "sptrsv_csr_syncfree_hybrid_persistent",
    "sptrsv_csr_hlevel_loadbalanced_solve",
    "sptrsv_csr_hlevel_hybrid_solve",
]
