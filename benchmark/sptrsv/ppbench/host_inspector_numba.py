import numpy as np
import scipy.sparse as sp
from numba import njit


def _as_int32(arr):
    return np.asarray(arr, dtype=np.int32)


def _require_csc(L):
    if not sp.isspmatrix_csc(L):
        raise TypeError("expected a CSC matrix")


@njit(cache=True)
def _incoming_edge_csc_kernel(indptr, indices, upper_triangular):
    n = indptr.size - 1
    incoming_edges = np.zeros(n, dtype=np.int32)

    if not upper_triangular:
        for col in range(n):
            for jj in range(indptr[col], indptr[col + 1]):
                incoming_edges[indices[jj]] += 1
    else:
        for col in range(n):
            for jj in range(indptr[col], indptr[col + 1]):
                row = indices[jj]
                if row < col:
                    incoming_edges[row] += 1

    return incoming_edges


@njit(cache=True)
def _compute_wavefront_csc_kernel(indptr, indices, upper_triangular):
    n = indptr.size - 1
    waves = np.empty(n, dtype=np.int32)
    wave_ptr = np.empty(n + 1, dtype=np.int32)
    current_wave = np.empty(n, dtype=np.int32)
    next_wave = np.empty(n, dtype=np.int32)

    wave_ptr[0] = 0
    total_nodes = 0
    num_waves = 0

    if not upper_triangular:
        dependency = _incoming_edge_csc_kernel(indptr, indices, False)
        current_size = 0
        for row in range(n):
            if dependency[row] == 1:
                current_wave[current_size] = row
                current_size += 1

        while current_size > 0:
            for i in range(current_size):
                waves[total_nodes + i] = current_wave[i]
            total_nodes += current_size
            num_waves += 1
            wave_ptr[num_waves] = total_nodes

            next_size = 0
            for i in range(current_size):
                col = current_wave[i]
                for jj in range(indptr[col] + 1, indptr[col + 1]):
                    row = indices[jj]
                    dependency[row] -= 1
                    if dependency[row] == 1:
                        next_wave[next_size] = row
                        next_size += 1

            tmp = current_wave
            current_wave = next_wave
            next_wave = tmp
            current_size = next_size
    else:
        dependency = np.zeros(n, dtype=np.int32)
        processed = np.zeros(n, dtype=np.uint8)

        for col in range(n):
            for jj in range(indptr[col], indptr[col + 1]):
                row = indices[jj]
                if row < col:
                    dependency[row] += 1

        current_size = 0
        for row in range(n):
            if dependency[row] == 0:
                current_wave[current_size] = row
                current_size += 1
                processed[row] = 1

        while current_size > 0:
            for i in range(current_size):
                waves[total_nodes + i] = current_wave[i]
            total_nodes += current_size
            num_waves += 1
            wave_ptr[num_waves] = total_nodes

            next_size = 0
            for i in range(current_size):
                col = current_wave[i]
                for jj in range(indptr[col], indptr[col + 1]):
                    row = indices[jj]
                    if row < col:
                        dependency[row] -= 1
                        if dependency[row] == 0 and processed[row] == 0:
                            next_wave[next_size] = row
                            next_size += 1
                            processed[row] = 1

            tmp = current_wave
            current_wave = next_wave
            next_wave = tmp
            current_size = next_size

    return wave_ptr[:num_waves + 1], waves[:total_nodes]


@njit(cache=True)
def _find_root(parent, node):
    root = node
    while parent[root] != root:
        root = parent[root]

    curr = node
    while curr != root:
        next_node = parent[curr]
        parent[curr] = root
        curr = next_node

    return root


@njit(cache=True)
def _union(parent, lhs, rhs):
    lhs_root = _find_root(parent, lhs)
    rhs_root = _find_root(parent, rhs)
    if lhs_root != rhs_root:
        parent[lhs_root] = rhs_root


@njit(cache=True)
def build_h_level_structure_host(indptr, indices, wavefronts, wave_ptr, band_width=64):
    num_standard_waves = wave_ptr.size - 1
    n_rows = indptr.size - 1

    h_task_nodes = np.zeros(wavefronts.size, dtype=np.int32)
    task_ptr = np.zeros(wavefronts.size + 1, dtype=np.int32)
    h_level_ptr = np.zeros(num_standard_waves + 1, dtype=np.int32)
    global_to_local = np.full(n_rows, -1, dtype=np.int32)

    current_wave = 0
    node_count = 0
    task_count = 0
    h_level_count = 0

    while current_wave < num_standard_waves:
        end_wave = min(current_wave + band_width, num_standard_waves)
        band_start = wave_ptr[current_wave]
        band_end = wave_ptr[end_wave]
        band_size = band_end - band_start

        if band_size == 0:
            current_wave = end_wave
            continue

        parent = np.arange(band_size, dtype=np.int32)
        processed = np.zeros(band_size, dtype=np.uint8)

        for local_idx in range(band_size):
            global_to_local[wavefronts[band_start + local_idx]] = local_idx

        for local_idx in range(band_size):
            row = wavefronts[band_start + local_idx]
            for jj in range(indptr[row], indptr[row + 1]):
                col = indices[jj]
                local_col = global_to_local[col]
                if local_col != -1 and local_col != local_idx:
                    _union(parent, local_idx, local_col)

        for local_idx in range(band_size):
            parent[local_idx] = _find_root(parent, local_idx)

        for local_idx in range(band_size):
            if processed[local_idx] != 0:
                continue

            root = parent[local_idx]
            for scan_idx in range(local_idx, band_size):
                if parent[scan_idx] == root:
                    h_task_nodes[node_count] = wavefronts[band_start + scan_idx]
                    node_count += 1
                    processed[scan_idx] = 1

            task_count += 1
            task_ptr[task_count] = node_count

        for local_idx in range(band_size):
            global_to_local[wavefronts[band_start + local_idx]] = -1

        h_level_count += 1
        h_level_ptr[h_level_count] = task_count
        current_wave = end_wave

    return (
        h_task_nodes[:node_count],
        task_ptr[:task_count + 1],
        h_level_ptr[:h_level_count + 1],
    )


def compute_wavefront_numba(L, upper_triangular=False):
    _require_csc(L)
    return _compute_wavefront_csc_kernel(
        _as_int32(L.indptr),
        _as_int32(L.indices),
        upper_triangular,
    )


def incoming_edge_numba(L, upper_triangular=False):
    _require_csc(L)
    return _incoming_edge_csc_kernel(
        _as_int32(L.indptr),
        _as_int32(L.indices),
        upper_triangular,
    )


def sptrsv_h_level_analyze_numba_host(L, band_width=64, upper_triangular=False):
    wave_ptr, waves = compute_wavefront_numba(L, upper_triangular=upper_triangular)
    return build_h_level_structure_host(
        _as_int32(L.indptr),
        _as_int32(L.indices),
        waves,
        wave_ptr,
        band_width=band_width,
    )
