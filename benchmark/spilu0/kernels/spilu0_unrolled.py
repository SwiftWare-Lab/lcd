import sys
import os
# Add project root to path for compiler module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# Add benchmark directory to path for bench_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from numba import cuda
from numba import int32, float64

@cuda.jit(device=True, inline=True)
def bin_search_col(colind, start, end, key):
    lo = start
    hi = end - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        c = colind[mid]
        if c < key:
            lo = mid + 1
        elif c > key:
            hi = mid - 1
        else:
            return mid
    return -1

# ============================================================================
# Wavefront Unrolling Kernels (5 variants: 4, 8, 16, 32, 64)
# ============================================================================

@cuda.jit
def ilu0_csr_wavefront_unrolling_kernel_4(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            j_ptr = k_ptr + 1 + tid
            while j_ptr + 3 * cuda.blockDim.x < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                    
                jcol = indices[j_ptr + cuda.blockDim.x]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr + cuda.blockDim.x] -= l_ik * data[pos]
                    
                jcol = indices[j_ptr + 2 * cuda.blockDim.x]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr + 2 * cuda.blockDim.x] -= l_ik * data[pos]
                    
                jcol = indices[j_ptr + 3 * cuda.blockDim.x]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr + 3 * cuda.blockDim.x] -= l_ik * data[pos]
                    
                j_ptr += 4 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x

        cuda.syncthreads()
        k_ptr += 1


@cuda.jit
def ilu0_csr_wavefront_unrolling_kernel_8(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            j_ptr = k_ptr + 1 + tid
            while j_ptr + 7 * cuda.blockDim.x < row_end:
                for unroll_idx in range(8):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 8 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x

        cuda.syncthreads()
        k_ptr += 1


@cuda.jit
def ilu0_csr_wavefront_unrolling_kernel_16(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            j_ptr = k_ptr + 1 + tid
            while j_ptr + 15 * cuda.blockDim.x < row_end:
                for unroll_idx in range(16):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 16 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x

        cuda.syncthreads()
        k_ptr += 1


@cuda.jit
def ilu0_csr_wavefront_unrolling_kernel_32(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            j_ptr = k_ptr + 1 + tid
            while j_ptr + 31 * cuda.blockDim.x < row_end:
                for unroll_idx in range(32):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 32 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x

        cuda.syncthreads()
        k_ptr += 1


@cuda.jit
def ilu0_csr_wavefront_unrolling_kernel_64(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            j_ptr = k_ptr + 1 + tid
            while j_ptr + 63 * cuda.blockDim.x < row_end:
                for unroll_idx in range(64):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 64 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x

        cuda.syncthreads()
        k_ptr += 1


# ============================================================================
# Sync-Free Unrolling Kernels (5 variants: 4, 8, 16, 32, 64)
# ============================================================================

@cuda.jit
def ilu0_csr_sync_free_unrolling_kernel_4(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    if tid == 0:
        while cuda.atomic.add(dep_counts, i, 0) > 0:
            pass
    cuda.syncthreads()

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 3 * cuda.blockDim.x < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                    
                jcol = indices[j_ptr + cuda.blockDim.x]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr + cuda.blockDim.x] -= l_ik * data[pos]
                    
                jcol = indices[j_ptr + 2 * cuda.blockDim.x]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr + 2 * cuda.blockDim.x] -= l_ik * data[pos]
                    
                jcol = indices[j_ptr + 3 * cuda.blockDim.x]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr + 3 * cuda.blockDim.x] -= l_ik * data[pos]
                    
                j_ptr += 4 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x
        
        cuda.syncthreads()
        k_ptr += 1

    cuda.threadfence()
    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


@cuda.jit
def ilu0_csr_sync_free_unrolling_kernel_8(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    if tid == 0:
        while cuda.atomic.add(dep_counts, i, 0) > 0:
            pass
    cuda.syncthreads()

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 7 * cuda.blockDim.x < row_end:
                for unroll_idx in range(8):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 8 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x
        
        cuda.syncthreads()
        k_ptr += 1

    cuda.threadfence()
    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


@cuda.jit
def ilu0_csr_sync_free_unrolling_kernel_16(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    if tid == 0:
        while cuda.atomic.add(dep_counts, i, 0) > 0:
            pass
    cuda.syncthreads()

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 15 * cuda.blockDim.x < row_end:
                for unroll_idx in range(16):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 16 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x
        
        cuda.syncthreads()
        k_ptr += 1

    cuda.threadfence()
    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


@cuda.jit
def ilu0_csr_sync_free_unrolling_kernel_32(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    if tid == 0:
        while cuda.atomic.add(dep_counts, i, 0) > 0:
            pass
    cuda.syncthreads()

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 31 * cuda.blockDim.x < row_end:
                for unroll_idx in range(32):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 32 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x
        
        cuda.syncthreads()
        k_ptr += 1

    cuda.threadfence()
    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


@cuda.jit
def ilu0_csr_sync_free_unrolling_kernel_64(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    if tid == 0:
        while cuda.atomic.add(dep_counts, i, 0) > 0:
            pass
    cuda.syncthreads()

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            
            j_ptr = k_ptr + 1 + tid
            while j_ptr + 63 * cuda.blockDim.x < row_end:
                for unroll_idx in range(64):
                    jcol = indices[j_ptr + unroll_idx * cuda.blockDim.x]
                    pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                    if pos != -1:
                        data[j_ptr + unroll_idx * cuda.blockDim.x] -= l_ik * data[pos]
                j_ptr += 64 * cuda.blockDim.x
            
            while j_ptr < row_end:
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
                j_ptr += cuda.blockDim.x
        
        cuda.syncthreads()
        k_ptr += 1

    cuda.threadfence()
    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


# ============================================================================
# Launch Functions for Wavefront Unrolling
# ============================================================================

def ilu0_csr_numba_wavefront_unrolling_4(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_unrolling_kernel_4[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_wavefront_unrolling_8(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_unrolling_kernel_8[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_wavefront_unrolling_16(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_unrolling_kernel_16[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_wavefront_unrolling_32(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_unrolling_kernel_32[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_wavefront_unrolling_64(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_unrolling_kernel_64[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


# ============================================================================
# Launch Functions for Sync-Free Unrolling
# ============================================================================

def ilu0_csr_numba_sync_free_unrolling_4(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_unrolling_kernel_4[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_sync_free_unrolling_8(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_unrolling_kernel_8[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_sync_free_unrolling_16(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_unrolling_kernel_16[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_sync_free_unrolling_32(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_unrolling_kernel_32[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


def ilu0_csr_numba_sync_free_unrolling_64(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_unrolling_kernel_64[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()


# ============================================================================
# Mapping Kernels (Unroll Factor 1)
# ============================================================================

@cuda.jit
def ilu0_csr_wavefront_mapping_kernel(data, indices, indptr, rows, diag_pos):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if b >= rows.size:
        return

    row = rows[b]
    row_start = indptr[row]
    row_end   = indptr[row + 1]

    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= row:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]

            for j_ptr in range(k_ptr + 1 + tid, row_end, cuda.blockDim.x):
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]

        cuda.syncthreads()
        k_ptr += 1


def ilu0_csr_numba_wavefront_mapping(data, indices, indptr, wave_rows, diag_pos, threads_per_block=128):
    blocks = wave_rows.size
    ilu0_csr_wavefront_mapping_kernel[blocks, threads_per_block](
        data, indices, indptr, wave_rows, diag_pos
    )
    cuda.synchronize()


@cuda.jit
def ilu0_csr_sync_free_mapping_kernel(data, indices, indptr,
                                    dep_counts, succ_offsets, succ_indices,
                                    diag_pos):
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if i >= dep_counts.size:
        return

    if tid == 0:
        while cuda.atomic.add(dep_counts, i, 0) > 0:
            pass
    cuda.syncthreads()

    row_start = indptr[i]
    row_end   = indptr[i + 1]
    sh_l = cuda.shared.array(1, dtype=float64)

    k_ptr = row_start
    while k_ptr < row_end:
        k = indices[k_ptr]
        if k >= i:
            break

        diag_ptr = diag_pos[k]
        u_kk = data[diag_ptr]

        if u_kk != 0.0:
            if tid == 0:
                l_ik = data[k_ptr] / u_kk
                data[k_ptr] = l_ik
                sh_l[0] = l_ik
            cuda.syncthreads()
            l_ik = sh_l[0]

            k_row_start = indptr[k]
            k_row_end   = indptr[k + 1]
            for j_ptr in range(k_ptr + 1 + tid, row_end, cuda.blockDim.x):
                jcol = indices[j_ptr]
                pos = bin_search_col(indices, k_row_start, k_row_end, jcol)
                if pos != -1:
                    data[j_ptr] -= l_ik * data[pos]
        cuda.syncthreads()
        k_ptr += 1

    cuda.threadfence()
    s0 = succ_offsets[i]
    s1 = succ_offsets[i + 1]
    for p in range(s0 + tid, s1, cuda.blockDim.x):
        s = succ_indices[p]
        cuda.atomic.add(dep_counts, s, -1)


def ilu0_csr_numba_sync_free_mapping(data, indices, indptr,
                                   dep_counts, succ_offsets, succ_indices, diag_pos,
                                   threads_per_block=128):
    M = dep_counts.size
    blocks = M
    ilu0_csr_sync_free_mapping_kernel[blocks, threads_per_block](
        data, indices, indptr, dep_counts, succ_offsets, succ_indices, diag_pos
    )
    cuda.synchronize()

