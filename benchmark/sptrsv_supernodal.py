import threading

from cupyx.jit import atomic_add

from utils.dense_blas import *
from numba import njit, cuda, config
from numba import prange, float32, int32, float64, objmode, void
import numpy as np

config.CUDA_ENABLE_PYNVJITLINK = 1

def sptrv_csc_supernodal(sn_begin, Lip, Li, Lp, Lx, b, x):
    #x = b.copy()
    for i in range(len(sn_begin)-1):
        col_b = sn_begin[i]
        col_e = sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = Lip[i+1] - Lip[i]
        val_beg = Lp[col_b]
        lsolve_dense_col_order(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        off_diag_val = Lx[val_beg+sup_width:]
        sol = dense_matvec_col_order(sup_len, sup_len-sup_width, sup_width,
                                     off_diag_val, x[col_b:col_e])
        cnt=0
        for j in range(Lip[i]+sup_width, Lip[i+1]):
            x[Li[j]] -= sol[cnt]
            sol[cnt] = 0
            cnt += 1
    #return x


@njit(void(int32[:], int32[:], int32[:], int32[:], float64[:],
                 float64[:], float64[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_numba(sn_begin, Lip, Li, Lp, Lx, b, x):
    sol = np.zeros(len(b), dtype=b.dtype)
    for i in range(len(sn_begin)-1):
        col_b = sn_begin[i]
        col_e = sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = Lip[i+1] - Lip[i]
        val_beg = Lp[col_b]
        lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        off_diag_val = Lx[val_beg+sup_width:]
        matvec_BLAS_python(sup_len, sup_len - sup_width,
                           sup_width,
                           off_diag_val, x[col_b:col_e], sol)
        cnt = 0
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            x[Li[j]] -= sol[cnt]
            # atomic_add(x[Li[j]], -sol[cnt])
            sol[cnt] = 0
            cnt += 1


@cuda.jit
def sptrsv_csc_supernodal_numba_cuda_kernel(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, tmp_mem, max_width, flag_thread):
    i = cuda.grid(1)
    #sol = tmp_mem[i * max_width: (i + 1) * max_width]
    M = len(sn_begin) - 1
    if i < M:
        #print("flag[i", flag_thread[i])
        while cuda.atomic.add(flag_thread, i, 0) > 0:
            pass
    #for i in range(len(sn_begin)  - 1):
        #print("flag[i] starts", flag_thread[i])
        sol = cuda.shared.array(shape=(1000,), dtype=float64)
        col_b = sn_begin[i]
        col_e = sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = Lip[i+1] - Lip[i]
        val_beg = Lp[col_b]
        lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        off_diag_val = Lx[val_beg+sup_width:]
        matvec_BLAS_python(sup_len, sup_len - sup_width,
                           sup_width,
                           off_diag_val, x[col_b:col_e], sol)
        cnt = 0
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            #x[Li[j]] -= sol[cnt]
            tmp = -sol[cnt]
            idx = Li[j]
            cuda.atomic.add(x, idx, tmp)
            idxs = col_to_sup[idx]
            cuda.atomic.add(flag_thread, idxs, -1)
            sol[cnt] = 0
            cnt += 1


def sptrsv_csc_supernodal_numba_cuda(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, flag_thread, max_width):
    #tmp_mem = cuda.to_device(np.zeros(len(sn_begin)*max_width, dtype=np.float64))
    tmp_mem = 1
    threads_per_block = 1
    blocks_per_grid = (len(sn_begin) + threads_per_block - 1) // threads_per_block
    sptrsv_csc_supernodal_numba_cuda_kernel[blocks_per_grid, threads_per_block](sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, tmp_mem, max_width, flag_thread)
    #return x



@njit(void(int32[:], int32[:], int32[:], int32[:], float64[:],
                 float64[:], float64[:], int32[:], int32[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_wavefront_numba(sn_begin, Lip, Li, Lp, Lx, b, x, wave_ptr, waves):
    sol = np.zeros(len(b), dtype=b.dtype)
    # with objmode:
    #     lock = threading.Lock()
    for w0 in range(len(wave_ptr)-1):
        for w1 in prange(wave_ptr[w0], wave_ptr[w0+1]):
            i = waves[w1]
            col_b = sn_begin[i]
            col_e = sn_begin[i + 1]
            sup_width = col_e - col_b
            sup_len = Lip[i + 1] - Lip[i]
            val_beg = Lp[col_b]
            lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:],
                                   x[col_b:col_e])
            off_diag_val = Lx[val_beg + sup_width:]
            matvec_BLAS_python(sup_len, sup_len - sup_width,
                                         sup_width,
                                         off_diag_val, x[col_b:col_e], sol)
            cnt = 0
            for j in range(Lip[i] + sup_width, Lip[i + 1]):
                x[Li[j]] -= sol[cnt]
                #atomic_add(x[Li[j]], -sol[cnt])
                sol[cnt] = 0
                cnt += 1






