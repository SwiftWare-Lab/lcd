from cupyx.jit import atomic_add

from utils.dense_blas import lsolve_dense_col_order, dense_matvec_col_order, \
    lsolve_dense_col_order_numba, dense_matvec_col_order_numba
from numba import njit
from numba import prange, float32, int32, float64


def sptrv_csc_supernodal(sn_begin, Lip, Li, Lp, Lx, b):
    x = b.copy()
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
    return x


def sptrsv_csc_supernodal_wavefront(sn_begin, Lip, Li, Lp, Lx, b,wave_ptr, waves):
    x = b.copy()
    #for w in range(len(waves)):
    #    for i in waves[w]:
    for w0 in range(len(wave_ptr)-1):
        for w1 in range(wave_ptr[w0], wave_ptr[w0+1]):
            i = waves[w1]
            col_b = sn_begin[i]
            col_e = sn_begin[i + 1]
            sup_width = col_e - col_b
            sup_len = Lip[i + 1] - Lip[i]
            val_beg = Lp[col_b]
            lsolve_dense_col_order(sup_len, sup_width, Lx[val_beg:],
                                   x[col_b:col_e])
            off_diag_val = Lx[val_beg + sup_width:]
            sol = dense_matvec_col_order(sup_len, sup_len - sup_width,
                                         sup_width,
                                         off_diag_val, x[col_b:col_e])
            cnt = 0
            for j in range(Lip[i] + sup_width, Lip[i + 1]):
                x[Li[j]] -= sol[cnt]
                sol[cnt] = 0
                cnt += 1
    return x


@njit(float64[:](int32[:], int32[:], int32[:], int32[:], float64[:],
                 float64[:], int32[:], int32[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_wavefront_numba(sn_begin, Lip, Li, Lp, Lx, b,wave_ptr, waves):
    x = b.copy()
    for w0 in range(len(wave_ptr)-1):
        for w1 in prange(wave_ptr[w0], wave_ptr[w0+1]):
            i = waves[w1]
            col_b = sn_begin[i]
            col_e = sn_begin[i + 1]
            sup_width = col_e - col_b
            sup_len = Lip[i + 1] - Lip[i]
            val_beg = Lp[col_b]
            lsolve_dense_col_order_numba(sup_len, sup_width, Lx[val_beg:],
                                   x[col_b:col_e])
            off_diag_val = Lx[val_beg + sup_width:]
            sol = dense_matvec_col_order_numba(sup_len, sup_len - sup_width,
                                         sup_width,
                                         off_diag_val, x[col_b:col_e])
            cnt = 0
            for j in range(Lip[i] + sup_width, Lip[i + 1]):
                x[Li[j]] -= sol[cnt]
                #atomic_add(x[Li[j]], -sol[cnt])
                sol[cnt] = 0
                cnt += 1
    return x





