import threading

from cupyx.jit import atomic_add
import cupyx.scipy.linalg

from utils.dense_blas import *
from numba import njit, cuda, config
from numba import prange, float32, int32, float64, objmode, void
import numpy as np
import cupy as cp

config.CUDA_ENABLE_PYNVJITLINK = 1

scatter_kernel = cp.RawKernel(r'''
extern "C" __global__
void scatter_kernel(const double* update_vals, const int* row_indices, 
                   double* x, int n_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_rows) {
        int idx = row_indices[i];
        double val = update_vals[i];
        atomicAdd(&x[idx], -val);
    }
}
''', 'scatter_kernel')

def sptrv_csc_supernodal(sn_begin, Lip, Li, Lp, Lx, b, x):
    for i in range(len(sn_begin)-1):
        col_b = sn_begin[i]
        col_e = sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = Lip[i+1] - Lip[i]
        val_beg = Lp[col_b]
        
        lsolve_dense_col_order(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        
        if sup_len > sup_width:
            off_diag_val = Lx[val_beg + sup_width:]
            sol = dense_matvec_col_order(sup_len, sup_len - sup_width, sup_width,
                                         off_diag_val, x[col_b:col_e])
            cnt = 0
            for j in range(Lip[i]+sup_width, Lip[i+1]):
                x[Li[j]] -= sol[cnt]
                sol[cnt] = 0
                cnt += 1


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
    M = len(sn_begin) - 1
    if i < M:
        while cuda.atomic.add(flag_thread, i, 0) > 0:
            pass
        
        sol = cuda.shared.array(shape=(1000,), dtype=float64)
        col_b = sn_begin[i]
        col_e = sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = Lip[i+1] - Lip[i]
        val_beg = Lp[col_b]
        
        lsolve_BLAS_python(sup_len, sup_width, Lx[val_beg:], x[col_b:col_e])
        off_diag_val = Lx[val_beg + sup_width:]
        matvec_BLAS_python(sup_len, sup_len - sup_width,
                           sup_width,
                           off_diag_val, x[col_b:col_e], sol)
        cnt = 0
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
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



def sptrsv_csc_supernodal_wavefront_cuda_graph(sn_begin, Lip, Li, Lp, Lx, b, x, wave_ptr, waves, reps):
    graphs = []
    cp_stream = cp.cuda.Stream()
    with cp_stream:
        for w0 in range(len(wave_ptr)- 1):
            cp_stream.begin_capture()
            w_start = int(wave_ptr[w0])
            w_end = int(wave_ptr[w0+1])
            for w1 in range(w_start, w_end):
                sn_id = int(waves[w1])
                col_b = int(sn_begin[sn_id])
                col_e = int(sn_begin[sn_id + 1])
                sup_width = col_e - col_b
                sup_len = int(Lip[sn_id + 1] - Lip[sn_id])
                val_beg = int(Lp[col_b])
                row_idx_start = int(Lip[sn_id]) + sup_width
                row_idx_end = int(Lip[sn_id + 1])
                n_scatter = row_idx_end - row_idx_start
                diag_block = Lx[val_beg:val_beg + sup_len * sup_width].reshape(
                    sup_len, sup_width, order="F"
                )
                x_block = x[col_b:col_e]

                L_diag = diag_block[:sup_width, :sup_width]
                cupyx.scipy.linalg.solve_triangular(L_diag, x_block, lower=True, overwrite_b=True)

                if sup_len > sup_width and n_scatter > 0:
                    off_diag = diag_block[sup_width:, :]
                    update = cp.dot(off_diag, x_block)

                    row_idx = Li[row_idx_start:row_idx_end]
                    threads_per_block = 256
                    blocks = (n_scatter + threads_per_block - 1) // threads_per_block
                    scatter_kernel(
                        (blocks,), (threads_per_block,),
                        (update, row_idx, x, cp.int32(n_scatter))
                    )
            g = cp_stream.end_capture()
            graphs.append(g)
        execs = [g.instantiate() for g in graphs]
        for gexec in execs:
            gexec.launch(cp_stream)
            cp_stream.synchronize()


