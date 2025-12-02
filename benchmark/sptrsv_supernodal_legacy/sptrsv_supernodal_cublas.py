import cupy as cp
from cupy.cuda import Stream
from cupyx.scipy.linalg import solve_triangular
import numpy as np


class SupernodalSpTrsvCuBLAS:
    def __init__(self, Ls, wave_ptr, waves, max_streams=32):
        self.Ls = Ls
        self.wave_ptr = wave_ptr
        self.waves = waves
        self.max_streams = max_streams
        
        self.d_Lx = cp.asarray(Ls['x'])
        self.d_Li = cp.asarray(Ls['i'], dtype=cp.int32)
        self.d_x = cp.asarray(np.zeros(Ls['supernodes'][-1]))
        
        self.streams = [Stream() for _ in range(max_streams)]
    
    def solve(self, b):
        self.d_x[:] = cp.asarray(b)
        
        sn_begin = self.Ls['supernodes']
        Lip = self.Ls['ip']
        Lp = self.Ls['p']
        
        for level_idx in range(len(self.wave_ptr) - 1):
            level_start = self.wave_ptr[level_idx]
            level_end = self.wave_ptr[level_idx + 1]
            level_size = level_end - level_start
            
            for local_idx in range(level_size):
                sn_id = self.waves[level_start + local_idx]
                stream_id = local_idx % self.max_streams
                
                with self.streams[stream_id]:
                    col_b = sn_begin[sn_id]
                    col_e = sn_begin[sn_id + 1]
                    sup_width = col_e - col_b
                    sup_len = Lip[sn_id + 1] - Lip[sn_id]
                    val_beg = Lp[col_b]
                    
                    diag_block = self.d_Lx[val_beg:val_beg + sup_len * sup_width].reshape(
                        sup_len, sup_width, order='F')
                    x_block = self.d_x[col_b:col_e]
                    
                    solve_triangular(
                        diag_block[:sup_width, :sup_width], 
                        x_block, 
                        lower=True, 
                        overwrite_b=True
                    )
                    
                    if sup_len > sup_width:
                        off_diag = diag_block[sup_width:, :]
                        update = cp.dot(off_diag, x_block)
                        
                        row_indices = self.d_Li[Lip[sn_id] + sup_width:Lip[sn_id + 1]].astype(cp.int32)
                        self.d_x[row_indices] -= update
            
            for stream in self.streams:
                stream.synchronize()
        
        return self.d_x.get()
    
    def solve_device(self, d_b):
        return cp.asarray(self.solve(d_b.get() if hasattr(d_b, 'get') else d_b))


def sptrsv_csc_supernodal_cublas(cublas_solver, b):
    return cublas_solver.solve(b)

