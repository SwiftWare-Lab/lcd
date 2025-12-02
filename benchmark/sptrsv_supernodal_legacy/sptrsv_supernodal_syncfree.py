import numpy as np
from numba import cuda, float64, int32


@cuda.jit
def persistent_supernodal_kernel(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, 
                                 dep_count, n_supernodes):
    sn_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    
    if sn_id >= n_supernodes:
        return
    
    if tid == 0:
        while cuda.atomic.add(dep_count, sn_id, 0) > 0:
            pass
    cuda.syncthreads()
    
    col_b = sn_begin[sn_id]
    col_e = sn_begin[sn_id + 1]
    sup_width = col_e - col_b
    sup_len = Lip[sn_id + 1] - Lip[sn_id]
    val_beg = Lp[col_b]
    
    sol = cuda.shared.array(shape=(1000,), dtype=float64)
    
    if tid == 0:
        for col in range(sup_width):
            diag_val = Lx[val_beg + col * sup_len + col]
            if abs(diag_val) > 1e-15:
                x[col_b + col] /= diag_val
            val = x[col_b + col]
            for row in range(col + 1, sup_width):
                x[col_b + row] -= Lx[val_beg + col * sup_len + row] * val
    cuda.syncthreads()
    
    if sup_len > sup_width:
        off_diag_start = val_beg + sup_width
        
        if tid == 0:
            for row in range(sup_len - sup_width):
                temp = 0.0
                for col in range(sup_width):
                    temp += Lx[off_diag_start + col * sup_len + row] * x[col_b + col]
                sol[row] = temp
        cuda.syncthreads()
        
        block_size = cuda.blockDim.x
        for j in range(Lip[sn_id] + sup_width + tid, Lip[sn_id + 1], block_size):
            idx = Li[j]
            local_idx = j - (Lip[sn_id] + sup_width)
            cuda.atomic.add(x, idx, -sol[local_idx])
            cuda.atomic.add(dep_count, col_to_sup[idx], -1)


@cuda.jit
def block_level_supernodal_kernel(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x, 
                                  dep_count, n_supernodes):
    sn_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    
    if sn_id >= n_supernodes:
        return
    
    if tid == 0:
        while cuda.atomic.add(dep_count, sn_id, 0) > 0:
            pass
    cuda.syncthreads()
    
    col_b = sn_begin[sn_id]
    col_e = sn_begin[sn_id + 1]
    sup_width = col_e - col_b
    sup_len = Lip[sn_id + 1] - Lip[sn_id]
    val_beg = Lp[col_b]
    
    sol = cuda.shared.array(shape=(1000,), dtype=float64)
    
    for col in range(sup_width):
        if tid == 0:
            diag_val = Lx[val_beg + col * sup_len + col]
            if abs(diag_val) > 1e-15:
                x[col_b + col] /= diag_val
        cuda.syncthreads()
        
        val = x[col_b + col]
        block_size = cuda.blockDim.x
        for row in range(col + 1 + tid, sup_width, block_size):
            x[col_b + row] -= Lx[val_beg + col * sup_len + row] * val
        cuda.syncthreads()
    
    if sup_len > sup_width:
        off_diag_start = val_beg + sup_width
        
        block_size = cuda.blockDim.x
        for row in range(tid, sup_len - sup_width, block_size):
            temp = 0.0
            for col in range(sup_width):
                temp += Lx[off_diag_start + col * sup_len + row] * x[col_b + col]
            sol[row] = temp
        cuda.syncthreads()
        
        for j in range(Lip[sn_id] + sup_width + tid, Lip[sn_id + 1], block_size):
            idx = Li[j]
            local_idx = j - (Lip[sn_id] + sup_width)
            cuda.atomic.add(x, idx, -sol[local_idx])
            cuda.atomic.add(dep_count, col_to_sup[idx], -1)


@cuda.jit
def hybrid_supernodal_kernel(sn_begin, col_to_sup, Lip, Li, Lp, Lx, x,
                             dep_count, n_supernodes, size_threshold):
    sn_id = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    
    if sn_id >= n_supernodes:
        return
    
    if tid == 0:
        while cuda.atomic.add(dep_count, sn_id, 0) > 0:
            pass
    cuda.syncthreads()
    
    col_b = sn_begin[sn_id]
    col_e = sn_begin[sn_id + 1]
    sup_width = col_e - col_b
    sup_len = Lip[sn_id + 1] - Lip[sn_id]
    val_beg = Lp[col_b]
    
    sol = cuda.shared.array(shape=(1000,), dtype=float64)
    
    if sup_width <= size_threshold:
        if tid == 0:
            for col in range(sup_width):
                diag_val = Lx[val_beg + col * sup_len + col]
                if abs(diag_val) > 1e-15:
                    x[col_b + col] /= diag_val
                val = x[col_b + col]
                for row in range(col + 1, sup_width):
                    x[col_b + row] -= Lx[val_beg + col * sup_len + row] * val
        cuda.syncthreads()
    else:
        for col in range(sup_width):
            if tid == 0:
                diag_val = Lx[val_beg + col * sup_len + col]
                if abs(diag_val) > 1e-15:
                    x[col_b + col] /= diag_val
            cuda.syncthreads()
            
            val = x[col_b + col]
            block_size = cuda.blockDim.x
            for row in range(col + 1 + tid, sup_width, block_size):
                x[col_b + row] -= Lx[val_beg + col * sup_len + row] * val
            cuda.syncthreads()
    
    if sup_len > sup_width:
        off_diag_start = val_beg + sup_width
        
        block_size = cuda.blockDim.x
        for row in range(tid, sup_len - sup_width, block_size):
            temp = 0.0
            for col in range(sup_width):
                temp += Lx[off_diag_start + col * sup_len + row] * x[col_b + col]
            sol[row] = temp
        cuda.syncthreads()
        
        for j in range(Lip[sn_id] + sup_width + tid, Lip[sn_id + 1], block_size):
            idx = Li[j]
            local_idx = j - (Lip[sn_id] + sup_width)
            cuda.atomic.add(x, idx, -sol[local_idx])
            cuda.atomic.add(dep_count, col_to_sup[idx], -1)


def sptrsv_supernodal_persistent(Ls, b, dep_count_cpu):
    d_x = cuda.to_device(b)
    d_Lx = cuda.to_device(Ls['x'])
    d_Li = cuda.to_device(Ls['i'])
    d_Lp = cuda.to_device(Ls['p'])
    d_sn_begin = cuda.to_device(Ls['supernodes'])
    d_Lip = cuda.to_device(Ls['ip'])
    d_col_to_sup = cuda.to_device(Ls['col_to_super'])
    d_dep_count = cuda.to_device(dep_count_cpu)
    
    n_supernodes = len(Ls['supernodes']) - 1
    threads_per_block = 256
    blocks = n_supernodes
    
    persistent_supernodal_kernel[blocks, threads_per_block](
        d_sn_begin, d_col_to_sup, d_Lip, d_Li, d_Lp, d_Lx, d_x,
        d_dep_count, n_supernodes)
    
    return d_x.copy_to_host()


def sptrsv_supernodal_block_level(Ls, b, dep_count_cpu):
    d_x = cuda.to_device(b)
    d_Lx = cuda.to_device(Ls['x'])
    d_Li = cuda.to_device(Ls['i'])
    d_Lp = cuda.to_device(Ls['p'])
    d_sn_begin = cuda.to_device(Ls['supernodes'])
    d_Lip = cuda.to_device(Ls['ip'])
    d_col_to_sup = cuda.to_device(Ls['col_to_super'])
    d_dep_count = cuda.to_device(dep_count_cpu)
    
    n_supernodes = len(Ls['supernodes']) - 1
    threads_per_block = 256
    blocks = n_supernodes
    
    block_level_supernodal_kernel[blocks, threads_per_block](
        d_sn_begin, d_col_to_sup, d_Lip, d_Li, d_Lp, d_Lx, d_x,
        d_dep_count, n_supernodes)
    
    return d_x.copy_to_host()


def sptrsv_supernodal_hybrid(Ls, b, dep_count_cpu, size_threshold=32):
    d_x = cuda.to_device(b)
    d_Lx = cuda.to_device(Ls['x'])
    d_Li = cuda.to_device(Ls['i'])
    d_Lp = cuda.to_device(Ls['p'])
    d_sn_begin = cuda.to_device(Ls['supernodes'])
    d_Lip = cuda.to_device(Ls['ip'])
    d_col_to_sup = cuda.to_device(Ls['col_to_super'])
    d_dep_count = cuda.to_device(dep_count_cpu)
    
    n_supernodes = len(Ls['supernodes']) - 1
    threads_per_block = 256
    blocks = n_supernodes
    
    hybrid_supernodal_kernel[blocks, threads_per_block](
        d_sn_begin, d_col_to_sup, d_Lip, d_Li, d_Lp, d_Lx, d_x,
        d_dep_count, n_supernodes, size_threshold)
    
    return d_x.copy_to_host()
