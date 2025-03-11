from utils.dense_blas import lsolve_dense_col_order, dense_matvec_col_order


def sptrv_bcsc(L, x, b):
    sn_begin = L['sn_begin']
    sn_row_begin = L['sn_row_begin']
    Lp = L['p']
    Lx = L['x']
    Li = L['i']
    for i in range(len(sn_begin)):
        col_b=sn_begin[i]
        col_e=sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = sn_row_begin[i+1] - sn_row_begin[i]
        val_beg = Lx[Lp[col_b]]
        lsolve_dense_col_order(sup_len, sup_width, Lx[val_beg:], b[sn_row_begin[i]:sn_row_begin[i+1]])
        off_diag_val = Lx[val_beg+sup_len*sup_width:]
        sol = dense_matvec_col_order(off_diag_val, b[sn_row_begin[i]:sn_row_begin[i+1]])
        for j in range(off_diag_val, off_diag_val + sup_len):
            x[Li[j]] -= sol[j - off_diag_val]
            sol[j - off_diag_val] = 0
    return x

