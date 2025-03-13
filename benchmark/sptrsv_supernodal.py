from utils.dense_blas import lsolve_dense_col_order, dense_matvec_col_order


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

