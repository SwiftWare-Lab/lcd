from utils.dense_blas import lsolve_dense_col_order, dense_matvec_col_order


def sptrv_csc_supernodal(L, b):
    x = b.copy()
    sn_begin = L['supernodes']
    Lip = L['indices_ptr']
    Lp = L['indptr']
    Lx = L['data']
    Li = L['indices']
    for i in range(len(sn_begin)):
        col_b = sn_begin[i]
        col_e = sn_begin[i+1]
        sup_width = col_e - col_b
        sup_len = Lip[i+1] - Lip[i]
        val_beg = Lp[col_b]
        lsolve_dense_col_order(sup_len, sup_width, Lx[val_beg:], b[col_b:col_e])
        off_diag_val = Lx[val_beg+sup_width:]
        sol = dense_matvec_col_order(sup_len, sup_len-sup_width, sup_width,
                                     off_diag_val, b[Lip[i]:Lip[i+1]])
        cnt=0
        for j in range(Lip[col_b]+sup_width, Lip[col_e]):
            x[Li[j]] -= sol[cnt]
            sol[cnt] = 0
            cnt += 1
    return x

