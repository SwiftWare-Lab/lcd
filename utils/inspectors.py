

@njit(float64[:](int32[:], int32[:], int32[:], int32[:], float64[:],
                 float64[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_numba(sn_begin, Lip, Li, Lp, Lx, b):
    for i in range(len(sn_begin ) -1):
        col_b = sn_begin[i]
        col_e = sn_begin[ i +1]
        sup_width = col_e - col_b
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            flag[ Li[j] ] += 1;

    thread_flag = np.zero(len(sn_begin ) -1)
    for i in range(len(sn_begin ) -1):
        col_b = sn_begin[i]
        col_e = sn_begin[ i +1]
        for kk in range(col_b, col_e):
            thread_flag[i] += flag[kk]
