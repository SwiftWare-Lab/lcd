
import numpy as np
from numba import njit, prange, float32, int32, float64, objmode, void



# @njit(float64[:](int32[:], int32[:], int32[:], int32[:], float64[:],
#                  float64[:]), parallel=True, fastmath=True)
def sptrsv_csc_supernodal_analysis_1(sn_begin, Lip, Li, x):
    flag = np.zeros(len(x), dtype=np.int32)
    for i in range(len(sn_begin ) -1):
        col_b = sn_begin[i]
        col_e = sn_begin[ i + 1]
        sup_width = col_e - col_b
        #flag[col_b:col_e] += 1
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            flag[ Li[j] ] += 1

    print("sum flag: ", np.sum(flag))
    thread_flag = np.zeros(len(sn_begin ) -1, dtype=np.int32)
    for i in range(len(sn_begin ) -1):
        col_b = sn_begin[i]
        col_e = sn_begin[ i +1]
        for kk in range(col_b, col_e):
            thread_flag[i] +=  flag[kk]
    return thread_flag


def sptrsv_csc_supernodal_analysis(sn_begin, Lip, Li, col_to_sup):
    thread_flag = np.zeros(len(sn_begin ) -1, dtype=np.int32)
    for i in range(len(sn_begin ) -1):
        col_b = sn_begin[i]
        col_e = sn_begin[ i + 1]
        sup_width = col_e - col_b
        # flag[col_b:col_e] += 1
        thread_flag[i] += 1
        for j in range(Lip[i] + sup_width, Lip[i + 1]):
            thread_flag[ col_to_sup[Li[j]] ] += 1
    return thread_flag