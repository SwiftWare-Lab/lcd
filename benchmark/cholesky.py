import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import scipy.sparse.linalg as spla

#from sksparse.cholmod import cholesky, analyze, Factor

from utils.symbolic_cholesky_csc import etree, post_order, row_col_counts

mtx_list = [
    "/Users/kazem/UFDB/LFAT5/LFAT5.mtx",
    "/home/kazem/Downloads/LFAT5/LFAT5.mtx", "/home/kazem/Downloads/ex5/ex5.mtx", "/home/kazem/Downloads/mesh1e1/mesh1e1.mtx","/home/kazem/Downloads/nos4/nos4.mtx",
    # "/home/kazem/Downloads/bcsstk03/bcsstk03.mtx",
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx"
            ]

for mtx in mtx_list:
    A = spio.mmread(mtx)
    # lower part
    Au = sp.triu(A).tocsc()
    A = sp.tril(A).tocsc()
    parent = np.array(A.shape[0]*[0], dtype=np.int32)
    common = {}
    post_order_array = np.array(A.shape[0]*[0], dtype=np.int32)
    row_count, col_count = np.array(A.shape[0]*[0], dtype=np.int32), np.array(A.shape[0]*[0], dtype=np.int32)

    e_ret = etree(A, parent, common)
    print("ETREE: ", parent+1)
    p_o_ret = post_order(parent, A.shape[0], None, post_order_array, common)
    print("P_ORDER: ", post_order_array+1)
    rc_ret = row_col_counts(A, None, None, parent, post_order_array, row_count, col_count, common)

    # etree_A = etree(Au)
    # print("ETREE: ", etree_A+1)
    # p_order = post_order(etree_A)
    # print("P_ORDER: ", p_order+1)
    # A_post = A[p_order[:,None], p_order[None, :]].tocsc()
    # col_count_A = column_count(A, etree_A, p_order)
    # print("COL_COUNT: ", col_count_A)
    # row_pattern_A = row_pattern(A, etree_A)

    print("Matrix: ", mtx)