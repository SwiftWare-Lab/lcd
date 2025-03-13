import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as spio
from sksparse.cholmod import cholesky, analyze
from sympy import factor

#from sksparse.cholmod import cholesky, analyze, Factor

from utils.symbolic_cholesky_csc import etree, post_order, row_col_counts, \
    detect_supernodes, convert_csc_to_supernodal
from sptrsv_supernodal import sptrv_csc_supernodal

mtx_list = [
    #"/Users/kazem/UFDB/LFAT5/LFAT5.mtx",
    #"/home/kazem/Downloads/LFAT5/LFAT5.mtx",
    #"/home/kazem/Downloads/ex5/ex5.mtx",
    "/home/kazem/Downloads/mesh1e1/mesh1e1.mtx","/home/kazem/Downloads/nos4/nos4.mtx",
    "/home/kazem/Downloads/bcsstk03/bcsstk03.mtx",
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx"
            ]

for mtx in mtx_list:
    Ao = spio.mmread(mtx)
    Ao_csc = Ao.tocsc()
    # lower part
    Au = sp.triu(Ao).tocsc()
    A = sp.tril(Ao).tocsc()

    factor = cholesky(Ao)
    L = factor.LD()
    P = factor.P()
    b = L.dot(np.ones(L.shape[0]))  #np.random.rand(Ao.shape[0])
    # Solve the linear system Ax = b
    #x = factor(b)
    x = factor.solve_A(b)
    print("nnz L: ", L.nnz)
    #L = factor.L()
    #factor = analyze(A)
    #factor.cholesky(A)
    #print("nz A: ", A.nnz, " nz L: ", L.nnz)

    #x = factor.solve_A(b)
    print("norm: ", np.linalg.norm(b-Ao.dot(x)))
    # permute the matrix Ao
    Aperm = Ao_csc[P, :][:, P]

    parent = np.array(A.shape[0] * [0], dtype=np.int32)
    common = {}

    Ls = convert_csc_to_supernodal(L)

    x_sup_csc = sptrv_csc_supernodal(L['supernodes'],
                                     L['indices_ptr'],
                                     L['indptr'],
                                     L['data'],
                                     L['indices'], b)

    #x_csc = L.solve(b)

    print("norm: ", np.linalg.norm(x_sup_csc-np.ones(L.shape[0])))

    #plt.spy(Ls, markersize=1)
    #plt.show()

    # Compute the etree
    # Aper_u = sp.triu(Aperm).tocsc()
    # etree_A = etree(Aper_u, parent, common)
    #
    # parent2 = np.array(A.shape[0] * [0], dtype=np.int32)
    # etree_2 = etree(L, parent2, common)
    # snodes = detect_supernodes(L, 2)
    # print("snodes: ", snodes)
    #plt.spy(L, markersize=1)
    #plt.show()

    # make sure parent and parent2 are the same
    #print("diffs", np.sum(parent != parent2))

    # parent = np.array(A.shape[0]*[0], dtype=np.int32)
    # common = {}
    # post_order_array = np.array(A.shape[0]*[0], dtype=np.int32)
    # row_count, col_count = np.array(A.shape[0]*[0], dtype=np.int32), np.array(A.shape[0]*[0], dtype=np.int32)
    #
    # e_ret = etree(A, parent, common)
    # print("ETREE: ", parent+1)
    # p_o_ret = post_order(parent, A.shape[0], None, post_order_array, common)
    # print("P_ORDER: ", post_order_array+1)
    # rc_ret = row_col_counts(A, None, None, parent, post_order_array, row_count, col_count, common)

    # etree_A = etree(Au)
    # print("ETREE: ", etree_A+1)
    # p_order = post_order(etree_A)
    # print("P_ORDER: ", p_order+1)
    # A_post = A[p_order[:,None], p_order[None, :]].tocsc()
    # col_count_A = column_count(A, etree_A, p_order)
    # print("COL_COUNT: ", col_count_A)
    # row_pattern_A = row_pattern(A, etree_A)

    print("Matrix: ", mtx)