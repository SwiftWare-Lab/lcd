import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as spio
from sksparse.cholmod import cholesky, analyze
from sympy import factor

from benchmark.cholesky import cholesky_left_looking

from utils.symbolic_cholesky_csc import etree, post_order, row_col_counts, \
    detect_supernodes, convert_csc_to_supernodal, pruning
from sptrsv_supernodal import sptrv_csc_supernodal


Relaxtion = 1
mtx_list = [
    #"/Users/kazem/UFDB/LFAT5/LFAT5.mtx",
    "/home/kazem/Downloads/LFAT5/LFAT5.mtx",
    "/home/kazem/Downloads/ex5/ex5.mtx",
    "/home/kazem/Downloads/mesh1e1/mesh1e1.mtx","/home/kazem/Downloads/nos4/nos4.mtx",
    "/home/kazem/Downloads/bcsstk03/bcsstk03.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx",
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx"
            ]

for mtx in mtx_list:
    Ao = spio.mmread(mtx)
    Ao_csc = Ao.tocsc()
    # lower part
    Au = sp.triu(Ao).tocsc()
    A = sp.tril(Ao).tocsc()

    factor = cholesky(Ao_csc)
    L = factor.L()
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
    # reorder A
    A_p = A[P, :][:, P].tocsc(copy=True).sorted_indices()
    #Au_p = Au[P, :][:, P]
    Ao_p = Ao_csc[P, :][:, P].tocsc(copy=True).sorted_indices()

    L_copy = L.copy()
    L_copy.data[:] = 0
    L_copy = L_copy.tocsc()

    prunePtr_r, pruneSet_r= np.zeros(L.shape[0]+1, dtype=np.int32), np.zeros(L.nnz, dtype=np.int32)
    parent, common = np.zeros(L.shape[0], dtype=np.int32), {}
    etree(L, parent, common)
    #pruning(L.shape[0], Ao_p.indptr, Ao_p.indices, parent, prunePtr_r, pruneSet_r)
    Lt = L.T.tocsc()
    prunePtr, pruneSet = Lt.indptr, Lt.indices

    cholesky_left_looking(L_copy.shape[0], Ao_p.indptr, Ao_p.indices, Ao_p.data,
                          L_copy.indptr, L_copy.indices, L_copy.data, prunePtr, pruneSet)

#
    print("diff L: ", np.linalg.norm(L.data-L_copy.data, np.inf))


    #print("norm: ", np.linalg.norm(b-Ao.dot(x)))


    print("Matrix: ", mtx)