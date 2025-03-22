import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as spio
from sksparse.cholmod import cholesky, analyze
from sympy import factor

from benchmark.cholesky import cholesky_left_looking, \
    cholesky_left_looking_coupled, cholesky_left_looking_numba, \
    cholesky_left_looking_numba_parallel

from utils.symbolic_cholesky_csc import etree, post_order, row_col_counts, \
    detect_supernodes, convert_csc_to_supernodal, pruning, ereach, \
    cholesky_left_looking_numba_parallel_inspector
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
    print("norm cholesky: ", np.linalg.norm(b - Ao.dot(x), ord=np.inf))
    print("norm cholesky csc: ", np.linalg.norm(b - Ao_csc.dot(x), ord=np.inf))
    #L = factor.L()
    #factor = analyze(A)
    #factor.cholesky(A)
    #print("nz A: ", A.nnz, " nz L: ", L.nnz)
    # reorder A
    A_p = A[P, :][:, P].tocsc(copy=True).sorted_indices()
    Au_p = Au[P, :][:, P].tocsc(copy=True).sorted_indices()
    #Ao_p = Ao_csc[P, :][:, P].tocsc(copy=True).sorted_indices()
    Ao_p = Ao_csc[P[:, np.newaxis], P[np.newaxis, :]].tocsc(copy=True).sorted_indices()
    #Ao_pt = Ao_p.T.tocsc(copy=True).sorted_indices()

    L_copy = L.copy()
    L_copy.data[:] = 0
    L_copy = L_copy.tocsc()

    prunePtr_r, pruneSet_r= np.zeros(L.shape[0]+1, dtype=np.int32), np.zeros(L.nnz, dtype=np.int32)
    parent, common = np.zeros(L.shape[0], dtype=np.int32), {}
    etree(Au_p, parent, common)
    #pruning(L.shape[0], Ao_p.indptr, Ao_p.indices, parent, prunePtr_r, pruneSet_r)
    Lt = L.T.tocsc()
    n=L.shape[0]
    # for i in range(n):
    #     if i == 5:
    #         print("dd")
    #     xi = np.zeros(2 * n, dtype=int)
    #     top = ereach(n, Ao_p.indptr, Ao_p.indices, i, parent, xi[0:n], xi[n:])
    #     if len(xi[top:n]) + 1 !=  len(Lt.indices [Lt.indptr[i]:Lt.indptr[i+1]] ):
    #         print(i, " : ", len( xi[top:n] ) + 1, " : ", len(Lt.indices [Lt.indptr[i]:Lt.indptr[i+1]] ))

    prunePtr, pruneSet = Lt.indptr, Lt.indices

    cholesky_left_looking(L_copy.shape[0], Ao_p.indptr, Ao_p.indices, Ao_p.data,
                          L_copy.indptr, L_copy.indices, L_copy.data, prunePtr, pruneSet)

#
    print("diff L: ", np.linalg.norm(L.data-L_copy.data, np.inf))

    L_copy2 = L.copy()
    L_copy2.data[:] = 0
    L_copy2 = L_copy2.tocsc()

    cholesky_left_looking_numba(L_copy2.shape[0], Ao_p.indptr, Ao_p.indices, Ao_p.data,
                                L_copy2.indptr, L_copy2.indices, L_copy2.data, prunePtr, pruneSet)

    print("diff L numba: ", np.linalg.norm(L.data-L_copy2.data, np.inf))

    thread_flags = np.zeros(L.shape[0], dtype=np.int32)
    cholesky_left_looking_numba_parallel_inspector(L_copy2.shape[0], Ao_p.indptr, Ao_p.indices,
                                L_copy2.indptr, L_copy2.indices, prunePtr, pruneSet, thread_flags)

    ret_val = True
    L_copy2.data[:] = 0
    cholesky_left_looking_numba_parallel(L_copy2.shape[0], Ao_p.indptr, Ao_p.indices, Ao_p.data,
                          L_copy2.indptr, L_copy2.indices, L_copy2.data, prunePtr, pruneSet, thread_flags, ret_val)

    print("diff L numba parallel: ", np.linalg.norm(L.data-L_copy2.data, np.inf))



    #print("norm: ", np.linalg.norm(b-Ao.dot(x)))


    print("Matrix: ", mtx)