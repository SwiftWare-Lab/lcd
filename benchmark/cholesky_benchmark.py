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

    x_sup_csc = sptrv_csc_supernodal(Ls['supernodes'],
                                     Ls['ip'],
                                     Ls['p'],
                                     Ls['x'],
                                     Ls['i'], b)

    #x_csc = L.solve(b)

    print("norm: ", np.linalg.norm(x_sup_csc-np.ones(L.shape[0])))


    print("Matrix: ", mtx)