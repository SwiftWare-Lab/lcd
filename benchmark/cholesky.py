
import scipy.sparse as sp
import scipy.io as spio
import scipy.sparse.linalg as spla

from sksparse.cholmod import cholesky, analyze, Factor

from utils.symbolic_cholesky_csc import etree, column_count, row_pattern, \
    compute_supernodes, post_order

mtx_list = [
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
    etree_A = etree(Au)
    p_order = post_order(etree_A)
    col_count_A = column_count(A, etree_A, p_order)
    row_pattern_A = row_pattern(A, etree_A)

    print(row_pattern_A)
    print("Matrix: ", mtx)