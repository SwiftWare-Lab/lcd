from bench_utils import *
from sptrsv import sptrsv_csr_numba
import scipy.io as sio

mtx_list = get_matrix_list("/home/salehm32/projects/fused-gnn/fusion/data/spd_list.txt", "/home/salehm32/projects/fused-gnn/fusion/data/")

for mtx_path in mtx_list[:1]:
    A = sio.mmread(mtx_path).tocsr()
    n = A.shape[0]
    Ap = A.indptr.astype(np.int32)
    Ai = A.indices.astype(np.int32)
    Ax = A.data.astype(np.float32)
    x = np.random.rand(n).astype(np.float32)

    out = sptrsv_csr_numba(Ap, Ai, Ax, x)
