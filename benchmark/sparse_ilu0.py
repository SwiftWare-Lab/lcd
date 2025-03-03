import sys
import numpy as np

def read_mtx(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header_found = False
    data_list = []
    nrows = ncols = nnz = 0
    for line in lines:
        line = line.strip()
        if not line or line[0] == '%':
            continue
        if not header_found:
            parts = line.split()
            nrows, ncols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
            header_found = True
        else:
            parts = line.split()
            if len(parts) < 3:
                continue
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
            val = float(parts[2])
            data_list.append((row, col, val))
    data_list = np.array(data_list)
    rows = data_list[:, 0].astype(int)
    cols = data_list[:, 1].astype(int)
    vals = data_list[:, 2]
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    indptr = np.zeros(nrows + 1, dtype=int)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    return vals, cols, indptr

def csr_find(indptr, indices, row, col):
    for idx in range(indptr[row], indptr[row + 1]):
        if indices[idx] == col:
            return idx
    return -1

def spilu_csr(A_data, A_indices, A_indptr):
    n = len(A_indptr) - 1
    ILU = A_data.copy()
    for i in range(n):
        for jj in range(A_indptr[i], A_indptr[i + 1]):
            j = A_indices[jj]
            sum_val = ILU[jj]
            for kk in range(A_indptr[i], jj):
                k = A_indices[kk]
                pos = csr_find(A_indptr, A_indices, k, j)
                if pos != -1:
                    sum_val -= ILU[kk] * ILU[pos]
            ILU[jj] = sum_val
            if j < i:
                diag_index = csr_find(A_indptr, A_indices, j, j)
                ILU[jj] /= ILU[diag_index]
    return ILU

def csr_matvec(A_data, A_indices, A_indptr, x):
    n = len(A_indptr) - 1
    y = np.zeros(n)
    for i in range(n):
        for idx in range(A_indptr[i], A_indptr[i + 1]):
            y[i] += A_data[idx] * x[A_indices[idx]]
    return y

def ilu_forward_solve(ILU, A_indices, A_indptr, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        sum_val = b[i]
        for idx in range(A_indptr[i], A_indptr[i + 1]):
            if A_indices[idx] < i:
                sum_val -= ILU[idx] * y[A_indices[idx]]
        y[i] = sum_val
    return y

def ilu_backward_solve(ILU, A_indices, A_indptr, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = y[i]
        diag = None
        for idx in range(A_indptr[i], A_indptr[i + 1]):
            col = A_indices[idx]
            if col == i:
                diag = ILU[idx]
            elif col > i:
                sum_val -= ILU[idx] * x[col]
        x[i] = sum_val / diag
    return x

def ilu_preconditioner(ILU, A_indices, A_indptr, r):
    y = ilu_forward_solve(ILU, A_indices, A_indptr, r)
    z = ilu_backward_solve(ILU, A_indices, A_indptr, y)
    return z

def pcg_with_ilu(A_data, A_indices, A_indptr, b, ILU, tol=1e-8, maxiter=1000):
    n = len(b)
    x = np.zeros(n)
    r = b - csr_matvec(A_data, A_indices, A_indptr, x)
    z = ilu_preconditioner(ILU, A_indices, A_indptr, r)
    p = z.copy()
    rz_old = np.dot(r, z)
    for it in range(maxiter):
        Ap = csr_matvec(A_data, A_indices, A_indptr, p)
        alpha = rz_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) < tol:
            break
        z = ilu_preconditioner(ILU, A_indices, A_indptr, r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x
