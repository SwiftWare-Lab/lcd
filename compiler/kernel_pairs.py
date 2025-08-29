import numpy as np

'''
SpMV-SpMV (Same Square Sparse Matrix)
'''

def spmv_spmv(Ap, Ai, Ax, B, m):
    y = np.zeros(m)
    z = np.zeros(m)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            y[i1] += Ax[i2] * B[Ai[i2]]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            z[j1] += Ax[j2] * y[Ai[j2]]
    return z


'''
SpMM-SpMM (Same Square Sparse Matrix)
'''


def spmm_spmm(Ap, Ai, Ax, B, m, n):
    Y = np.zeros(m)
    Z = np.zeros(m)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1 * n + i3] += Ax[i2] * B[Ai[i2] * n + i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Z[j1 * n + j3] += Ax[j2] * Y[Ai[j2] * n + j3]
    return Z


'''
SpMM-GeMM (Unfused - Square Sparse Matrix)
GeMM
SpMM
A(m*m) sparse
B(m*k)
C(k*n)
'''


def gemm_spmm(Ap, Ai, Ax ,B, C, m, n, k):
    X = np.zeros(m*n)
    Y = np.zeros(m*n)
    for i1 in range(m):
        for i2 in range(k):
            for i3 in range(n):
                X[i1][i3] += B[i1][i2] * C[i2][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1+1]):
            for j3 in range(n):
                Y[j1][j3] += Ax[j2] * X[Ai[j2]][j3]


'''
SpMM-GeMM (Unfused - Square Sparse Matrix)
SpMM
GeMM
A(m*m) sparse
B(m*k)
C(k*n)
'''


def spmm_gemm(Ap, Ai, Ax, B, C, m, n, k):
    X = np.zeros(m*k)
    Y = np.zeros(m*n)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1+1]):
            for i3 in range(k):
                X[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for i1 in range(m):
        for i2 in range(k):
            for i3 in range(n):
                Y[i1][i3] += X[i1][i2] * C[i2][i3]



'''
SDDMM-SpMM ( Fused SDDMM)
A(m*m) sparse
C(m*k)
B(m*k)
D(m*n)
Z = (A.(BxC))xD
'''

def sddmm_spmm(m, n, k, nnz, Ap, Ai, B, C, D):
    Xx = np.zeros(nnz)
    Y = np.zeros(m*n)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1 * k + i3] * C[Ai[i2] * k + i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1 * n + j3] += Xx[j2] * D[j1 * n + j3]
    return Y

'''
SDDMM-SpMM ( Unfused SDDMM - square sparse matrix)
A(m*m) sparse
C(m*k)
B(m*k)
D(m*n)
Z = (A.(BxC))xD
'''

def sddmm_spmm(m, n, k, p, nnz, Ap, Ai, Ax, B, C, D):
    X = np.zeros(m*m)
    Yx = np.zeros(nnz)
    Z = np.zeros(m*n)
    for i1 in range(m):
        for i2 in range(m):
            for i3 in range(k):
                X[i1 * m + i2] += B[i1 * k + i3] * C[i2 * k + i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            Yx[j2] = Ax[j2] * X[j1 * m + Ai[j2]]
    for l1 in range(m):
        for l2 in range(Ap[l1], Ap[l1 + 1]):
            for l3 in range(n):
                Z[l1 * n + l3] += Yx[l2] * D[l1 * n + l3]

    return Z

'''
SDDMM-SpMM-GeMM ( Fused SDDMM)
SDDMM
SpMM
GeMM
A(m*m) sparse
C(m*k)
B(m*k)
D(m*n)
E(n*l)
Z = (A.(BxC))xDxE
'''

def sddmm_spmm_gemm(m, n, k, l, nnz, Ap, Ai, B, C, D, E):
    Xx = np.zeros(nnz)
    Y = np.zeros(m*n)
    Z = np.zeros(m*l)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Y[j1][j3] += Xx[j2] * D[j1][j3]
    for p1 in range(m):
        for p2 in range(n):
            for p3 in range(l):
                Z[p1][p3] += Y[p1][p2] * E[p2][p3]
    return Y

'''
SDDMM-SpMM-GeMM (Fused SDDMM)
GeMM
SDDMM
SpMM
A(m*m) sparse
C(m*k)
B(m*k)
D(m*n)
E(n*l)
Z = (A.(BxC))xDxE
'''
def sddmm_spmm_gemm(m, n, k, l, nnz, Ap, Ai, B, C, D, E):
    Xx = np.zeros(nnz)
    Y = np.zeros(m*l)
    Z = np.zeros(m*l)
    for p1 in range(m):
        for p2 in range(n):
            for p3 in range(l):
                Y[p1][p3] += D[p1][p2] * E[p2][p3]
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(k):
                Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                Z[j1][j3] += Xx[j2] * Y[j1][j3]
    return Y


'''
Triangle Counting (unweighted undirected graph)
Algebraic notation:
A(n*n): sparse matrix
X(n*n): dense matrix
R(n*n): sparse matrix(Ri = Ai, Rp = Ap)
X = AxA
R = A.X
SumTri = ComponentSum(R)/6
'''


def tri_count(Ap, Ai, nnz, n):
    X = np.zeros(n*n)
    Rx = np.zeros(nnz)
    sum_tri = 0
    for i in range(n):
        for j in range(Ap[i], Ap[i + 1]):
            for p in range(Ap[Ai[j]], Ap[Ai[j] + 1]):
                X[i][Ai[p]] += 1
    for i in range(n):
        for j in range(Ap[i], Ap[i + 1]):
            Rx[j] = X[i][Ai[j]]
    for i in range(n):
        for j in range(Ap[i], Ap[i + 1]):
            sum_tri += Rx[j]
    print(sum_tri)
    sum_tri = sum_tri / 6
    return sum_tri


'''
Softmax-SpMM (Sparsified Weight Matrix)
Algebraic notation:
A(m*k): sparse matrix
B(k*n): dense matrix
Y(m*n): dense matrix
Y = softmax(AxB)
'''

def spmm(Ap, Ai, Ax, B, m, n):
    Y = np.zeros(m * n)
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1 * n + i3] += Ax[i2] * B[Ai[i2] * n + i3]
    return Y

def softmax(x):
    x_max = np.max(axis=1)[0]
    z = x - x_max[:, None]
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1)[:, None]
    ret = numerator / denominator[:, None]
    return ret

def softmax_spmm(Ap, Ai, Ax, B, m, n):
    Y = np.zeros((m, n))
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                Y[i1][i3] += Ax[i2] * B[Ai[i2] * n + i3]
    for i1 in range(m):
        max_val = np.max(Y[i1][:])
        for i2 in range(n):
            Y[i1][i2] -= max_val
        sum = 0
        for i2 in range(n):
            Y[i1][i2] = np.exp(Y[i1][i2])
            sum += Y[i1][i2]
        for i2 in range(n):
            Y[i1][i2] /= sum
    return Y



'''
SpMMH-GeMM (Unfused - Square Sparse Matrix) 

version 1:
Hadamard Product
GeMM
SpMM

A(m*m) sparse
B(m*n)
C(m*n)
D(n*k)
'''

def spmmh_gemm(Ap, Ai, Ax, B, C, D, m, n, k):
    X = np.zeros((m, k), dtype=np.float32)
    X_h = np.zeros((m, n), dtype=np.float32)
    Y = np.zeros((m, k), dtype=np.float32)
    for i1 in range(m):
        for i2 in range(n):
            X_h[i1][i2] = B[i1][i2] * C[i1][i2]
    for j1 in range(m):
        for j2 in range(n):
            for j3 in range(k):
                X[j1][j3] += X_h[j1][j2] * D[j2][j3]
    for p1 in range(m):
        for p2 in range(Ap[p1], Ap[p1 + 1]):
            for p3 in range(k):
                Y[p1][p3] += Ax[p2] * X[Ai[p2]][p3]
    return Y


'''
SpMMH-GeMM (Unfused - Square Sparse Matrix) 

version 2:
Hadamard
SpMM
GeMM

A(m*m) sparse
B(m*n)
C(m*n)
D(n*k)
'''

def spmmh_gemm(Ap, Ai, Ax, B, C, D, m, n, k):
    X = np.zeros((m, n), dtype=np.float32)
    X_h = np.zeros((m, n), dtype=np.float32)
    Y = np.zeros((m, k), dtype=np.float32)
    for i1 in range(m):
        for i2 in range(n):
            X_h[i1][i2] = B[i1][i2] * C[i1][i2]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                X[j1][j3] += Ax[j2] * X_h[Ai[j2]][j3]
    for p1 in range(m):
        for p2 in range(n):
            for p3 in range(k):
                Y[p1][p3] += X[p1][p2] * D[p2][p3]
    return Y

'''
MTTKRP
A (m*n*k) sparse COO
B (n*l)
C (k*l) 
X = A.(B*C)
'''

def MTTKRP_coo(Av, Ai, Aj, Ak, B, C, m, n, k, l, nnz):
    X = np.zeros((m * l))
    for i1 in range(nnz):
        for i2 in range(l):
            X[Ai[i1]][i2] += B[Aj[i1]][i2] * C[Ak[i1]][i2] * Av[i1]


'''MTTKRP
A (m*n*k) sparse SPLATT(CSF)
B (n*l)
C (k*l) 
X = A.(B*C)
'''
#TODO: Test these two implementations
def MTTKRP(Av, Aip, Akp, Aj, Ak, B, C, m, n, k, l):
    X = np.zeros((m * l))
    for i1 in range(m):
        for i2 in range(Aip[i1], Aip[i1 + 1]):
            s = np.zeros(l)
            for i3 in range(Akp[i2], Akp[i2 + 1]):
                for i4 in range(l):
                    s[i4] += B[Aj[i3]][i4] * Av[i3]
            for ii3 in range(l):
                X[i1][ii3] += s[ii3] * C[Ak[i2]][ii3]


'''
MTTKRP-GeMM
A (m*n*k) sparse SPLATT(CSF)
B (n*l)
C (k*l) 
X = A.(B*C)xD
'''


def MTTKRP_GeMM(Av, Aip, Akp, Aj, Ak, B, C, D, m, n, k, l, q):
    X = np.zeros((m * l))
    Y = np.zeros((m * q))
    for i1 in range(m):
        for i2 in range(Aip[i1], Aip[i1 + 1]):
            s = np.zeros(l)
            for i3 in range(Akp[i2], Akp[i2 + 1]):
                for i4 in range(l):
                    s[i4] += B[Aj[i3]][i4] * Av[i3]
            for ii3 in range(l):
                X[i1][ii3] += s[ii3] * C[Ak[i2]][ii3]
    for j1 in range(m):
        for j2 in range(l):
            for j3 in range(q):
                Y[j1][j3] += X[j1][j2] * D[j2][j3]
    return Y


'''
SpTTM - SpTTM 
A (m*n*k) sparse SPLATT(CSF) Av, Aip, Ajp, Aj, Ak
B (k*l)
C (l*o)
X (m*n*l) sparse SPLATT(CSF) Xv, Xip, Xj (Xv (len(Xj)*n*l))
X = AxB
Y = XxC
'''


# TODO: Sparse LNR has 2 other implementations of SpTTM-SpTTM
# FUSE: one example is j2 -> i2
def spttm_spttm_csf(Av, Aip, Ajp, Aj, Ak, B, C, m, n, k, l, o):
    Xv = np.zeros((len(Aj) * n, l))
    Xj = Aj
    Xip = Aip
    count = 0
    for i1 in range(m):
        for i2 in range(Aip[i1], Aip[i1 + 1]):
            for i3 in range(Ajp[i2], Ajp[i2 + 1]):
                for i4 in range(l):
                    Xv[i2][i4] += Av[i3] * B[Ak[i3]][i4]
    Yv = np.zeros((len(Aj) * n, l))
    Yj = Aj
    Yip = Aip
    for j1 in range(m):
        for j2 in range(Xip[j1], Xip[j1 + 1]):
            for j3 in range(l):
                for j4 in range(o):
                    Yv[j2][j4] += Xv[j2][j3] * C[j3][j4]
    return Yv, Yip, Yj

'''EXTRA Kernels'''



def sparse_1D_conv(Ap, Ai, Ax, W, h, w, k, cin, cout):
    output_w = w - k + 1
    Y = np.zeros((cout, cin, output_w))
    for c1 in range(cout):
        for c2 in range(cin):
            for i in range(output_w):
                for j in range(-k // 2, k // 2 + 1):
                    for l in range(Ap[i + j], Ap[i + j + 1]):
                        if Ai[l] == i:
                            Y[c1][c2][i] += Ax[l] * W[c1][c2][j]
    return Y



'''
Sparse 2D Convolution (input is a sparse matrix)
Assume no padding with stride 1
Assume kernel size is odd
Assume output is dense
'''

def sparse_2D_conv(Ap, Ai, Ax, W, h, w, k, cin, cout):
    output_h = h - k + 1
    output_w = w - k + 1
    Y = np.zeros((cout, cin, output_h, output_w))
    for c1 in range(cout):
        for c2 in range(cin):
            for i1 in range(output_h):
                for i2 in range(output_w):
                    for j1 in range(-k // 2, k // 2 + 1):
                        for j2 in range(-k // 2, k // 2 + 1):
                            for l1 in range(Ap[i1 + j1], Ap[i1 + j1 + 1]):
                                if Ai[l1] == i2 + j2:
                                    Y[c1][c2][i1][i2] += Ax[l1] * W[c1][c2][j1][j2]

def batchNormalization(Ap, Ai, Ax, X, mean, var, gamma, beta, epsilon, m, n):
    Y = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            Y[i][j] = gamma[j] * (X[i][j] - mean[j]) / np.sqrt(var[j] + epsilon) + beta[j]
    return Y



