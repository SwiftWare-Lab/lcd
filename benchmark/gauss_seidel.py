
def gauss_seidel_csr(A, IA, JA, b, x, tol=1e-6, max_iter=100):
    n = len(IA) - 1
    for iter in range(max_iter):
        norm = 0.0
        for i in range(n):
            x_old = x[i]
            sum = 0.0
            for j in range(IA[i], IA[i + 1]):
                if i != JA[j]:
                    sum += A[j] * x[JA[j]]
                else:
                    diag = A[j]
            x[i] = (b[i] - sum) / diag
            norm += (x[i] - x_old) ** 2
        if norm < tol:
            break
    return x, iter


def gauss_seidel_diag_csr(A, Ap, Ai, Diag, b, x, tol=1e-6, max_iter=100):
    n, iter = len(Ap) - 1, 0
    for iter in range(max_iter):
        norm = 0.0
        for i in range(n):
            x_old = x[i]
            sum = 0.0
            for j in range(Ap[i], Ap[i + 1]):
                sum += A[j] * x[Ai[j]]
            x[i] = (b[i] - sum) / Diag[i]
            norm += (x[i] - x_old) ** 2
        if norm < tol:
            break
    return x, iter