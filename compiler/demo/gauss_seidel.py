"""Gauss-Seidel examples with an enclosing host-side sweep loop."""

from compiler.decorator import parallel


FUNCTION_PROPS_CSR = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
}


@parallel(block_map="i", function_props=FUNCTION_PROPS_CSR, execution_mode="wavefront", threads_per_block=128)
def simple_gauss_seidel_csr(Ap, Ai, Ax, D, b, x, n, max_iter):
    for sweep in range(max_iter):
        for i in range(n):
            x[i] = b[i]
            for j in range(Ap[i], Ap[i + 1]):
                x[i] -= Ax[j] * x[Ai[j]]
            x[i] /= D[i]
    return x


@parallel(block_map="i", function_props=FUNCTION_PROPS_CSR, execution_mode="wavefront", threads_per_block=128)
def full_gauss_seidel_csr(Ap, Ai, Ax, b, x, n, max_iter):
    for sweep in range(max_iter):
        for i in range(n):
            sigma = 0.0
            diag = 1.0
            for j in range(Ap[i], Ap[i + 1]):
                if Ai[j] == i:
                    diag = Ax[j]
                else:
                    sigma += Ax[j] * x[Ai[j]]
            x[i] = (b[i] - sigma) / diag
    return x

