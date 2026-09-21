"""CSR and CSC sparse triangular-solve decorator examples."""

import numpy as np

from compiler.decorator import parallel


FUNCTION_PROPS_CSR = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
}

FUNCTION_PROPS_CSC = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
    ("Ap", "Ai"): ["triangularity"],
}


@parallel(loop="i", function_props=FUNCTION_PROPS_CSR, execution_mode="wavefront", threads_per_block=128)
def simple_sptrsv_csr(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(Ap[i], Ap[i + 1] - 1):
            x[i] -= Ax[j] * x[Ai[j]]
        x[i] /= Ax[Ap[i + 1] - 1]
    return x


@parallel(loop="i", function_props=FUNCTION_PROPS_CSC, execution_mode="wavefront", threads_per_block=128)
def simple_sptrsv_csc(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[i]) / Ax[Ap[i]]
        for j in range(Ap[i] + 1, Ap[i + 1]):
            x[Ai[j]] += Ax[j] * x[i]
    return x

