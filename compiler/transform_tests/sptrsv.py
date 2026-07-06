from compiler.decorator import parallel, fusion
import numpy as np
import scipy.io as scio
import sys
function_props_csr = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
}
function_props_csc = {
    'Ap': ['monotonically_increasing'],
    'Ai': [],
    ('Ap', 'Ai'): ['triangularity']
}
@parallel(loop="i", function_props=function_props_csr, execution_mode="wavefront")
def simple_sptrsv_csr(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(Ap[i], Ap[i + 1] - 1):
            x[i] -= Ax[j] * x[Ai[j]]
        x[i] /= Ax[Ap[i + 1] - 1]
    return x

@parallel(loop="i", function_props=function_props_csc, execution_mode="wavefront")
def simple_sptrsv_csc(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[i]) / Ax[Ap[i]]
        for j in range(Ap[i] + 1, Ap[i + 1]):
            x[Ai[j]] += Ax[j] * x[i]
    return x
