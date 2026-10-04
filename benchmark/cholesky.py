

import numpy as np
import math

import numba
from numba import njit, prange
import threading

from utils.symbolic_cholesky_csc import ereach


def cholesky_left_looking(n, c, r, values, lC, lR, lValues, prunePtr, pruneSet):
    """
    Performs a Cholesky decomposition on a given matrix (c, r, values), i.e.
    stored in compressed column format, and produces L, which are
    stored in column compressed format.

    Args:
        n (int): Size of the matrix.
        c (np.ndarray): Column pointers for the input matrix.
        r (np.ndarray): Row indices for the input matrix.
        values (np.ndarray): Values of the input matrix.
        lC (np.ndarray): Column pointers for the L matrix.
        lR (np.ndarray): Row indices for the L matrix.
        lValues (np.ndarray): Values of the L matrix (modified in place).
        prunePtr (np.ndarray): Prune pointers.
        pruneSet (np.ndarray): Prune set.

    Returns:
        bool: True if successful, False if the matrix is not SPD.
    """

    f = np.zeros(n)
    finger = np.array(lC, copy=True)  # Initialize finger with a copy of lC
    for colNo in range(n):
        # Uncompressing a col into a 1D array
        for nzNo in range(c[colNo], c[colNo + 1]):
            f[r[nzNo]] = values[nzNo]  # Copying nonzero of the col

        for i in range(prunePtr[colNo], prunePtr[colNo + 1] - 1):
            spCol = pruneSet[i]
            for l in range(lC[spCol], lC[spCol + 1]):
                if lR[l] >= colNo:
                    f[lR[l]] -= lValues[l] * lValues[finger[spCol]]
                    #print("in col: ", colNo, " for spcol: ", spCol, " row finger is:", lR[finger[spCol]])
            finger[spCol] += 1

        finger[colNo] += 1  # Skip diagonal row

        if f[colNo] <= 0:
            return False  # The matrix is not SPD

        tmpSqrt = math.sqrt(f[colNo])
        f[colNo] = 0
        lValues[lC[colNo]] = tmpSqrt

        for j in range(lC[colNo] + 1, lC[colNo + 1]):
            lValues[j] = f[lR[j]] / tmpSqrt
            f[lR[j]] = 0

    return True


@njit(cache=True)
def cholesky_left_looking_numba(n, c, r, values, lC, lR, lValues, prunePtr, pruneSet):
    """
    Performs a Cholesky decomposition on a given matrix (c, r, values), i.e.
    stored in compressed column format, and produces L, which are
    stored in column compressed format.

    Serial numeric kernel. Symbolic columns must be sorted with the diagonal
    first; pruning rows must contain earlier contributing columns followed by
    their own diagonal, which is skipped. Callers may supply read-only indices.

    Args:
        n (int): Size of the matrix.
        c (np.ndarray): Column pointers for the input matrix.
        r (np.ndarray): Row indices for the input matrix.
        values (np.ndarray): Values of the input matrix.
        lC (np.ndarray): Column pointers for the L matrix.
        lR (np.ndarray): Row indices for the L matrix.
        lValues (np.ndarray): Values of the L matrix (modified in place).
        prunePtr (np.ndarray): Prune pointers.
        pruneSet (np.ndarray): Prune set.

    Returns:
        bool: True if successful, False if the matrix is not SPD.
    """

    f = np.zeros(n)
    for colNo in range(n):
        lb, ub = prunePtr[colNo], prunePtr[colNo + 1] - 1
        # Uncompressing a col into a 1D array
        for nzNo in range(c[colNo], c[colNo + 1]):
            f[r[nzNo]] = values[nzNo]  # Copying nonzero of the col
        for i in range(lb, ub):
            spCol = pruneSet[i]
            facing_val = 0.0
            facing_idx = -1
            for l in range(lC[spCol], lC[spCol + 1]):
                if lR[l] == colNo:
                    facing_val = lValues[l]
                    facing_idx = l
                    break
            if facing_idx < 0:
                return False  # Inconsistent symbolic dependency; avoid index -1.
            for l in range(facing_idx, lC[spCol + 1]):
                f[lR[l]] -= lValues[l] * facing_val
        if f[colNo] <= 0 or not np.isfinite(f[colNo]):
            return False  # The matrix is not SPD
        tmpSqrt = math.sqrt(f[colNo])
        f[colNo] = 0
        lValues[lC[colNo]] = tmpSqrt
        for j in range(lC[colNo] + 1, lC[colNo + 1]):
            lValues[j] = f[lR[j]] / tmpSqrt
            f[lR[j]] = 0
    return True


# Define a lock
lock = threading.Lock()

@njit(numba.void(numba.int32, numba.int32[:], numba.int32[:], numba.float64[:],
                 numba.int32[:], numba.int32[:], numba.float64[:], numba.int32[:],
                 numba.int32[:], numba.int32[:], numba.bool), parallel=True, fastmath=True)
def cholesky_left_looking_numba_parallel(n, c, r, values, lC, lR, lValues,
                                         prunePtr, pruneSet, thread_flags, ret_val):
    ret_val = True
    for colNo in prange(n):
        # print("colNo begin: ", colNo)
        # Uncompressing a col into a 1D array
        f = np.zeros(n)
        for nzNo in range(c[colNo], c[colNo + 1]):
            f[r[nzNo]] = values[nzNo]  # Copying nonzero of the col
        lb, ub = prunePtr[colNo], prunePtr[colNo + 1] - 1
        th_flag = thread_flags[colNo]
        while not th_flag:
            if ub - lb == 0:
                th_flag = True
                break
            for i in range(lb, ub):
                spCol = pruneSet[i]
                if not thread_flags[spCol]:
                    th_flag = False
                    break
                th_flag = True
        for i in range(lb, ub):
            spCol = pruneSet[i]
            facing_val, tmp = 0, 0
            facing_idx = -1
            for l in range(lC[spCol], lC[spCol + 1]):
                if lR[l] == colNo:
                    facing_val = lValues[l]
                    facing_idx = l
                    break
            for l in range(facing_idx, lC[spCol + 1]):
                f[lR[l]] -= lValues[l] * facing_val
        if f[colNo] <= 0:
            ret_val = False  # The matrix is not SPD
        tmpSqrt = math.sqrt(f[colNo])
        f[colNo] = 0
        lValues[lC[colNo]] = tmpSqrt
        for j in range(lC[colNo] + 1, lC[colNo + 1]):
            lValues[j] = f[lR[j]] / tmpSqrt
            f[lR[j]] = 0
        with numba.objmode():
            thread_flags[colNo] = True



def cholesky_left_looking_coupled(n, c, r, values, lC, lR, lValues, etree):
    """
    Performs a Cholesky decomposition on a given matrix (c, r, values), i.e.
    stored in compressed column format, and produces L, which are
    stored in column compressed format.

    Args:
        n (int): Size of the matrix.
        c (np.ndarray): Column pointers for the input matrix.
        r (np.ndarray): Row indices for the input matrix.
        values (np.ndarray): Values of the input matrix.
        lC (np.ndarray): Column pointers for the L matrix.
        lR (np.ndarray): Row indices for the L matrix.
        lValues (np.ndarray): Values of the L matrix (modified in place).
        prunePtr (np.ndarray): Prune pointers.
        pruneSet (np.ndarray): Prune set.

    Returns:
        bool: True if successful, False if the matrix is not SPD.
    """

    f = np.zeros(n)
    finger = np.array(lC, copy=True)  # Initialize finger with a copy of lC
    xi = np.zeros(2 * n, dtype=np.int32)
    for colNo in range(n):
        # Uncompressing a col into a 1D array
        for nzNo in range(c[colNo], c[colNo + 1]):
            if r[nzNo] >= colNo:
                f[r[nzNo]] = values[nzNo]  # Copying nonzero of the col

        top = ereach(n, c, r, colNo, etree, xi[0:n], xi[n:])
        #print(colNo, " ", xi[top:n])
        for i in range(top, n):
            spCol = xi[i]
            for l in range(lC[spCol], lC[spCol + 1]):
                if lR[l] >= colNo:
                    f[lR[l]] -= lValues[l] * lValues[finger[spCol]]
            finger[spCol] += 1

        finger[colNo] += 1  # Skip diagonal row

        if f[colNo] <= 0:
            return False  # The matrix is not SPD

        tmpSqrt = math.sqrt(f[colNo])
        f[colNo] = 0
        lValues[lC[colNo]] = tmpSqrt

        for j in range(lC[colNo] + 1, lC[colNo + 1]):
            lValues[j] = f[lR[j]] / tmpSqrt
            f[lR[j]] = 0

    return True
