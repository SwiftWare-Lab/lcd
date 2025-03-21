

import numpy as np
import math


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