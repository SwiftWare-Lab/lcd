import numpy as np
from scipy.linalg import cho_factor, solve_triangular

def cholesky_left_looking_supernodal(n, c, r, values, lC, lR, Li_ptr, lValues, blockSet, supNo, timing,
                        prunePtr, pruneSet, super_max, col_max):
    """
    Python implementation of the cholesky_left_sn_07 function.

    Args:
        n (int): Size of the matrix.
        c (np.ndarray): Column pointers for the input matrix.
        r (np.ndarray): Row indices for the input matrix.
        values (np.ndarray): Values of the input matrix.
        lC (np.ndarray): Column pointers for the L matrix.
        lR (np.ndarray): Row indices for the L matrix.
        Li_ptr (np.ndarray): Row pointers for the L matrix.
        lValues (np.ndarray): Values of the L matrix.
        blockSet (np.ndarray): Block set defining supernodes.
        supNo (int): Number of supernodes.
        timing (np.ndarray): Timing information (not used in this implementation).
        prunePtr (np.ndarray): Prune pointers.
        pruneSet (np.ndarray): Prune set.
        super_max (int): Maximum size of a supernode.
        col_max (int): Maximum column size.

    Returns:
        bool: True if successful.
    """

    map_arr = np.zeros(n, dtype=int)
    xi = np.zeros(2 * supNo, dtype=int)
    contribs = np.zeros(super_max * col_max, dtype=float)

    for s in range(1, supNo + 1):
        curCol = blockSet[s - 1] if s != 0 else 0
        nxtCol = blockSet[s]
        supWdt = nxtCol - curCol
        nSupR = Li_ptr[nxtCol] - Li_ptr[curCol]

        for i, cnt in zip(range(Li_ptr[curCol], Li_ptr[nxtCol]), range(nSupR)):
            map_arr[lR[i]] = cnt

        for i in range(curCol, nxtCol):
            pad = i - curCol
            for j in range(c[i], c[i + 1]):
                lValues[lC[i] + map_arr[r[j]]] = values[j]

        cur = lValues[lC[curCol]:].reshape(nSupR, supWdt, order='F')

        for i in range(prunePtr[s - 1], prunePtr[s]):
            lSN = pruneSet[i]
            cSN = blockSet[lSN]
            cNSN = blockSet[lSN + 1]
            Li_ptr_cNSN = Li_ptr[cNSN]
            Li_ptr_cSN = Li_ptr[cSN]
            nSNRCur = Li_ptr_cNSN - Li_ptr_cSN
            supWdts = cNSN - cSN
            lb = 0
            ub = 0
            sw = True

            for j in range(Li_ptr_cSN, Li_ptr_cNSN):
                if lR[j] >= curCol and sw:
                    lb = j - Li_ptr_cSN
                    sw = False
                if lR[j] < curCol + supWdt and not sw:
                    ub = j - Li_ptr_cSN

            nSupRs = Li_ptr_cNSN - Li_ptr_cSN - lb
            ndrow1 = ub - lb + 1
            ndrow3 = nSupRs - ndrow1
            src = lValues[lC[cSN] + lb:lC[cSN] + lb + nSNRCur * supWdts].reshape(nSNRCur, supWdts, order='F')
            srcL = lValues[lC[cSN] + ub + 1:lC[cSN] + ub + 1 + nSNRCur * supWdts].reshape(nSNRCur, supWdts, order='F')

            contribs[:] = 0.0

            contribs[:ndrow1 * nSupRs].reshape(nSupRs, ndrow1, order='F')[:, :ndrow1] = src[:nSNRCur, :supWdts][:ndrow1].T @ src[:nSNRCur, :supWdts][:ndrow1]

            if ndrow3 > 0:
                contribs[ndrow1 * nSupRs:].reshape(nSupRs, ndrow1, order='F')[ndrow1:, :ndrow1] = srcL[:nSNRCur, :supWdts][ndrow1:].T @ src[:nSNRCur, :supWdts][:ndrow1]

            for i in range(ndrow1):
                col = map_arr[lR[Li_ptr_cSN + i + lb]]
                for j in range(i, nSupRs):
                    cRow = lR[Li_ptr_cSN + j + lb]
                    cur[map_arr[cRow], col] -= contribs[i * nSupRs + j]

        cur_part = cur[:supWdt, :supWdt]
        cur[:supWdt, :supWdt] = cho_factor(cur_part, lower=True)[0]
        rowNo = nSupR - supWdt

        cur[supWdt:nSupR, :supWdt] = solve_triangular(cur[:supWdt, :supWdt], cur[supWdt:nSupR, :supWdt], lower=True, trans='T')

    return True