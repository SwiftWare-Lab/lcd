import time

import numpy as np
import scipy.sparse as sp
import scipy
from benchmark.bench_utils import compact_2d_list

EMPTY = -1


def subtree(j, k, Ap, Ai, Anz, SuperMap, Sparent, mark, sorted, k1, Flag, Ls, Lpi2):
    p = Ap[j]
    pend = Ap[j + 1] if Anz is None else p + Anz[j]
    while p < pend:
        i = Ai[p]
        if i < k1:
            si = SuperMap[i]
            while Flag[si] < mark:
                assert si <= SuperMap[k]
                Ls[Lpi2[si]] = k
                Lpi2[si] += 1
                Flag[si] = mark
                si = Sparent[si]
        elif sorted:
            break
        p += 1


def super_symbolic2(for_whom, A, F, Parent, L, Common):
    """
    Perform symbolic analysis for Cholesky or SPQR factorization.

    Parameters
    ----------
    for_whom : int
        FOR_SPQR (0): for SPQR but not GPU-accelerated
        FOR_CHOLESKY (1): for Cholesky (GPU or not)
        FOR_SPQRGPU (2): for SPQR with GPU acceleration
    A : cholmod_sparse
        Matrix to analyze
    F : cholmod_sparse
        F = A' or A[:,f]'
    Parent : np.ndarray
        Elimination tree
    L : cholmod_factor
        Simplicial symbolic on input, supernodal symbolic on output
    Common : cholmod_common
        Common workspace and parameters
    """
    # Check inputs
    zrelax0, zrelax1, zrelax2 = 0.0, 0.0, 0.0
    n = A.shape[0]

    # Allocate workspace
    ok = True
    w = n * 5
    if not ok:
        raise MemoryError("problem too large")

    # Initialize variables
    Ap, Ai, Anz = A.indptr, A.indices, A.data
    if A.format != 'csc':
        raise ValueError("Matrix A must be in CSC format")

    if F is not None:
        Fp, Fj, Fnz = F.indptr, F.indices, F.data
        packed = F.has_sorted_indices
    else:
        Fp, Fj, Fnz = None, None, None
        packed = True

    ColCount = L.ColCount
    nrelax0, nrelax1, nrelax2 = Common.nrelax
    zrelax0, zrelax1, zrelax2 = Common.zrelax

    # Find the fundamental supernodes
    Wi = np.zeros(n, dtype=int)
    for j in range(n):
        parent = Parent[j]
        if parent != -1:
            Wi[parent] += 1

    Super = np.zeros(n + 1, dtype=int)
    nfsuper = 1 if n > 0 else 0
    Super[0] = 0

    for j in range(1, n):
        if Parent[j - 1] != j or ColCount[j - 1] != ColCount[j] + 1 or Wi[j] > 1:
            Super[nfsuper] = j
            nfsuper += 1
    Super[nfsuper] = n

    Nscol = Wi
    SuperMap = np.zeros(n, dtype=int)
    for s in range(nfsuper):
        for k in range(Super[s], Super[s + 1]):
            SuperMap[k] = s

    Sparent = np.zeros(nfsuper, dtype=int)
    for s in range(nfsuper):
        j = Super[s + 1] - 1
        parent = Parent[j]
        Sparent[s] = -1 if parent == -1 else SuperMap[parent]

    Zeros = np.zeros(nfsuper, dtype=int)
    Merged = np.full(nfsuper, -1, dtype=int)
    Snz = np.zeros(nfsuper, dtype=int)
    for s in range(nfsuper):
        Nscol[s] = Super[s + 1] - Super[s]
        Snz[s] = ColCount[Super[s]]

    for s in range(nfsuper - 2, -1, -1):
        ss = Sparent[s]
        if ss == -1:
            continue

        for ss in range(Sparent[s], -1, Sparent[ss]):
            if Merged[ss] != -1:
                ss = Merged[ss]

        if ss != s + 1:
            continue

        nscol0, nscol1 = Nscol[s], Nscol[s + 1]
        ns = nscol0 + nscol1
        totzeros = Zeros[s + 1]
        lnz1 = Snz[s + 1]

        if ns <= nrelax0:
            merge = True
        else:
            lnz0 = Snz[s]
            newzeros = nscol0 * (Snz[s + 1] + nscol0 - Snz[s])
            if newzeros == 0:
                merge = True
            else:
                xtotzeros = totzeros + newzeros
                xtotsize = (ns * (ns + 1) / 2) + ns * (lnz1 - nscol1)
                z = xtotzeros / xtotsize
                merge = ((ns <= nrelax1 and z < zrelax0) or
                         (ns <= nrelax2 and z < zrelax1) or
                         (z < zrelax2)) and (xtotsize < np.iinfo(np.int64).max)

        if merge:
            Zeros[s] = totzeros + newzeros
            Merged[s + 1] = s
            Snz[s] = nscol0 + Snz[s + 1]
            Nscol[s] += Nscol[s + 1]

    nsuper = 0
    for s in range(nfsuper):
        if Merged[s] == -1:
            Super[nsuper] = Super[s]
            Snz[nsuper] = Snz[s]
            nsuper += 1
    Super[nsuper] = n

    SuperMap = np.zeros(n, dtype=int)
    for s in range(nsuper):
        for k in range(Super[s], Super[s + 1]):
            SuperMap[k] = s

    Sparent = np.zeros(nsuper, dtype=int)
    for s in range(nsuper):
        j = Super[s + 1] - 1
        parent = Parent[j]
        Sparent[s] = -1 if parent == -1 else SuperMap[parent]

    ssize, xsize = 0, 0
    find_xsize = for_whom in [1, 2]
    for s in range(nsuper):
        nscol = Super[s + 1] - Super[s]
        nsrow = Snz[s]
        ssize += nsrow
        if find_xsize:
            xsize += nscol * nsrow

    L.ssize = max(1, ssize)
    L.xsize = max(1, xsize)
    L.nsuper = nsuper

    Lpi = np.zeros(nsuper + 1, dtype=int)
    Lpx = np.zeros(nsuper + 1, dtype=int)
    Ls = np.zeros(ssize, dtype=int)
    Lsuper = np.zeros(nsuper + 1, dtype=int)

    for s in range(nsuper + 1):
        Lsuper[s] = Super[s]

    p = 0
    for s in range(nsuper):
        Lpi[s] = p
        p += Snz[s]
    Lpi[nsuper] = p

    if for_whom in [1, 2]:
        p = 0
        for s in range(nsuper):
            nscol = Super[s + 1] - Super[s]
            nsrow = Snz[s]
            Lpx[s] = p
            p += nscol * nsrow
        Lpx[nsuper] = p

    Lpi2 = np.copy(Lpi)
    Asorted = A.has_sorted_indices

    for s in range(nsuper):
        k1, k2 = Super[s], Super[s + 1]
        for k in range(k1, k2):
            Ls[Lpi2[s]] = k
            Lpi2[s] += 1

        for k in range(k1, k2):
            mark = np.zeros(n, dtype=bool)
            mark[s] = True

            if A.format != 'csc':
                raise ValueError("Matrix A must be in CSC format")

            if F is not None:
                p = Fp[k]
                pend = Fp[k + 1] if packed else p + Fnz[k]
                for p in range(p, pend):
                    subtree(Fj[p], k, Ap, Ai, Anz, SuperMap, Sparent, mark, Asorted, k1, Ls, Lpi2)
            else:
                subtree(k, k, Ap, Ai, Anz, SuperMap, Sparent, mark, Asorted, k1, Ls, Lpi2)
    return True


def post_order(Parent, n_input, Weight, Post, Common):
    """
    Perform postordering of an elimination tree.

    Parameters
    ----------
    Parent : np.ndarray
        Size n. Parent[j] = p if p is the parent of j
    n_input : int
        Number of nodes
    Weight : np.ndarray or None
        Size n, optional. Weight[j] is weight of node j
    Post : np.ndarray
        Size n. Post[k] = j is kth in postordered tree
    Common : dict
        Common workspace and parameters

    Returns
    -------
    int
        Number of nodes postordered
    """
    n = int(n_input)
    ok = True

    if Parent is None or Post is None or Common is None:
        return -1

    Common['status'] = 'CHOLMOD_OK'

    # Allocate workspace
    s = 2 * n
    if not ok:
        raise MemoryError("problem too large")

    Common['Head'] = np.full(n, -1, dtype=int)
    Common['Iwork'] = np.zeros(2 * n, dtype=int)
    Head = Common['Head']
    Iwork = Common['Iwork']
    Next = Iwork[:n]
    Pstack = Iwork[n:]

    # Construct a link list of children for each node
    if Weight is None:
        for j in range(n - 1, -1, -1):
            p = Parent[j]
            if 0 <= p < n:
                Next[j] = Head[p]
                Head[p] = j
    else:
        Whead = Pstack
        Whead.fill(-1)
        for j in range(n):
            p = Parent[j]
            if 0 <= p < n:
                w = Weight[j]
                w = max(0, w)
                w = min(w, n - 1)
                Next[j] = Whead[w]
                Whead[w] = j
        for w in range(n - 1, -1, -1):
            j = Whead[w]
            while j != -1:
                nextj = Next[j]
                p = Parent[j]
                Next[j] = Head[p]
                Head[p] = j
                j = nextj

    # Start a DFS at each root node of the etree
    k = 0
    for j in range(n):
        if Parent[j] == -1:
            k = dfs(j, k, Post, Head, Next, Pstack)

    Head.fill(-1)
    return k


def dfs(p, k, Post, Head, Next, Pstack):
    """
    Perform a depth-first search and return the new value of k.

    Parameters
    ----------
    p : int
        Start the DFS at a root node p
    k : int
        Start the node numbering at k
    Post : np.ndarray
        Post ordering, modified on output
    Head : np.ndarray
        Head[p] = youngest child of p; EMPTY on output
    Next : np.ndarray
        Next[j] = sibling of j; unmodified
    Pstack : np.ndarray
        Workspace of size n, undefined on input or output

    Returns
    -------
    int
        The next node will be numbered k
    """
    phead = 0
    Pstack[phead] = p

    while phead >= 0:
        p = Pstack[phead]
        j = Head[p]
        if j == EMPTY:
            phead -= 1
            Post[k] = p
            k += 1
        else:
            Head[p] = Next[j]
            phead += 1
            Pstack[phead] = j

    return k


def initialize_node(k, Post, Parent, ColCount, PrevNbr):
    """
    Initial work for kth node in postordered etree.

    Parameters
    ----------
    k : int
        At the kth step of the algorithm (and kth node)
    Post : np.ndarray
        Post[k] = i, the kth node in postordered etree
    Parent : np.ndarray
        Parent[i] is the parent of i in the etree
    ColCount : np.ndarray
        ColCount[c] is the current weight of node c
    PrevNbr : np.ndarray
        PrevNbr[u] = k if u was last considered at step k

    Returns
    -------
    int
        The kth node in the postordered etree
    """
    p = Post[k]
    parent = Parent[p]
    if parent != EMPTY:
        ColCount[parent] -= 1
    PrevNbr[p] = k
    return p


def process_edge(p, u, k, First, PrevNbr, ColCount, PrevLeaf, RowCount, SetParent, Level):
    """
    Process edge (p, u) of the matrix.

    Parameters
    ----------
    p : int
        Process edge (p, u) of the matrix
    u : int
        Node u
    k : int
        We are at the kth node in the postordered etree
    First : np.ndarray
        First[i] = k if the postordering of first descendant of node i is k
    PrevNbr : np.ndarray
        u was last considered at step k = PrevNbr[u]
    ColCount : np.ndarray
        ColCount[c] is the current weight of node c
    PrevLeaf : np.ndarray
        s = PrevLeaf[u] means that s was the last leaf seen in the subtree rooted at u
    RowCount : np.ndarray
        RowCount[i] is # of nonzeros in row i of L, including the diagonal. Not computed if None
    SetParent : np.ndarray
        The FIND/UNION data structure, which forms a set of trees
    Level : np.ndarray
        Level[i] = length of path from node i to root
    """
    if First[p] > PrevNbr[u]:
        ColCount[p] += 1
        prevleaf = PrevLeaf[u]
        if prevleaf == EMPTY:
            q = u
        else:
            q = prevleaf
            while q != SetParent[q]:
                q = SetParent[q]
            s = prevleaf
            while s != q:
                sparent = SetParent[s]
                SetParent[s] = q
                s = sparent
            ColCount[q] -= 1
        if RowCount is not None:
            RowCount[u] += (Level[p] - Level[q])
        PrevLeaf[u] = p
    PrevNbr[u] = k


def finalize_node(p, Parent, SetParent):
    """
    Compute UNION (p, Parent[p]).

    Parameters
    ----------
    p : int
        Node p
    Parent : np.ndarray
        Parent[p] is the parent of p in the etree
    SetParent : np.ndarray
        The FIND/UNION data structure, which forms a set of trees
    """
    if Parent[p] != EMPTY:
        SetParent[p] = Parent[p]


def row_col_counts(A, fset, fsize, Parent, Post, RowCount, ColCount, First, Level, Common):
    """
    Perform row and column counts for Cholesky or SPQR factorization.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
        Matrix to analyze, symmetric lower or unsymmetric
    fset : np.ndarray
        Subset of 0:(A.shape[1])-1
    fsize : int
        Size of fset
    Parent : np.ndarray
        Size nrow. Parent[i] = p if p is the parent of i
    Post : np.ndarray
        Size nrow. Post[k] = i if i is the kth node in the postordered etree
    RowCount : np.ndarray
        Size nrow. RowCount[i] = # entries in the ith row of L, including the diagonal
    ColCount : np.ndarray
        Size nrow. ColCount[i] = # entries in the ith column of L, including the diagonal
    First : np.ndarray
        Size nrow. First[i] = k is the least postordering of any descendant of i
    Level : np.ndarray
        Size nrow. Level[i] is the length of the path from i to the root, with Level[root] = 0
    Common : dict
        Common workspace and parameters

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    nrow = A.shape[0]
    ncol = A.shape[1]
    stype = A.getformat() == 'csc'
    packed = A.has_sorted_indices

    if stype > 0:
        raise ValueError("symmetric upper not supported")

    Common['status'] = 'CHOLMOD_OK'

    # Allocate workspace
    Iwork = np.zeros(2 * nrow + (0 if stype else ncol), dtype=int)
    SetParent = Iwork[:nrow]
    PrevNbr = Iwork[nrow:2 * nrow]
    Anext = Iwork[2 * nrow:] if not stype else None
    PrevLeaf = np.full(nrow, -1, dtype=int)
    Head = np.full(nrow + 1, -1, dtype=int) if not stype else None

    # Initialize First and Level
    First.fill(-1)
    for k in range(nrow):
        i = Post[k]
        ColCount[i] = 1 if First[i] == -1 else 0
        len = 0
        r = i
        while r != -1 and First[r] == -1:
            First[r] = k
            r = Parent[r]
            len += 1
        if r == -1:
            len -= 1
        else:
            len += Level[r]
        s = i
        while s != r:
            Level[s] = len
            len -= 1
            s = Parent[s]

    # AA' case: sort columns of A according to first postordered row index
    fl = 0.0
    if stype == 0:
        Ipost = PrevNbr
        for k in range(nrow):
            Ipost[Post[k]] = k
        use_fset = fset is not None
        if use_fset:
            nf = fsize
            Anext.fill(-2)
            for jj in range(nf):
                j = fset[jj]
                if j < 0 or j >= ncol or Anext[j] != -2:
                    raise ValueError("fset invalid")
                Anext[j] = -1
        else:
            nf = ncol
        for jj in range(nf):
            j = fset[jj] if use_fset else jj
            p = A.indptr[j]
            pend = A.indptr[j + 1] if packed else p + A.data[j]
            ff = max(0, pend - p)
            fl += ff * ff + ff
            if pend > p:
                k = Ipost[A.indices[p]]
                for pp in range(p, pend):
                    inew = Ipost[A.indices[pp]]
                    k = min(k, inew)
                Anext[j] = Head[k]
                Head[k] = j

    # Compute the row counts and node weights
    if RowCount is not None:
        RowCount.fill(1)
    PrevLeaf.fill(-1)
    PrevNbr.fill(-1)
    SetParent[:] = np.arange(nrow)

    if stype != 0:
        anz = nrow
        for k in range(nrow):
            j = initialize_node(k, Post, Parent, ColCount, PrevNbr)
            p = A.indptr[j]
            pend = A.indptr[j + 1] if packed else p + A.data[j]
            for pp in range(p, pend):
                i = A.indices[pp]
                if i > j:
                    anz += 1
                    process_edge(j, i, k, First, PrevNbr, ColCount, PrevLeaf, RowCount, SetParent, Level)
            finalize_node(j, Parent, SetParent)
        Common['anz'] = anz
    else:
        for k in range(nrow):
            inode = initialize_node(k, Post, Parent, ColCount, PrevNbr)
            for j in range(Head[k], -1, Anext[j]):
                p = A.indptr[j]
                pend = A.indptr[j + 1] if packed else p + A.data[j]
                for pp in range(p, pend):
                    i = A.indices[pp]
                    if PrevNbr[i] < k:
                        process_edge(inode, i, k, First, PrevNbr, ColCount, PrevLeaf, RowCount, SetParent, Level)
            Head[k] = -1
            finalize_node(inode, Parent, SetParent)

    # Finish computing the column counts
    for j in range(nrow):
        parent = Parent[j]
        if parent != -1:
            ColCount[parent] += ColCount[j]

    # Clear workspace
    Common['mark'] = -1
    Common['fl'] = fl
    Common['lnz'] = sum(ColCount)
    return True


def update_etree(k, i, Parent, Ancestor):
    """
    Update the elimination tree.

    Parameters
    ----------
    k : int
        Process the edge (k, i) in the input graph
    i : int
        Node i
    Parent : np.ndarray
        Parent[t] = p if p is the parent of t
    Ancestor : np.ndarray
        Ancestor[t] is the ancestor of node t in the partially-constructed etree
    """
    while True:
        a = Ancestor[k]
        if a == i:
            return
        Ancestor[k] = i
        if a == EMPTY:
            Parent[k] = i
            return
        k = a


def etree(A, Parent, Common):
    """
    Find the elimination tree of A or A'*A.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
        Input matrix
    Parent : np.ndarray
        Size ncol. Parent[j] = p if p is the parent of j
    Common : dict
        Common workspace and parameters

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    nrow, ncol = A.shape
    Ap = A.indptr
    Ai = A.indices
    Anz = A.data
    packed = A.has_sorted_indices
    stype = 0  # Assuming unsymmetric case for this example

    Common['status'] = 'CHOLMOD_OK'
    Iwork = np.zeros(ncol + nrow, dtype=int)
    Ancestor = Iwork[:ncol]
    Prev = Iwork[ncol:]

    Parent.fill(EMPTY)
    Ancestor.fill(EMPTY)

    if stype > 0:
        for j in range(ncol):
            p = Ap[j]
            pend = Ap[j + 1] if packed else p + Anz[j]
            for p in range(p, pend):
                i = Ai[p]
                if i < j:
                    update_etree(i, j, Parent, Ancestor)
    elif stype == 0:
        Prev.fill(EMPTY)
        for j in range(ncol):
            p = Ap[j]
            pend = Ap[j + 1] if packed else p + Anz[j]
            for p in range(p, pend):
                i = Ai[p]
                jprev = Prev[i]
                if jprev != EMPTY:
                    update_etree(jprev, j, Parent, Ancestor)
                Prev[i] = j
    else:
        raise ValueError("symmetric lower not supported")

    return True


def analyze_ordering(A, ordering, Perm, fset, fsize, Parent, Post, ColCount, First, Level, Common):
    """
    Perform symbolic analysis for Cholesky or SPQR factorization.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
        Matrix to analyze
    ordering : int
        Ordering method used
    Perm : np.ndarray
        Size n, fill-reducing permutation to analyze
    fset : np.ndarray
        Subset of 0:(A.shape[1])-1
    fsize : int
        Size of fset
    Parent : np.ndarray
        Size n, elimination tree
    Post : np.ndarray
        Size n, postordering of elimination tree
    ColCount : np.ndarray
        Size n, nnz in each column of L
    First : np.ndarray
        Size n workspace for postorder
    Level : np.ndarray
        Size n workspace for postorder
    Common : dict
        Common workspace and parameters

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    n = A.shape[0]
    do_rowcolcounts = ColCount is not None

    # Permute A according to Perm and fset
    #ok, A1, A2, S, F = permute_matrices(A, ordering, Perm, fset, fsize, do_rowcolcounts, Common)
    F = A.T
    S = A
    ok = True
    # Find etree of S (symmetric upper/lower case) or F (unsym case)
    if ok:
        ok = etree(S if A.has_sorted_indices else F, Parent, Common)

    # Postorder the etree (required by rowcolcounts)
    if ok:
        ok = post_order(Parent, n, None, Post, Common) == n

    # Set status if postorder returns < n
    if not ok and Common['status'] == 'CHOLMOD_OK':
        Common['status'] = 'CHOLMOD_INVALID'

    # Analyze LL'=S or SS' or S(:,f)*S(:,f)'
    if do_rowcolcounts and ok:
        ok = row_col_counts(S if A.has_sorted_indices else F, fset, fsize, Parent, Post, None, ColCount, First, Level, Common)

    # Free temporary matrices and return result

    return ok


def CHOLMOD_analyze_p2(for_whom, A, UserPerm, fset, fsize, Common):
    """
    Perform symbolic analysis for Cholesky or SPQR factorization.

    Parameters
    ----------
    for_whom : int
        FOR_SPQR (0): for SPQR but not GPU-accelerated
        FOR_CHOLESKY (1): for Cholesky (GPU or not)
        FOR_SPQRGPU (2): for SPQR with GPU acceleration
    A : scipy.sparse.csc_matrix
        Matrix to order and analyze
    UserPerm : np.ndarray
        User-provided permutation, size A.shape[0]
    fset : np.ndarray
        Subset of 0:(A.shape[1])-1
    fsize : int
        Size of fset
    Common : dict
        Common workspace and parameters
    """
    lnz_best = None
    n = A.shape[0]
    ok = True

    # Check inputs
    if A is None or Common is None:
        return None

    Common['status'] = 'CHOLMOD_OK'
    status = 'CHOLMOD_OK'
    Common['selected'] = None
    Common['called_nd'] = False

    # Set the default strategy
    lnz_best = None
    skip_best = False
    nmethods = min(Common['nmethods'], 'CHOLMOD_MAXMETHODS')
    nmethods = max(0, nmethods)

    default_strategy = (nmethods == 0)
    if default_strategy:
        Common['method'][0]['ordering'] = 'CHOLMOD_GIVEN'
        Common['method'][1]['ordering'] = 'CHOLMOD_AMD'
        Common['method'][2]['ordering'] = 'CHOLMOD_METIS' if not Common['default_nesdis'] else 'CHOLMOD_NESDIS'
        amd_backup = False
        nmethods = 3 if 'NPARTITION' not in globals() else 2
    else:
        amd_backup = (nmethods > 1) or (nmethods == 1 and (Common['method'][0]['ordering'] in ['CHOLMOD_METIS', 'CHOLMOD_NESDIS', 'CHOLMOD_GIVEN'] and UserPerm is None))

    if 'NSUPERNODAL' in globals():
        Common['supernodal'] = 'CHOLMOD_SIMPLICIAL'

    # Allocate workspace
    uncol = A.shape[1] if A.getformat() == 'csc' else 0
    s = 6 * n + uncol
    if not ok:
        raise MemoryError("problem too large")

    # Allocate more workspace, and an empty simplicial symbolic factor
    L = {'Perm': np.zeros(n, dtype=int), 'ColCount': np.zeros(n, dtype=int)}
    Lparent = np.zeros(n, dtype=int)
    Perm = np.zeros(n, dtype=int)
    ColCount = np.zeros(n, dtype=int)
    if Common['status'] < 'CHOLMOD_OK':
        return None

    Lperm = L['Perm']
    Lcolcount = L['ColCount']
    Common['anz'] = None
    Parent = L['Parent']

    # Try all the requested ordering options and backup to AMD if needed
    Common['try_catch'] = True

    for method in range(nmethods + 1):
        Common['fl'] = None
        Common['lnz'] = None
        skip_analysis = False

        if method == nmethods:
            if Common['selected'] is None and amd_backup:
                ordering = 'CHOLMOD_AMD'
            else:
                break
        else:
            ordering = Common['method'][method]['ordering']
        Common['current'] = method

        if ordering == 'CHOLMOD_NATURAL':
            Perm = np.arange(n)
        elif ordering == 'CHOLMOD_GIVEN':
            if UserPerm is None:
                continue
            Perm = UserPerm.copy()


        if Common['status'] < 'CHOLMOD_OK':
            status = min(status, Common['status'])
            Common['status'] = 'CHOLMOD_OK'
            continue

        if not skip_analysis:
            analyze_ordering(A, ordering, Perm, fset, fsize, Parent, L['Post'], Lcolcount, L['First'], L['Level'], Common)

        if Common['selected'] is None or Common['lnz'] < lnz_best:
            Common['selected'] = method
            L['ordering'] = ordering
            lnz_best = Common['lnz']
            Lperm[:] = Perm
            if not skip_analysis:
                Lcolcount[:] = ColCount
                Lparent[:] = Parent

        if default_strategy and ordering == 'CHOLMOD_AMD':
            if (Common['fl'] < 500 * Common['lnz']) or (Common['lnz'] < 5 * Common['anz']):
                break

    Common['try_catch'] = False

    if Common['selected'] is None:
        status = 'CHOLMOD_INVALID' if status >= 'CHOLMOD_OK' else status
        raise RuntimeError("all methods failed")

    Common['fl'] = Common['method'][Common['selected']]['fl']
    Common['lnz'] = Common['method'][Common['selected']]['lnz']

    if skip_best:
        analyze_ordering(A, L['ordering'], Lperm, fset, fsize, Lparent, L['Post'], Lcolcount, L['First'], L['Level'], Common)

    if Common['postorder']:
        # Call to postorder function
        pass

    if 'NSUPERNODAL' not in globals() and (Common['supernodal'] > 'CHOLMOD_AUTO' or (Common['supernodal'] == 'CHOLMOD_AUTO' and Common['lnz'] > 0 and (Common['fl'] / Common['lnz']) >= Common['supernodal_switch'])):
        # Call to super_symbolic2 function
        pass

    return L


def detect_supernodes(L, relaxation):
    """
    Detect supernodes in the elimination by
    comparing column patterns to be the same approximately.
    Parameters
    ----------
    etree
    L
    relaxation

    Returns
    -------

    """
    n = L.shape[0]
    supernodes, cur_sn = [], []
    for i in range(n-1):
        col_i_p = L.indices[L.indptr[i]:L.indptr[i+1]]
        col_ipone_p = L.indices[L.indptr[i+1]:L.indptr[i+2]]
        # union of two sets
        union = set(col_i_p).union(set(col_ipone_p))
        intersect = set(col_i_p).intersection(set(col_ipone_p))
        # if len(union) < len(col_i_p) + len(col_ipone_p) - 1 - relaxation:
        if len(union) - len(intersect) <= relaxation + 1:
            cur_sn.append(i)
        else:
            if len(cur_sn) > 0:
                cur_sn.append(i)
                supernodes.append(cur_sn)
                cur_sn = []
            else:
                supernodes.append([i])
    if len(cur_sn) > 0:
        cur_sn.append(n-1)
        supernodes.append(cur_sn)
    else:
        supernodes.append([n-1])
    assert len(set([item for sublist in supernodes for item in sublist])) == n
    return supernodes


def convert_supernodal_to_csc(Ls):
    Lx = Ls['data']
    Lp = Ls['indptr']
    Li = Ls['indices']
    Lip = Ls['indices_ptr']
    n = Ls['rows']
    col_to_super = Ls['col_to_super']
    nnz_l = np.sum(Lx != 0)
    new_x = np.zeros(nnz_l, dtype=Lx.dtype)
    new_i = np.zeros(nnz_l, dtype=int)
    new_p = np.zeros(n+1, dtype=int)
    new_p[0] = 0
    for i in range(n):
        col_cnt = new_p[i]
        super, cnt = col_to_super[i], 0
        for j in range(Lp[i], Lp[i+1]):
            row_id = Li[Lip[super] + cnt]
            cnt+=1
            if Lx[j] != 0:
                new_x[col_cnt] = Lx[j]
                new_i[col_cnt] = row_id
                col_cnt += 1
        new_p[i+1] = col_cnt
    return {'data': new_x, 'indices': new_i, 'indptr': new_p, 'rows': n, 'cols': n}


def convert_csc_to_supernodal(L, relaxation=0):
    int_type = np.int32
    Ls = {}
    n = L.shape[0]
    t, max_width = {}, 0
    t['step 1'] = time.time()
    supernodes = detect_supernodes(L, relaxation)
    t['step 1'] = time.time() - t['step 1']
    t['step 2'] = time.time()
    row_patterns, Li, Lip, Lp = [], [], [], []
    Lip.append(0)
    Lp.append(0)
    for sn in supernodes:
        max_width = max(max_width, len(sn))
        rp_union = set()
        for i in sn:
            rp_union = rp_union.union(set(L.indices[L.indptr[i]:L.indptr[i+1]]))
        for i in sn:
            Lp.append(Lp[-1] + len(rp_union))
        row_patterns.append(rp_union)
        Li.extend(sorted(rp_union))
        Lip.append(len(Li))
    t['step 2'] = time.time() - t['step 2']
    t['step 3'] = time.time()
    # find the begining of each supernode
    super_begin = np.array((len(supernodes)+1) * [0], dtype=int)
    for i, sn in enumerate(supernodes):
        super_begin[i] = sn[0]
    super_begin[-1] = n
    col_to_super = np.zeros(n, dtype=int_type)
    for i, sn in enumerate(supernodes):
        for j in sn:
            col_to_super[j] = i
    lnnz = Lp[-1]
    data = np.zeros(lnnz, dtype=L.data.dtype)
    lookup_table = np.zeros(n, dtype=int_type)
    for i in range(len(supernodes)):
        for m in range(Lip[i], Lip[i + 1]):
            lookup_table[Li[m]] = m - Lip[i]
        for j in range(super_begin[i], super_begin[i+1]):
            for k in range(L.indptr[j], L.indptr[j+1]):
                row_idx= L.indices[k]
                data[Lp[j] + lookup_table[row_idx]] = L.data[k]

                # for m in range(Lip[i], Lip[i + 1]):
                #     lookup_table[Li[m]] = m - Lip[i]
                # for m in range(Lip[i], Lip[i + 1]):
                #     if row_idx == Li[m]:
                #         data[Lp[j] + lookup_table[row_idx]] = L.data[k]

                # for m in range(Lip[i], Lip[i+1]):
                #     if row_idx == Li[m]:
                #         offset = m - Lip[i]
                #         data[Lp[j] + offset] = L.data[k]

    t['step 3'] = time.time() - t['step 3']
    print(t)

    Ls['x'] = data
    Ls['i'] = np.array(Li, dtype=int_type) # atual row indices
    Ls['ip'] = np.array(Lip, dtype=int_type) # row pointer for each supernode
    Ls['p'] = np.array(Lp, dtype=int_type) # column pointer for each column
    Ls['supernodes'] = np.array(super_begin, dtype=int_type)
    Ls['col_to_super'] = col_to_super
    Ls['rows'] = n
    Ls['cols'] = n
    Ls['nSuper'] = len(supernodes)
    Ls['max_width'] = max_width
    if False:
        L_test = convert_supernodal_to_csc(Ls)
        assert np.all(L_test['data'] == L.data)
    return Ls


def get_supernodal_dag(Ls):
    sup_begin = Ls['supernodes']
    col_to_super = Ls['col_to_super']
    Lip, Li, Lp = Ls['ip'], Ls['i'], Ls['p']
    m = Ls['nSuper']
    nnz = 0
    DAG = []
    for i in range(m):
        DAG.append([])
        for k in range(Lip[i], Lip[i+1]):
            row_idx = Li[k]
            dest_sup = col_to_super[row_idx]
            if dest_sup not in DAG[i]:
                DAG[i].append(dest_sup)
            nnz += 1
        # sort DAG[i]
        DAG[i] = sorted(DAG[i])
    # convert DAG to CSC
    Dp = np.zeros(m + 1, dtype=int)
    Di = np.zeros(nnz, dtype=int)
    for i in range(0, m):
        Dp[i+1] = Dp[i] + len(DAG[i])
    for i in range(0, m):
        Di[Dp[i]:Dp[i+1]] = DAG[i]
    # convert it scipy csc
    L_d = sp.csc_matrix((np.ones(nnz), Di, Dp), shape=(m, m))
    return L_d


import numpy as np

def cs_mark(w, k):
    """Marks or unmarks a node in the workspace array."""
    w[k] = -w[k]-2  # Toggle the bit at index k

def cs_marked(w, k):
    """Checks if a node is marked."""
    return w[k] < 0

def ereach(n, Ap, Ai, k, parent, s, w):
    """
    Computes the elimination tree reach (ereach) for a given column k.

    Args:
        n (int): Size of the matrix.
        Ap (np.ndarray): Column pointers for the matrix A.
        Ai (np.ndarray): Row indices for the matrix A.
        k (int): Column index.
        parent (np.ndarray): Parent array representing the elimination tree.
        s (np.ndarray): Stack array.
        w (np.ndarray): Workspace array for marking nodes.

    Returns:
        int: Top index of the stack s, or -1 if inputs are invalid.
    """
    if Ap is None or Ai is None or parent is None or s is None or w is None:
        return -1

    top = n
    cs_mark(w, k)  # mark node k as visited

    for p in range(Ap[k], Ap[k + 1]):
        i = Ai[p]  # A(i,k) is nonzero
        if i > k:
            continue  # only use upper triangular part of A

        len_ = 0
        while not cs_marked(w, i):  # traverse up etree
            s[len_] = i  # L(k,i) is nonzero
            cs_mark(w, i)  # mark i as visited
            i = parent[i]
            len_ += 1
            assert i>=0, "Parent array is invalid"

        while len_ > 0:
            top -= 1
            len_ -= 1
            s[top] = s[len_]  # push path onto stack

    for p in range(top, n):
        cs_mark(w, s[p])  # unmark all nodes

    cs_mark(w, k)  # unmark node k
    return top  # s [top..n-1] contains pattern of L(k,:)

def pruning(n, c, r, ET, prunePtr, pruneSet):
    """
    Specifies the row patterns of L and pruneSet.

    Args:
        n (int): Size of the matrix.
        c (np.ndarray): Column pointers for the matrix.
        r (np.ndarray): Row indices for the matrix.
        ET (np.ndarray): Elimination tree (parent array).
        prunePtr (np.ndarray): Prune pointers (modified in place).
        pruneSet (np.ndarray): Prune set (modified in place).
    """


    prunePtr[0] = 0
    prunePtr[1] = 0

    for colNo in range(1, n):
        xi = np.zeros(2 * n, dtype=int)
        top = ereach(n, c, r, colNo, ET, xi[0:n], xi[n:])
        prunePtr[colNo + 1] = prunePtr[colNo] + (n - top)
        for i, cnt in zip(range(top, n), range(n - top)):
            pruneSet[prunePtr[colNo] + cnt] = xi[i]




