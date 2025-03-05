
#

import numpy as np
import scipy.sparse as sp

def etree(A):
    n = A.shape[0]
    parent = np.full(n, -1, dtype=int)
    ancestor = np.full(n, -1, dtype=int)
    for k in range(n):
        for i in range(A.indptr[k], A.indptr[k+1]):
            j = A.indices[i]
            while j != -1 and j < k:
                next_j = ancestor[j]
                ancestor[j] = k
                if next_j == -1:
                    parent[j] = k
                j = next_j
    return parent


def cs_leaf(i, j, first, maxfirst, prevleaf, ancestor):
    """
    Consider A(i,j), node j in ith row subtree and return lca(jprev,j)
    """
    if first is None or maxfirst is None or prevleaf is None or ancestor is None:
        return -1, 0

    jleaf = 0
    if i <= j or first[j] <= maxfirst[i]:
        return -1, jleaf  # j not a leaf

    maxfirst[i] = first[j]  # update max first[j] seen so far
    jprev = prevleaf[i]  # jprev = previous leaf of ith subtree
    prevleaf[i] = j
    jleaf = 1 if jprev == -1 else 2  # j is first or subsequent leaf

    if jleaf == 1:
        return i, jleaf  # if 1st leaf, q = root of ith subtree

    q = jprev
    while q != ancestor[q]:
        q = ancestor[q]

    s = jprev
    while s != q:
        sparent = ancestor[s]  # path compression
        ancestor[s] = q
        s = sparent

    return q, jleaf  # q = least common ancestor (jprev, j)


def post_order(parent):
    n = len(parent)
    post = np.full(n, -1, dtype=int)
    stack = np.zeros(n, dtype=int)
    k = 0
    head = np.full(n, -1, dtype=int)
    next = np.full(n, -1, dtype=int)

    for j in range(n-1, -1, -1):
        if parent[j] == -1:
            continue
        next[j] = head[parent[j]]
        head[parent[j]] = j

    for j in range(n):
        if parent[j] != -1:
            continue
        stack[0] = j
        top = 0
        while top >= 0:
            j = stack[top]
            if j >= 0:
                stack[top] = -j - 1
                top += 1
                for i in range(head[j], -1, -1):
                    if post[i] == -1:
                        stack[top] = i
            else:
                j = -j - 1
                if post[j] == -1:
                    post[j] = k
                    k += 1
                    top -= 1
    return post


def column_count(A, parent, post, ata=False):
    n = A.shape[0]
    col_count = np.zeros(n, dtype=int)
    delta = np.zeros(n, dtype=int)
    first = np.full(n, -1, dtype=int)
    ancestor = np.arange(n, dtype=int)
    maxfirst = np.full(n, -1, dtype=int)
    prevleaf = np.full(n, -1, dtype=int)
    head = np.full(n, -1, dtype=int) if ata else None
    next = np.full(n, -1, dtype=int) if ata else None

    for k in range(n):
        j = post[k]
        delta[j] = 1 if first[j] == -1 else 0
        while j != -1 and first[j] == -1:
            first[j] = k
            j = parent[j]

    if ata:
        for i in range(A.shape[1]):
            k = n
            for p in range(A.indptr[i], A.indptr[i + 1]):
                k = min(k, post[A.indices[p]])
            next[i] = head[k]
            head[k] = i

    for k in range(n):
        j = post[k]
        if parent[j] != -1:
            delta[parent[j]] -= 1
        J = head[k] if ata else j
        while J != -1:
            for p in range(A.indptr[J], A.indptr[J + 1]):
                i = A.indices[p]
                q, jleaf = cs_leaf(i, j, first, maxfirst, prevleaf,
                                   ancestor)
                if jleaf >= 1:
                    delta[j] += 1
                if jleaf == 2:
                    delta[q] -= 1
            J = next[J] if ata else -1
        if parent[j] != -1:
            ancestor[j] = parent[j]

    for j in range(n):
        col_count[j] = delta[j]
        if parent[j] != -1:
            col_count[parent[j]] += col_count[j]

    return col_count


def row_pattern(A, parent):
    n = A.shape[0]
    row_pattern = [set() for _ in range(n)]
    for j in range(n):
        row_pattern[j].add(j)
        for i in range(A.indptr[j], A.indptr[j+1]):
            k = A.indices[i]
            while k != -1 and k < j:
                row_pattern[j].add(k)
                k = parent[k]
    return row_pattern


def compute_supernodes(A, parent, col_count):
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    supernodes = []
    for j in range(n):
        if visited[j]:
            continue
        supernode = [j]
        visited[j] = True
        k = parent[j]
        while k != -1 and not visited[k]:
            supernode.append(k)
            visited[k] = True
            k = parent[k]
        supernodes.append(supernode)
    return supernodes



def symbolic_cholesky(A):
    parent = etree(A)
    col_count = column_count(A, parent)
    ro_pattern = row_pattern(A, parent)
    return parent, col_count, ro_pattern




