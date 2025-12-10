import numpy as np
import scipy.io as sio
import os

# compute wavefronts of sparse matrix L (CSC)
# For lower triangular: forward substitution (process columns 0 -> n-1)
# For upper triangular: backward substitution (process columns n-1 -> 0)
def compute_wavefront(L, upper_triangular=False):
    wavefront_list = []
    
    if not upper_triangular:
        # Lower triangular: original algorithm
        incoming_edges = np.zeros(L.shape[0], dtype=int)
        # compute incoming edges
        for i in range(L.shape[0]):
            for j in range(L.indptr[i], L.indptr[i + 1]):
                incoming_edges[L.indices[j]] += 1
        # compute wavefronts
        wavefront = np.where(incoming_edges == 1)[0]
        while wavefront.size > 0:
            wavefront_list.append(wavefront)
            new_wavefront = np.array([], dtype=int)
            for i in wavefront:
                for j in range(L.indptr[i]+1, L.indptr[i + 1]):
                    incoming_edges[L.indices[j]] -= 1
                    if incoming_edges[L.indices[j]] == 1:
                        new_wavefront = np.append(new_wavefront, L.indices[j])
            wavefront = new_wavefront
    else:
        # Upper triangular: count outgoing dependencies (columns that this column depends on)
        outgoing_edges = np.zeros(L.shape[0], dtype=int)
        # For column i, count how many columns j > i where U[i,j] != 0
        # Column i depends on columns j > i where U[i,j] != 0
        # In CSC upper triangular, column j contains U[row,j] where row <= j
        # So for column i, we need to count: in how many columns j > i does row i appear?
        for j in range(L.shape[0]):  # for each column j
            for idx in range(L.indptr[j], L.indptr[j + 1]):
                row_i = L.indices[idx]
                if row_i < j:  # U[row_i, j] != 0 where row_i < j
                    outgoing_edges[row_i] += 1
        
        # Start with columns that have no outgoing dependencies
        wavefront = np.where(outgoing_edges == 0)[0]
        processed = np.zeros(L.shape[0], dtype=bool)
        
        while wavefront.size > 0:
            wavefront_list.append(wavefront)
            processed[wavefront] = True
            new_wavefront = []
            
            # For each processed column j, decrement dependencies of columns i < j that depend on j
            for j in wavefront:
                # Check column j: which rows i < j appear in it (meaning U[i,j] != 0)?
                for idx in range(L.indptr[j], L.indptr[j + 1]):
                    i = L.indices[idx]
                    if i < j:  # U[i,j] != 0, so column i depends on column j
                        outgoing_edges[i] -= 1
                        if outgoing_edges[i] == 0 and not processed[i]:
                            new_wavefront.append(i)
                            processed[i] = True
            
            wavefront = np.array(new_wavefront, dtype=int)
    
    return wavefront_list


def compact_2d_list(wavefront_list):
    # convert wavefront list to 1D array
    waves = [element for sublist in wavefront_list for element in sublist]
    waves = np.array(waves, dtype=np.int32)
    wave_start_loc = np.zeros(len(wavefront_list) + 1, dtype=np.int32)
    wave_start_loc[0] = 0
    for i in range(1, len(wavefront_list)+1):
        wave_start_loc[i] = wave_start_loc[i - 1] + len(wavefront_list[i - 1])
    return wave_start_loc, waves


# count incoming edges of each node
# For CSC format:
# - Lower triangular (forward sub): column i depends on columns j < i where L[i,j] != 0
# - Upper triangular (backward sub): column i depends on columns j > i where L[i,j] != 0
def incoming_edge(L, upper_triangular=False):
    incoming_edges = np.zeros(L.shape[0], dtype=np.int32)
    
    if not upper_triangular:
        # Lower triangular: original algorithm
        # Column i receives updates from columns < i
        for i in range(L.shape[0]):
            for j in range(L.indptr[i], L.indptr[i + 1]):
                incoming_edges[L.indices[j]] += 1
    else:
        # Upper triangular: count dependencies from columns > i
        # For column i, count how many columns j > i contain row i
        # (meaning A[i,j] != 0, so column i depends on column j)
        for j in range(L.shape[0]):  # for each column j
            for idx in range(L.indptr[j], L.indptr[j + 1]):
                row_i = L.indices[idx]
                if row_i < j:  # A[row_i, j] where row_i < j (upper triangular off-diagonal)
                    incoming_edges[row_i] += 1
    
    return incoming_edges


def incoming_edge_csr(L, upper_triangular=False):
    # Build map of column -> rows that contain that column
    # outgoing_edges[i] = number of columns in row i
    # For sync-free algorithms: outgoing_edges is the initial dependency count
    incoming_edges = [[] for _ in range(L.shape[1])]
    outgoing_edges = np.zeros(L.shape[0], dtype=np.int32)
    
    for i in range(L.shape[0]):
        for j in range(L.indptr[i], L.indptr[i + 1]):
            incoming_edges[L.indices[j]].append(i)
        # For dependency counting:
        # Lower triangular: row i depends on off-diagonal elements (exclude last element - diagonal)
        # Upper triangular: row i depends on off-diagonal elements (exclude last element - diagonal)
        # In both cases, subtract 1 for diagonal
        row_len = L.indptr[i + 1] - L.indptr[i]
        if row_len > 0:
            outgoing_edges[i] = row_len - 1  # Exclude diagonal
        else:
            outgoing_edges[i] = 0
    
    map_ptr = np.zeros(L.shape[1] + 1, dtype=np.int32)
    map_vals = np.zeros(L.nnz, dtype=np.int32)
    for i in range(L.shape[1]):
        map_ptr[i + 1] = map_ptr[i] + len(incoming_edges[i])
        map_vals[map_ptr[i]:map_ptr[i + 1]] = incoming_edges[i]
    return map_ptr, map_vals, outgoing_edges


def get_matrix_list(file_path, base_path):
    with open(file_path, 'r') as f:
        matrix_list = f.readlines()
    matrix_list = [x.strip() for x in matrix_list]
    matrix_list = [base_path + x for x in matrix_list]
    # remove bundle1.mtx from the list
    matrix_list = [x for x in matrix_list if "bundle1.mtx" not in x]
    # sort matrix_list by nnz
    nnz_list, new_matrixlist = [], []
    for matrix in matrix_list:
        A = sio.mmread(matrix)
        if A.data.dtype != np.float64:
            print(f"Matrix {matrix} is not double precision")
            A.data = A.data.astype(np.float64)
        nnz_list.append(A.nnz)
    matrix_list = [x for _, x in sorted(zip(nnz_list, matrix_list))]
    return matrix_list


def get_matrix_signature(A, name):
    dict = {}
    dict['name'] = name
    dict['n'] = A.shape[0]
    dict['nnz'] = A.nnz
    dict['nnz_per_row'] = A.nnz / A.shape[0]
    dict['nnz_per_col'] = A.nnz / A.shape[1]
    dict['max nnz per row'] = max(A.getnnz(axis=1))
    dict['max nnz per col'] = max(A.getnnz(axis=0))
    dict['density'] = A.nnz / (A.shape[0] * A.shape[1])
    dict["wavefronts"] = len(compute_wavefront(A))
    return dict

def get_L_signature(L, Ls, name):
    dict = {}
    dict['lname'] = name
    dict['lnnz'] = L.nnz
    dict['lnnz_per_row'] = L.nnz / L.shape[0]
    dict['lnnz_per_col'] = L.nnz / L.shape[1]
    dict['max lnnz per row'] = max(L.getnnz(axis=1))
    dict['max lnnz per col'] = max(L.getnnz(axis=0))
    dict['ldensity'] = L.nnz / (L.shape[0] * L.shape[1])
    dict["lwavefronts"] = len(compute_wavefront(L))
    dict['num supernodes'] = len(Ls['supernodes']) - 1
    dict['max supernode width'] = Ls['max_width']
    dict['average supernode width'] = L.shape[0] / (len(Ls['supernodes']) - 1)
    return dict


import scipy.sparse as sp

def convert_csr_matrix_to_power2_dim(A):
    m, n = A.shape
    m_new = int(2 ** np.ceil(np.log2(m)))
    n_new = int(2 ** np.ceil(np.log2(n)))
    new_Aptr = np.zeros(m_new + 1, dtype=np.int32)
    new_Aptr[:m+1] = A.indptr
    new_Aptr[m+1:] = A.indptr[-1]
    new_Adata = A.data
    new_Aindices = A.indices
    # new csr matrix
    new_A = sp.csr_matrix((new_Adata, new_Aindices, new_Aptr),
                          shape=(int(m_new), int(n_new)))
    return new_A

