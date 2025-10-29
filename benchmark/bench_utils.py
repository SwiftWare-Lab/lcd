import numpy as np
import scipy.io as sio
import os

# compute wavefronts of sparse matrix L (CSC)
def compute_wavefront(L):
    wavefront_list = []
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
def incoming_edge(L):
    incoming_edges = np.zeros(L.shape[0], dtype=np.int32)
    # compute incoming edges
    for i in range(L.shape[0]):
        for j in range(L.indptr[i], L.indptr[i + 1]): # excluding diagonal
            incoming_edges[L.indices[j]] += 1
    return incoming_edges


def incoming_edge_csr(L):
    incoming_edges = [[] for _ in range(L.shape[1])]
    outgoing_edges = np.zeros(L.shape[0], dtype=np.int32)
    for i  in range(L.shape[0]):
        for j in range(L.indptr[i], L.indptr[i + 1]):
            incoming_edges[L.indices[j]].append(i)
        outgoing_edges[i] = L.indptr[i + 1] - L.indptr[i]
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
    dict['num supernodes'] = len(Ls['supernodes'])
    dict['max supernode width'] = Ls['max_width']
    dict['average supernode width'] = L.shape[0] / len(Ls['supernodes'])
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

