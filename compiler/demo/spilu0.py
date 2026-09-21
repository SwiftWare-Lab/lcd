"""CSR and CSC ILU0 inputs for the complete decorator pipeline."""

from compiler.decorator import parallel


FUNCTION_PROPS = {
    "A_indptr": ["monotonically_increasing"],
    "A_indices": [],
    ("A_indptr", "A_indices"): [
        ("segmented_monotonically_increasing", "n"),
    ],
}


def ilu0_csr(A_data, A_indices, A_indptr):
    n = len(A_indptr) - 1
    for i in range(n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]
        for k_ptr in range(row_start, row_end):
            k = A_indices[k_ptr]
            if k >= i:
                break
            diag_val = 0.0
            for diag_ptr in range(A_indptr[k], A_indptr[k + 1]):
                if A_indices[diag_ptr] == k:
                    diag_val = A_data[diag_ptr]
                    break
            if diag_val != 0.0:
                A_data[k_ptr] = A_data[k_ptr] / diag_val
                l_ik = A_data[k_ptr]
                for j_ptr in range(k_ptr + 1, row_end):
                    j = A_indices[j_ptr]
                    for kj_ptr in range(A_indptr[k], A_indptr[k + 1]):
                        if A_indices[kj_ptr] == j:
                            u_kj = A_data[kj_ptr]
                            A_data[j_ptr] = (
                                A_data[j_ptr] - l_ik * u_kj
                            )
                            break
    return A_data


def ilu0_csc(A_data, A_indices, A_indptr):
    n = len(A_indptr) - 1
    for k in range(n):
        col_start = A_indptr[k]
        col_end = A_indptr[k + 1]
        diag_val = 0.0
        for dptr in range(col_start, col_end):
            if A_indices[dptr] == k:
                diag_val = A_data[dptr]
                break
        for i_ptr in range(col_start, col_end):
            i = A_indices[i_ptr]
            if i <= k:
                continue
            if diag_val != 0.0:
                A_data[i_ptr] = A_data[i_ptr] / diag_val
                l_ik = A_data[i_ptr]
                for j in range(k + 1, n):
                    j_start = A_indptr[j]
                    j_end = A_indptr[j + 1]
                    u_kj = 0.0
                    for kj_ptr in range(j_start, j_end):
                        if A_indices[kj_ptr] == k:
                            u_kj = A_data[kj_ptr]
                            break
                    if u_kj != 0.0:
                        for ij_ptr in range(j_start, j_end):
                            if A_indices[ij_ptr] == i:
                                A_data[ij_ptr] = (
                                    A_data[ij_ptr] - l_ik * u_kj
                                )
                                break
    return A_data


def compile_ilu0_csr(execution_mode="wavefront", threads_per_block=128):
    """Compile the CSR example with the requested executor settings."""
    return parallel(
        block_map="i",
        function_props=FUNCTION_PROPS,
        execution_mode=execution_mode,
        threads_per_block=threads_per_block,
    )(ilu0_csr)


def compile_ilu0_csc(execution_mode="wavefront", threads_per_block=128):
    """Compile the CSC example with the requested executor settings."""
    return parallel(
        block_map="k",
        function_props=FUNCTION_PROPS,
        execution_mode=execution_mode,
        threads_per_block=threads_per_block,
    )(ilu0_csc)

