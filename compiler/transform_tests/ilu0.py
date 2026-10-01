from compiler.decorator import parallel


function_props_csr = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
}


@parallel(loop="i", function_props=function_props_csr, execution_mode="wavefront")
def ilu0_csr_row_dependencies(Ap, Ai, row_done, n):
    for i in range(n):
        row_accum = 0.0
        scale = 0.5 / (Ap[i + 1] - Ap[i] + 1)
        for k_ptr in range(Ap[i], Ap[i + 1]):
            if Ai[k_ptr] < i:
                row_accum += scale * row_done[Ai[k_ptr]]
        row_done[i] = row_accum + 1.0
    return row_done


@parallel(loop="i", function_props=function_props_csr, execution_mode="sync_free")
def ilu0_csr_row_dependencies_sync_free(Ap, Ai, row_done, n):
    for i in range(n):
        row_accum = 0.0
        scale = 0.5 / (Ap[i + 1] - Ap[i] + 1)
        for k_ptr in range(Ap[i], Ap[i + 1]):
            if Ai[k_ptr] < i:
                row_accum += scale * row_done[Ai[k_ptr]]
        row_done[i] = row_accum + 1.0
    return row_done


def ilu0_csr_reference(Ap, Ai, Ax, n):
    for i in range(n):
        row_start = Ap[i]
        row_end = Ap[i + 1]
        for k_ptr in range(row_start, row_end):
            k = Ai[k_ptr]
            if k < i:
                diag_val = 0.0
                for diag_ptr in range(Ap[k], Ap[k + 1]):
                    if Ai[diag_ptr] == k:
                        diag_val = Ax[diag_ptr]
                if diag_val != 0.0:
                    Ax[k_ptr] = Ax[k_ptr] / diag_val
                    l_ik = Ax[k_ptr]
                    for j_ptr in range(k_ptr + 1, row_end):
                        j = Ai[j_ptr]
                        for kj_ptr in range(Ap[k], Ap[k + 1]):
                            if Ai[kj_ptr] == j:
                                u_kj = Ax[kj_ptr]
                                Ax[j_ptr] = Ax[j_ptr] - l_ik * u_kj
    return Ax
