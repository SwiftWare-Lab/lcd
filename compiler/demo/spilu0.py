"""CSR and CSC ILU0 inputs for the complete decorator pipeline."""

import argparse

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from compiler.decorator import parallel
from compiler.demo.matrix import DEFAULT_MATRIX, DEFAULT_MAX_N, load_matrix_market


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


def compile_ilu0_csr(
        execution_mode="wavefront",
        threads_per_block=128,
        thread_unroll=None,
):
    """Compile CSR ILU0, scheduling unrolling unless it is explicitly set."""
    return parallel(
        block_map="i",
        thread_map="j_ptr",
        function_props=FUNCTION_PROPS,
        execution_mode=execution_mode,
        threads_per_block=threads_per_block,
        thread_unroll=thread_unroll,
    )(ilu0_csr)


def compile_ilu0_csc(execution_mode="wavefront", threads_per_block=128):
    """Compile the CSC example with the requested executor settings."""
    return parallel(
        block_map="k",
        function_props=FUNCTION_PROPS,
        execution_mode=execution_mode,
        threads_per_block=threads_per_block,
    )(ilu0_csc)


def _ilu0_csr_reference(data, indices, indptr):
    return ilu0_csr(data.copy(), indices, indptr)


def _ilu0_csc_reference(data, indices, indptr):
    return ilu0_csc(data.copy(), indices, indptr)


def run_demo(
        matrix_path=DEFAULT_MATRIX,
        max_size=DEFAULT_MAX_N,
        storage="both",
        execution_mode="wavefront",
        threads_per_block=128,
        thread_unroll=None,
):
    """Execute generated ILU0 variants on a Matrix Market input."""
    source = load_matrix_market(matrix_path, max_size)
    matrix_types = {"csr": csr_matrix, "csc": csc_matrix}
    selected = (
        matrix_types
        if storage == "both"
        else {storage: matrix_types[storage]}
    )
    results = {}
    for name, matrix_type in selected.items():
        matrix = matrix_type(source)
        matrix.sort_indices()
        indptr = matrix.indptr.astype(np.int32)
        indices = matrix.indices.astype(np.int32)
        data = matrix.data.copy()
        if name == "csr":
            factor = compile_ilu0_csr(
                execution_mode,
                threads_per_block,
                thread_unroll,
            )
            reference = _ilu0_csr_reference(data, indices, indptr)
        else:
            factor = compile_ilu0_csc(execution_mode, threads_per_block)
            reference = _ilu0_csc_reference(data, indices, indptr)
        result = factor(data.copy(), indices, indptr)
        error = float(np.max(np.abs(result - reference)))
        results[name] = result
        schedule = factor.current_schedule
        print(
            f"ILU0 {name.upper()}: n={matrix.shape[0]}, nnz={matrix.nnz}, "
            f"mode={schedule.execution_mode}, "
            f"threads={schedule.threads_per_block}, "
            f"unroll={schedule.thread_unroll}, "
            f"max_abs_error={error:.3e}"
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run generated CSR/CSC ILU0 on a Matrix Market input."
    )
    parser.add_argument(
        "matrix",
        nargs="?",
        default=DEFAULT_MATRIX,
        help="path to a square real Matrix Market file",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="leading principal sample size; use 0 for the full matrix",
    )
    parser.add_argument(
        "--storage",
        choices=("csr", "csc", "both"),
        default="both",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("wavefront", "sync_free"),
        default="wavefront",
    )
    parser.add_argument("--threads-per-block", type=int, default=128)
    parser.add_argument(
        "--thread-unroll",
        type=int,
        choices=(1, 2, 4, 8),
        help="CSR j_ptr unroll factor; omit to let the scheduler choose",
    )
    args = parser.parse_args(argv)
    if args.max_n < 0:
        parser.error("--max-n must be nonnegative")
    if args.threads_per_block <= 0:
        parser.error("--threads-per-block must be positive")
    run_demo(
        args.matrix,
        args.max_n,
        args.storage,
        args.execution_mode,
        args.threads_per_block,
        args.thread_unroll,
    )


if __name__ == "__main__":
    main()
