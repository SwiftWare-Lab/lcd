"""CSR and CSC sparse triangular-solve decorator examples."""

import argparse

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, tril
from scipy.sparse.linalg import spsolve_triangular

from compiler.decorator import parallel
from compiler.demo.matrix import DEFAULT_MATRIX, DEFAULT_MAX_N, load_matrix_market


FUNCTION_PROPS_CSR = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
}

FUNCTION_PROPS_CSC = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
    ("Ap", "Ai"): ["triangularity"],
}


@parallel(loop="i", function_props=FUNCTION_PROPS_CSR, execution_mode="wavefront", threads_per_block=128)
def simple_sptrsv_csr(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(Ap[i], Ap[i + 1] - 1):
            x[i] -= Ax[j] * x[Ai[j]]
        x[i] /= Ax[Ap[i + 1] - 1]
    return x


@parallel(loop="i", function_props=FUNCTION_PROPS_CSC, execution_mode="wavefront", threads_per_block=128)
def simple_sptrsv_csc(Ap, Ai, Ax, b, n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - x[i]) / Ax[Ap[i]]
        for j in range(Ap[i] + 1, Ap[i + 1]):
            x[Ai[j]] += Ax[j] * x[i]
    return x


def run_demo(matrix_path=DEFAULT_MATRIX, max_size=DEFAULT_MAX_N, storage="both"):
    """Execute the generated SpTRSV solver on a Matrix Market input."""
    source = load_matrix_market(matrix_path, max_size)
    lower = tril(source, format="csr")
    lower.sort_indices()
    expected = np.linspace(1.0, 2.0, lower.shape[0])
    right_hand_side = np.asarray(lower @ expected).ravel()
    solvers = {
        "csr": (csr_matrix, simple_sptrsv_csr),
        "csc": (csc_matrix, simple_sptrsv_csc),
    }
    selected = solvers if storage == "both" else {storage: solvers[storage]}
    results = {}
    for name, (matrix_type, solver) in selected.items():
        matrix = matrix_type(lower)
        matrix.sort_indices()
        result = solver(
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data.copy(),
            right_hand_side,
            matrix.shape[0],
        )
        reference = spsolve_triangular(
            matrix,
            right_hand_side,
            lower=True,
        )
        error = float(np.max(np.abs(result - reference)))
        results[name] = result
        print(
            f"SpTRSV {name.upper()}: n={matrix.shape[0]}, "
            f"nnz={matrix.nnz}, max_abs_error={error:.3e}"
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run generated CSR/CSC SpTRSV on a Matrix Market input."
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
    args = parser.parse_args(argv)
    if args.max_n < 0:
        parser.error("--max-n must be nonnegative")
    run_demo(args.matrix, args.max_n, args.storage)


if __name__ == "__main__":
    main()

