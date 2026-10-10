"""Gauss-Seidel examples with an enclosing host-side sweep loop."""

import argparse

import numpy as np
from scipy.sparse import tril

from compiler.decorator import parallel
from compiler.demo.matrix import DEFAULT_MATRIX, DEFAULT_MAX_N, load_matrix_market


FUNCTION_PROPS_CSR = {
    "Ap": ["monotonically_increasing"],
    "Ai": [],
}


@parallel(block_map="i", function_props=FUNCTION_PROPS_CSR, execution_mode="wavefront", threads_per_block=128)
def simple_gauss_seidel_csr(Ap, Ai, Ax, D, b, x, n, max_iter):
    for sweep in range(max_iter):
        for i in range(n):
            x[i] = b[i]
            for j in range(Ap[i], Ap[i + 1]):
                x[i] -= Ax[j] * x[Ai[j]]
            x[i] /= D[i]
    return x


@parallel(block_map="i", function_props=FUNCTION_PROPS_CSR, execution_mode="wavefront", threads_per_block=128)
def full_gauss_seidel_csr(Ap, Ai, Ax, b, x, n, max_iter):
    for sweep in range(max_iter):
        for i in range(n):
            sigma = 0.0
            diag = 1.0
            for j in range(Ap[i], Ap[i + 1]):
                if Ai[j] == i:
                    diag = Ax[j]
                else:
                    sigma += Ax[j] * x[Ai[j]]
            x[i] = (b[i] - sigma) / diag
    return x


def _simple_reference(Ap, Ai, Ax, diagonal, b, x, max_iter):
    result = x.copy()
    for _ in range(max_iter):
        for i in range(len(diagonal)):
            result[i] = b[i]
            for j in range(Ap[i], Ap[i + 1]):
                result[i] -= Ax[j] * result[Ai[j]]
            result[i] /= diagonal[i]
    return result


def _full_reference(Ap, Ai, Ax, b, x, max_iter):
    result = x.copy()
    for _ in range(max_iter):
        for i in range(len(Ap) - 1):
            sigma = 0.0
            diagonal = 1.0
            for j in range(Ap[i], Ap[i + 1]):
                if Ai[j] == i:
                    diagonal = Ax[j]
                else:
                    sigma += Ax[j] * result[Ai[j]]
            result[i] = (b[i] - sigma) / diagonal
    return result


def run_demo(
        matrix_path=DEFAULT_MATRIX,
        max_size=DEFAULT_MAX_N,
        variant="both",
        iterations=2,
):
    """Execute generated Gauss-Seidel variants on a Matrix Market input."""
    source = load_matrix_market(matrix_path, max_size)
    full_lower = tril(source, format="csr")
    full_lower.sort_indices()
    strict_lower = tril(source, k=-1, format="csr")
    strict_lower.sort_indices()
    size = source.shape[0]
    expected = np.linspace(1.0, 2.0, size)
    right_hand_side = np.asarray(full_lower @ expected).ravel()
    initial = np.zeros(size, dtype=np.float64)
    results = {}

    if variant in {"simple", "both"}:
        Ap = strict_lower.indptr.astype(np.int32)
        Ai = strict_lower.indices.astype(np.int32)
        Ax = strict_lower.data.copy()
        diagonal = source.diagonal().copy()
        result = simple_gauss_seidel_csr(
            Ap,
            Ai,
            Ax,
            diagonal,
            right_hand_side,
            initial.copy(),
            size,
            iterations,
        )
        reference = _simple_reference(
            Ap, Ai, Ax, diagonal, right_hand_side, initial, iterations
        )
        error = float(np.max(np.abs(result - reference)))
        results["simple"] = result
        print(
            f"Gauss-Seidel simple CSR: n={size}, nnz={strict_lower.nnz}, "
            f"sweeps={iterations}, max_abs_error={error:.3e}"
        )

    if variant in {"full", "both"}:
        Ap = full_lower.indptr.astype(np.int32)
        Ai = full_lower.indices.astype(np.int32)
        Ax = full_lower.data.copy()
        result = full_gauss_seidel_csr(
            Ap,
            Ai,
            Ax,
            right_hand_side,
            initial.copy(),
            size,
            iterations,
        )
        reference = _full_reference(
            Ap, Ai, Ax, right_hand_side, initial, iterations
        )
        error = float(np.max(np.abs(result - reference)))
        results["full"] = result
        print(
            f"Gauss-Seidel full CSR: n={size}, nnz={full_lower.nnz}, "
            f"sweeps={iterations}, max_abs_error={error:.3e}"
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run generated Gauss-Seidel on a Matrix Market input."
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
        "--variant",
        choices=("simple", "full", "both"),
        default="both",
    )
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args(argv)
    if args.max_n < 0:
        parser.error("--max-n must be nonnegative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    run_demo(args.matrix, args.max_n, args.variant, args.iterations)


if __name__ == "__main__":
    main()

