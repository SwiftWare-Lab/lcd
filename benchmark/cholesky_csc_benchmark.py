"""Portable sparse Cholesky benchmark with separate symbolic/numeric timings."""

import argparse
import csv
import json
from math import isqrt
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numba
import numpy as np
import scipy
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, spsolve_triangular
from threadpoolctl import threadpool_info, threadpool_limits

from benchmark.cholesky_csc import _factor_prepared, cholesky_csc
from utils.cholesky_symbolic import _analyze_prepared, _prepare_csc_input


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def generated_matrix(family, n, rng):
    """Generate SPD matrices with very different natural-order fill behavior."""
    if family == "tridiagonal":
        return sp.diags((-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)),
                        (-1, 0, 1), shape=(n, n), format="csc")
    if family == "grid":
        side = isqrt(n)
        t = sp.diags((-np.ones(side - 1), 2 * np.ones(side), -np.ones(side - 1)),
                     (-1, 0, 1), shape=(side, side), format="csc")
        return (sp.kron(sp.eye(side), t) + sp.kron(t, sp.eye(side))).tocsc()
    edges = sp.random(n, n, density=min(8. / n, 1.), format="csc", random_state=rng,
                      data_rvs=lambda count: rng.uniform(-1., 1., count))
    edges = sp.tril(edges, k=-1, format="csc")
    a = edges + edges.T
    return (a + sp.diags(np.asarray(abs(a).sum(axis=1)).ravel() + 1.)).tocsc()


def benchmark_case(a, repeats, rng):
    lower_input = _prepare_csc_input(a)
    symbolic = _analyze_prepared(lower_input)
    lower = _factor_prepared(lower_input, symbolic)  # Warm compilation outside timing.
    canonical = lower_input + sp.tril(lower_input, k=-1).T
    norm_a = sp.linalg.norm(canonical)
    reconstruction = sp.linalg.norm(canonical - lower @ lower.T) / norm_a
    b = rng.standard_normal(a.shape[0])
    y = spsolve_triangular(lower, b, lower=True)
    x = spsolve_triangular(lower.T.tocsc(), y, lower=False)
    solve_residual = np.linalg.norm(canonical @ x - b) / np.linalg.norm(b)
    reference_x = spsolve(canonical.tocsc(), b)
    solve_difference = np.linalg.norm(x - reference_x) / np.linalg.norm(reference_x)
    if not all(np.isfinite(v) and v < 1e-10 for v in (reconstruction, solve_residual, solve_difference)):
        raise RuntimeError("Sparse factorization failed reconstruction or solve verification")
    # Only the benchmark's optional small-matrix reference converts to dense.
    if a.shape[0] <= 256:
        np.testing.assert_allclose(lower.toarray(), np.linalg.cholesky(canonical.toarray()),
                                   rtol=1e-10, atol=1e-12)

    stages = {
        "symbolic": lambda: _analyze_prepared(lower_input),
        "numeric": lambda: _factor_prepared(lower_input, symbolic),
        "reuse_api": lambda: cholesky_csc(a, symbolic),
        "total_api": lambda: cholesky_csc(a),
    }
    for function in stages.values():
        function()
    samples = {name: [] for name in stages}
    names = list(stages)
    for _ in range(repeats):
        for index in rng.permutation(len(names)):
            name = names[index]
            function = stages[name]
            start = perf_counter()
            result = function()
            samples[name].append((perf_counter() - start) * 1000)
            del result
    row = {
        "n": a.shape[0], "nnz_a_lower": lower_input.nnz, "nnz_l": symbolic.nnz,
        "fill_ratio": symbolic.nnz / max(1, lower_input.nnz),
        "tree_roots": int(np.count_nonzero(symbolic.parent == -1)),
        "relative_reconstruction": reconstruction, "relative_solve_residual": solve_residual,
        "relative_solve_vs_scipy": solve_difference,
    }
    for name, values in samples.items():
        row[name + "_median_ms"] = median(values)
        row[name + "_min_ms"] = min(values)
        row[name + "_max_ms"] = max(values)
    return row, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=[64, 256, 1024],
                        help="Target dimensions; grid uses floor(sqrt(n)) squared")
    parser.add_argument("--families", choices=["tridiagonal", "grid", "random"], nargs="+",
                        default=["tridiagonal", "grid", "random"])
    parser.add_argument("--matrix", type=Path, nargs="+", help="Matrix Market files instead of generated cases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    matrix_rng, timing_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed)
    if args.matrix:
        cases = ((str(path.resolve()), sio.mmread(path)) for path in args.matrix)
    else:
        cases = ((f"{family}_{n}", generated_matrix(family, n, matrix_rng))
                 for family in args.families for n in args.sizes)
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "numba": numba.__version__, "scipy": scipy.__version__,
        "ordering": "natural", "seed": args.seed, "repeats": args.repeats,
        "sizes": args.sizes, "families": args.families,
        "matrix_files": [str(p.resolve()) for p in args.matrix] if args.matrix else [],
        "thread_policy": "custom symbolic and numeric kernels serial; detected reference BLAS pools limited to 1",
        "timing": {
            "symbolic": "prepared lower input; tree, pattern, dependencies, and owned arrays",
            "numeric": "prepared lower input + existing analysis; allocation, kernel, result construction",
            "reuse_api": "validation/canonicalization + pattern check + numeric",
            "total_api": "validation/canonicalization + symbolic + numeric",
            "excluded": "JIT warmup, matrix generation, reference factors, reconstruction, solves",
        },
        "samples_ms": [],
    }
    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        metadata["blas_pools"] = threadpool_info()
        print("Natural-order CSC; serial symbolic and numeric kernels; timings exclude JIT and verification.")
        print(f"{'case':<22} {'n':>6} {'nnz L':>9} {'fill':>7} {'sym ms':>10} {'num ms':>10} {'total ms':>10}")
        for name, a in cases:
            row, samples = benchmark_case(a, args.repeats, timing_rng)
            row = {"case": name, **row}
            rows.append(row)
            metadata["samples_ms"].append({"case": name, "stages": samples})
            print(f"{name:<22} {row['n']:6d} {row['nnz_l']:9d} {row['fill_ratio']:7.2f} "
                  f"{row['symbolic_median_ms']:10.4f} {row['numeric_median_ms']:10.4f} "
                  f"{row['total_api_median_ms']:10.4f}", flush=True)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.csv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
