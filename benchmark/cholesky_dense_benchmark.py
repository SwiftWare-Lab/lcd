"""Reproducible CPU benchmark: python -m benchmark.cholesky_dense_benchmark."""

import argparse
import csv
from functools import partial
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numba
import numpy as np
import scipy
from scipy.linalg import cholesky
from threadpoolctl import threadpool_info, threadpool_limits

from benchmark.cholesky_dense import _cholesky_dense_kernel, cholesky_dense_numba
from benchmark.cholesky_dense_optimized import (
    DEFAULT_BLOCK_SIZE, _cholesky_dense_blocked_kernel, cholesky_dense_numba_optimized,
)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def measure(implementations, a, repeats, order_rng):
    """Warm/verify all methods, then interleave their timed calls in seeded order."""
    residuals, samples = {}, {name: [] for name in implementations}
    reference = np.linalg.cholesky(a)
    for name, function in implementations.items():
        lower = function(a)
        np.testing.assert_allclose(lower, reference, rtol=1e-11, atol=1e-12)
        # Independent scalar contraction avoids relying on reference BLAS here.
        reconstructed = np.einsum("ik,jk->ij", lower, lower, optimize=False)
        residual = np.linalg.norm(a - reconstructed) / np.linalg.norm(a)
        if not np.isfinite(residual) or residual > 1e-12:
            raise RuntimeError(f"{name} failed verification: residual={residual}")
        residuals[name] = residual
    names = list(implementations)
    for _ in range(repeats):
        for index in order_rng.permutation(len(names)):
            name = names[index]
            function = implementations[name]
            start = perf_counter()
            result = function(a)
            samples[name].append((perf_counter() - start) * 1000)
            del result
    return samples, residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=[32, 128, 256, 512, 1024])
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-sizes", type=positive_int, nargs="+", default=[DEFAULT_BLOCK_SIZE],
                        help="Optimized panel/tile widths to compare on the same matrices")
    parser.add_argument("--csv", type=Path, help="Also save results and a .metadata.json sidecar")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    order_rng = np.random.default_rng(args.seed)
    implementations = {
        "numba_kernel": _cholesky_dense_kernel,
        "numba_validated": cholesky_dense_numba,
        "numpy": np.linalg.cholesky,
        "scipy": lambda a: cholesky(a, lower=True, check_finite=False),
    }
    blocks = {}
    for block_size in dict.fromkeys(args.block_sizes):
        kernel_name = f"optimized_b{block_size}_kernel"
        api_name = f"optimized_b{block_size}_validated"
        implementations[kernel_name] = partial(_cholesky_dense_blocked_kernel, block_size=block_size)
        implementations[api_name] = partial(cholesky_dense_numba_optimized, block_size=block_size)
        blocks[kernel_name] = blocks[api_name] = block_size
    rows = []
    build_info = {"numpy": np.show_config(mode="dicts"),
                  "scipy": scipy.show_config(mode="dicts")}
    with threadpool_limits(limits=1, user_api="blas"):
        metadata = {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "seed": args.seed,
            "sizes": args.sizes, "repeats": args.repeats,
            "block_sizes": list(dict.fromkeys(args.block_sizes)),
            "blas_pools": threadpool_info(),
            "library_build_info": build_info,
            "thread_policy": "Numba serial; detected BLAS pools limited to 1; undetected backends unverified",
            "timing": "seeded interleaved calls; median; allocation included; JIT, generation and verification excluded",
            "samples_ms": [],
        }
        print(json.dumps({k: v for k, v in metadata.items()
                          if k not in ("library_build_info", "samples_ms")}, indent=2))
        print("\nNumba is serial; detected BLAS pools are limited to one thread.")
        print("Thread counts for undetected backends (such as Apple Accelerate) are unverified.")
        print("Validated methods also include dtype/layout and finite/symmetry checks.")
        print("numpy/scipy use their library implementations; scipy finite checks are disabled.")
        print(f"\n{'size':>6}  {'implementation':<26} {'median ms':>12} {'rel. residual':>15}")
        for n in args.sizes:
            m = rng.standard_normal((n, n))
            a = np.einsum("ik,jk->ij", m, m, optimize=False) + n * np.eye(n)
            samples, residuals = measure(implementations, a, args.repeats, order_rng)
            metadata["samples_ms"].append({"size": n, "implementations": samples})
            for name in implementations:
                elapsed_ms, residual = median(samples[name]), residuals[name]
                rows.append({"size": n, "implementation": name,
                             "block_size": blocks.get(name, ""), "median_ms": elapsed_ms,
                             "min_ms": min(samples[name]), "max_ms": max(samples[name]),
                             "relative_residual": residual})
                print(f"{n:6d}  {name:<26} {elapsed_ms:12.4f} {residual:15.3e}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.csv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
