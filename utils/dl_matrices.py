#!/usr/bin/env python3
"""
Download matrices from the SuiteSparse Matrix Collection using ssgetpy.

Install dependency first:
    pip install ssgetpy

Usage:
    python download_matrices.py --list <list_file> --output <output_dir> [--format MM|RB|MAT] [--extract]

List file format (one entry per line):
    bcsstk10/bcsstk10.mtx   ->  searches for matrix named "bcsstk10"
    bcsstk11/bcsstk11.mtx   ->  searches for matrix named "bcsstk11"
    ...
"""

import argparse
import sys
from pathlib import Path

try:
    import ssgetpy
except ImportError:
    print("[ERROR] ssgetpy is not installed. Run: pip install ssgetpy")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SuiteSparse matrices listed in a file using ssgetpy."
    )
    parser.add_argument(
        "--list", "-l",
        required=True,
        help="Path to the text file containing matrix paths (e.g. bcsstk10/bcsstk10.mtx)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./matrices",
        help="Directory to save downloaded matrices (default: ./matrices)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["MM", "RB", "MAT"],
        default="MM",
        help="Download format: MM (Matrix Market, default), RB (Rutherford-Boeing), or MAT (MATLAB)"
    )
    parser.add_argument(
        "--extract", "-e",
        action="store_true",
        help="Automatically extract downloaded .tar.gz archives"
    )
    return parser.parse_args()


def read_matrix_list(list_file: str) -> list:
    """
    Parse the list file and extract just the matrix name from each line.
    Handles formats like:
      bcsstk10/bcsstk10.mtx  ->  "bcsstk10"
      bcsstk10.mtx           ->  "bcsstk10"
      bcsstk10               ->  "bcsstk10"
    Ignores blank lines and comments starting with #.
    """
    path = Path(list_file)
    if not path.exists():
        print(f"[ERROR] List file not found: {list_file}")
        sys.exit(1)

    names = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Take the last path component and strip the extension
            name = Path(line.replace("\\", "/")).name   # handles both separators
            name = Path(name).stem                       # strips .mtx (or any extension)

            if not name:
                print(f"[WARNING] Line {lineno}: could not parse '{line}'. Skipping.")
                continue

            names.append(name)
    return names


def download_matrix(name: str, output_dir: Path, fmt: str, extract: bool) -> bool:
    """Search for a matrix by exact name and download it. Returns True on success."""
    results = ssgetpy.search(name)

    if not results:
        print(f"  [NOT FOUND] No results for '{name}'")
        return False

    # Pick the entry whose name matches exactly (search may return partial matches)
    matrix = next((m for m in results if m.name == name), None)
    if matrix is None:
        print(f"  [NOT FOUND] No exact match for '{name}' (candidates: {[m.name for m in results]})")
        return False

    try:
        archive_path, extracted_path = matrix.download(
            format=fmt,
            destpath=str(output_dir),
            extract=extract
        )
        display_path = extracted_path or archive_path
        print(f"  [{matrix.group}/{matrix.name}]  ->  {display_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download '{name}': {e}")
        return False


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Format           : {args.format}")
    print(f"Extract archives : {args.extract}\n")

    names = read_matrix_list(args.list)
    total = len(names)
    print(f"Matrices to download: {total}\n")

    succeeded, failed = [], []

    for i, name in enumerate(names, start=1):
        print(f"[{i}/{total}] {name}")
        ok = download_matrix(name, output_dir, args.format, args.extract)
        (succeeded if ok else failed).append(name)

    # Summary
    print("\n" + "=" * 50)
    print(f"Done. {len(succeeded)}/{total} matrices downloaded successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for entry in failed:
            print(f"  x {entry}")


if __name__ == "__main__":
    main()