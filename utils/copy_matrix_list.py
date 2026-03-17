#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path


def copy_selected_matrices(list_file, source_root, dest_root):
    list_file = Path(list_file)
    source_root = Path(source_root)
    dest_root = Path(dest_root)

    if not list_file.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    errors = 0

    with list_file.open("r") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue

            rel = Path(rel_path)

            # Expected format: matrix_name/matrix_name.mtx
            if len(rel.parts) < 2:
                print(f"Skipping malformed line: {rel_path}")
                skipped += 1
                continue

            matrix_name = rel.parts[0]
            src_folder = source_root / matrix_name
            src_file = source_root / rel

            dst_folder = dest_root / matrix_name
            dst_file = dest_root / rel

            if not src_folder.exists():
                print(f"Missing source folder: {src_folder}")
                errors += 1
                continue

            if not src_file.exists():
                print(f"Missing source file: {src_file}")
                errors += 1
                continue

            try:
                dst_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied += 1
                print(f"Copied: {src_file} -> {dst_file}")
            except Exception as e:
                print(f"Error copying {src_file}: {e}")
                errors += 1

    print("\nSummary")
    print(f"Copied : {copied}")
    print(f"Skipped: {skipped}")
    print(f"Errors : {errors}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 copy_selected_matrices.py <list_file> <source_root> <dest_root>")
        sys.exit(1)

    copy_selected_matrices(
        list_file=sys.argv[1],
        source_root=sys.argv[2],
        dest_root=sys.argv[3],
    )