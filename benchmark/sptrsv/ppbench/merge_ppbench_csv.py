import argparse
import csv
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Merge ppbench CSV shards.")
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for task CSV files, for example 'results/*.csv'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged CSV output path.",
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_rows(paths):
    rows = []
    fieldnames = None
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
    if fieldnames is None:
        raise ValueError("no input csv files found")
    return fieldnames, rows


def row_sort_key(row):
    idx = row.get("matrix_index", "")
    try:
        return (0, int(idx))
    except ValueError:
        return (1, idx)


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    fieldnames, rows = read_rows(paths)
    rows.sort(key=row_sort_key)

    ensure_parent_dir(args.output)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"merged {len(paths)} files into {args.output}")


if __name__ == "__main__":
    main()
