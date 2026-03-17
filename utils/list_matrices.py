#!/usr/bin/env python3

import sys
import re
from pathlib import Path
from urllib.request import urlretrieve

from scipy.io import loadmat

INDEX_URL = "https://sparse.tamu.edu/files/ss_index.mat"
INDEX_FILE = "ss_index.mat"
OUTPUT_FILE = "suitesparse_matrix_list.txt"


def normalize(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_key(d, candidates):
    norm_to_key = {normalize(k): k for k in d.keys()}
    for cand in candidates:
        nc = normalize(cand)
        if nc in norm_to_key:
            return norm_to_key[nc]
    for cand in candidates:
        nc = normalize(cand)
        for nk, orig in norm_to_key.items():
            if nc in nk or nk in nc:
                return orig
    return None


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.squeeze().tolist() if x.size > 1 else [x.item()]
    except Exception:
        pass
    return [x]


def scalar(v):
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return v.item()
            return v.squeeze().tolist()
    except Exception:
        pass
    return v


def as_bool(v):
    v = scalar(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"yes", "true", "1", "y"}:
            return True
        if s in {"no", "false", "0", "n"}:
            return False
    try:
        return bool(int(v))
    except Exception:
        return bool(v)


def as_str(v):
    v = scalar(v)
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    return str(v)


def main():
    out_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(OUTPUT_FILE)

    if not Path(INDEX_FILE).exists():
        print(f"Downloading metadata index: {INDEX_URL}")
        urlretrieve(INDEX_URL, INDEX_FILE)

    data = loadmat(INDEX_FILE, simplify_cells=True)

    # In most ss_index.mat files, the main metadata struct is under "ss_index".
    ss = data.get("ss_index", data)

    # Try common field names used for SuiteSparse metadata.
    group_key = find_key(ss, ["Group", "group"])
    name_key = find_key(ss, ["Name", "name"])
    rows_key = find_key(ss, ["nrows", "rows", "m"])
    cols_key = find_key(ss, ["ncols", "cols", "n"])
    nnz_key = find_key(ss, ["nnz", "nz", "nonzeros"])
    posdef_key = find_key(ss, ["posdef", "positive_definite", "PositiveDefinite"])
    rbtype_key = find_key(ss, ["RBtype", "rbtype", "RutherfordBoeingType", "type"])

    missing = [k for k, v in {
        "group": group_key,
        "name": name_key,
        "rows": rows_key,
        "cols": cols_key,
        "nnz": nnz_key,
        "posdef": posdef_key,
        "rbtype": rbtype_key,
    }.items() if v is None]

    if missing:
        raise RuntimeError(
            f"Could not locate these fields in ss_index.mat: {missing}\n"
            f"Available top-level keys: {list(ss.keys())}"
        )

    groups = to_list(ss[group_key])
    names = to_list(ss[name_key])
    rows = to_list(ss[rows_key])
    cols = to_list(ss[cols_key])
    nnz = to_list(ss[nnz_key])
    posdef = to_list(ss[posdef_key])
    rbtype = to_list(ss[rbtype_key])

    n = min(len(groups), len(names), len(rows), len(cols), len(nnz), len(posdef), len(rbtype))

    selected = []
    for i in range(n):
        r = int(scalar(rows[i]))
        c = int(scalar(cols[i]))
        z = int(scalar(nnz[i]))
        pd = as_bool(posdef[i])
        t = as_str(rbtype[i]).strip().lower()

        # Rutherford-Boeing first letter encodes numeric type:
        # r = real, c = complex, p = pattern, i = integer
        is_real = len(t) > 0 and t[0] == "r"

        if (not pd) and (r == c) and is_real and (10**6 <= z <= 10**7):
            name = as_str(names[i]).strip()
            selected.append(f"{name}/{name}.mtx")

    selected = sorted(set(selected))

    out_file.write_text("\n".join(selected) + ("\n" if selected else ""))
    print(f"Wrote {len(selected)} matrix names to {out_file}")


if __name__ == "__main__":
    main()