import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "matrices" / "suitesparse"
MANIFEST_NAME = "manifest.txt"

DEFAULT_MATRIX_NAMES = [
    "1138_bus",
    "2cubes_sphere",
    "Chem97ZtZ",
    "Dubcova1",
    "Dubcova2",
    "Dubcova3",
    "Emilia_923",
    "Fault_639",
    "Flan_1565",
    "G2_circuit",
    "G3_circuit",
    "Geo_1438",
    "Hook_1498",
    "Kuu",
    "LF10000",
    "Muu",
    "PFlow_742",
    "Pres_Poisson",
    "Serena",
    "StocF-1465",
    "Trefethen_2000",
    "Trefethen_20000",
    "bcsstk08",
    "bcsstk09",
    "bcsstk10",
    "bcsstk11",
    "bcsstk12",
    "bcsstk13",
    "bcsstk14",
    "bcsstk15",
    "bcsstk16",
    "bcsstk17",
    "bcsstk18",
    "bcsstk21",
    "bcsstk23",
    "bcsstk24",
    "bcsstk25",
    "bcsstk26",
    "bcsstk27",
    "bcsstk28",
    "bcsstk36",
    "bcsstk38",
    "bcsstm08",
    "bcsstm09",
    "bcsstm11",
    "bcsstm12",
    "bcsstm21",
    "bcsstm23",
    "bcsstm24",
    "bcsstm25",
    "bcsstm26",
    "bcsstm39",
    "bibd_81_2",
    "bmw7st_1",
    "bmwcra_1",
    "bodyy4",
    "bodyy5",
    "bodyy6",
    "bone010",
    "boneS01",
    "cbuckle",
    "cfd1",
    "cfd2",
    "crystm01",
    "crystm02",
    "crystm03",
    "ct20stif",
    "cvxbqp1",
    "denormal",
    "ecology2",
    "ex10",
    "ex10hs",
    "ex13",
    "ex15",
    "ex3",
    "ex33",
    "ex9",
    "finan512",
    "fv1",
    "fv2",
    "fv3",
    "gyro",
    "inline_1",
    "jnlbrng1",
    "ldoor",
    "msdoor",
    "nasa1824",
    "nasa2146",
    "nasa2910",
    "nasa4704",
    "nasasrb",
    "nd12k",
    "nd24k",
    "nd3k",
    "nd6k",
    "obstclae",
    "offshore",
    "oilpan",
    "olafu",
    "parabolic_fem",
    "pdb1HYS",
    "plat1919",
    "plbuckle",
    "pwtk",
    "qa8fm",
    "raefsky4",
    "s1rmq4m1",
    "s1rmt3m1",
    "s2rmq4m1",
    "s2rmt3m1",
    "s3dkq4m2",
    "s3dkt3m2",
    "s3rmq4m1",
    "s3rmt3m1",
    "s3rmt3m3",
    "shallow_water1",
    "shallow_water2",
    "ship_001",
    "ship_003",
    "shipsec1",
    "shipsec5",
    "shipsec8",
    "smt",
    "sts4098",
    "t2dah_e",
    "t2dal_e",
    "t3dl_e",
    "ted_B",
    "thermal1",
    "thermal2",
    "thermomech_TC",
    "thermomech_TK",
    "thermomech_dM",
    "tmt_sym",
    "torsion1",
    "vanbody",
    "wathen100",
    "wathen120",
    "x104",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SuiteSparse matrices for the ILU0 benchmark."
    )
    parser.add_argument(
        "matrix_names",
        nargs="*",
        default=DEFAULT_MATRIX_NAMES,
        help="SuiteSparse matrix names to download. Defaults to the curated ILU0 list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded matrices. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def find_existing_mtx(output_dir, matrix_name):
    matches = sorted(output_dir.rglob(f"{matrix_name}.mtx"))
    return matches[0] if matches else None


def find_downloaded_mtx(output_dir, matrix_name, previous_mtx_files):
    exact_match = find_existing_mtx(output_dir, matrix_name)
    if exact_match is not None:
        return exact_match

    current_mtx_files = set(output_dir.rglob("*.mtx"))
    new_files = sorted(current_mtx_files - previous_mtx_files)
    return new_files[0] if new_files else None


def select_exact_matrix(matrices, matrix_name):
    for matrix in matrices:
        if matrix.name == matrix_name:
            return matrix
    return None


def remove_download_archive(download_result):
    paths = download_result
    if not isinstance(paths, (tuple, list)):
        paths = [paths]

    for path in paths:
        if path is None:
            continue
        archive = Path(path)
        if archive.suffixes[-2:] == [".tar", ".gz"] and archive.exists():
            archive.unlink()
            print(f"Removed archive: {archive}")


def write_manifest(output_dir, matrix_paths):
    manifest = output_dir / MANIFEST_NAME
    with manifest.open("w", encoding="utf-8") as f:
        for matrix_path in sorted(matrix_paths):
            f.write(f"{matrix_path.relative_to(output_dir)}\n")
    print(f"Wrote manifest: {manifest}")


def main():
    args = parse_args()

    import ssgetpy

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_or_existing_paths = []
    downloaded_count = 0

    for matrix_name in args.matrix_names:
        existing_mtx = find_existing_mtx(output_dir, matrix_name)
        if existing_mtx is not None:
            print(f"Skipping existing matrix: {existing_mtx}")
            downloaded_or_existing_paths.append(existing_mtx)
            continue

        print(f"Searching for matrix: {matrix_name}")
        matrices = ssgetpy.search(name=matrix_name)
        matrix_entry = select_exact_matrix(matrices, matrix_name)

        if matrix_entry is None:
            print(f"Matrix {matrix_name} not found in SuiteSparse collection.")
            continue

        print(f"Downloading and extracting matrix: {matrix_entry.name}")
        previous_mtx_files = set(output_dir.rglob("*.mtx"))
        download_result = matrix_entry.download(
            destpath=str(output_dir),
            format="MM",
            extract=True,
        )
        remove_download_archive(download_result)

        matrix_path = find_downloaded_mtx(output_dir, matrix_name, previous_mtx_files)
        if matrix_path is None:
            print(f"Downloaded {matrix_name}, but no .mtx file was found.")
            continue

        downloaded_or_existing_paths.append(matrix_path)
        downloaded_count += 1

    if downloaded_or_existing_paths:
        write_manifest(output_dir, downloaded_or_existing_paths)

    print(f"Total matrices downloaded: {downloaded_count}")
    print(f"Matrix directory: {output_dir}")


if __name__ == "__main__":
    main()
