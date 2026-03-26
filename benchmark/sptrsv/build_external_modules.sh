#!/usr/bin/env bash
# Build yysptrsv_wrapper and cusparse_sptrsv pybind modules, copy *.so into this directory
# so `import yysptrsv_wrapper` works when running slurm_sptrsv_benchmark.py from benchmark/sptrsv.

set -euo pipefail

SPTRSV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SPTRSV_DIR}/../.." && pwd)"
YY_SRC="${REPO_ROOT}/benchmark/yysptrsv"
CUSPARSE_SRC="${REPO_ROOT}/benchmark/cusparse_benchmark"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} not found (set PYTHON_BIN)" >&2
  exit 1
fi

if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found (install cmake)" >&2
  exit 1
fi

_PY_EXE="$("$PYTHON_BIN" -c "import sys; print(sys.executable)" 2>/dev/null)" || _PY_EXE="$PYTHON_BIN"
if ! "$PYTHON_BIN" -c "import pybind11" 2>/dev/null; then
  echo "ERROR: pybind11 is not installed for this interpreter: ${_PY_EXE}" >&2
  echo "  Install with:  ${PYTHON_BIN} -m pip install pybind11" >&2
  echo "  (Use the same venv you use for slurm_sptrsv_benchmark.py; set PYTHON_BIN if needed.)" >&2
  exit 1
fi
PYBIND11_CMAKE_DIR="$("$PYTHON_BIN" -c "import pybind11; print(pybind11.get_cmake_dir())")"
_PY_VER="$("$PYTHON_BIN" -c "import sys; print(\"%d.%d\" % sys.version_info[:2])")"

# Paths for CMake FindPython3 (venv interpreters often fail "Development.Module" without these hints).
_PY_CMAKE_ARGS=()
if ! _PY_PATHS_OUT="$("$PYTHON_BIN" - <<'PY'
import glob
import os
import sys
import sysconfig

inc = sysconfig.get_path("include")
if not os.path.isfile(os.path.join(inc, "Python.h")):
    sys.stderr.write(
        f"ERROR: Python.h not found under:\n  {inc}\n"
        "Install OS headers, e.g. on Ubuntu:\n"
        f"  sudo apt install python{sys.version_info.major}.{sys.version_info.minor}-dev\n"
        "or: sudo apt install python3-dev\n"
    )
    sys.exit(2)

libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlib = sysconfig.get_config_var("LDLIBRARY") or ""
lib = None
if ldlib:
    cand = ldlib if os.path.isabs(ldlib) else os.path.join(libdir, ldlib)
    if os.path.isfile(cand):
        lib = cand

ver = sysconfig.get_config_var("VERSION") or f"{sys.version_info.major}.{sys.version_info.minor}"
if lib is None or lib.endswith(".a"):
    for name in (
        f"libpython{ver}.so",
        f"libpython{ver}.so.1.0",
    ):
        cand = os.path.join(libdir, name)
        if os.path.isfile(cand):
            lib = cand
            break

if lib is None or not os.path.isfile(lib):
    cands = sorted(
        p for p in glob.glob(os.path.join(libdir, "libpython*.so*"))
        if os.path.isfile(p) and not p.endswith(".a")
    )
    lib = cands[0] if cands else None

if not lib:
    sys.stderr.write(
        f"ERROR: Could not find libpython shared library under:\n  {libdir}\n"
        "Install OS Python dev package (see Python.h message above).\n"
    )
    sys.exit(3)

print(inc)
print(lib)
PY
)"; then
  exit 1
fi
_PY_INC="$(echo "$_PY_PATHS_OUT" | sed -n '1p')"
_PY_LIB="$(echo "$_PY_PATHS_OUT" | sed -n '2p')"
_PY_CMAKE_ARGS+=(
  "-DPython3_INCLUDE_DIR=${_PY_INC}"
  "-DPython3_LIBRARY=${_PY_LIB}"
)

GPU_ARCH="${SPTRSV_CUDA_ARCH:-}"
if [[ -z "${GPU_ARCH}" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
  fi
fi
if [[ -z "${GPU_ARCH}" ]]; then
  echo "WARNING: could not detect GPU arch; using native (may fail on headless nodes)" >&2
  GPU_ARCH="native"
fi

CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
export CUDA_HOME="${CUDA_ROOT}"

echo "Building external SpTRSV baselines -> ${SPTRSV_DIR}"
echo "  Python: ${_PY_EXE} (${_PY_VER})"
echo "  GPU arch (CMAKE_CUDA_ARCHITECTURES): ${GPU_ARCH}"
echo "  CUDA: ${CUDA_ROOT}"
echo "  pybind11: ${PYBIND11_CMAKE_DIR}"

# If the chosen interpreter changed, drop CMake caches (otherwise you can link a cpython-313 .so while using a 3.12 venv).
_EXT_PY_MARKER="${SPTRSV_DIR}/.external_build_python"
if [[ ! -f "${_EXT_PY_MARKER}" ]] || [[ "$(cat "${_EXT_PY_MARKER}")" != "${_PY_EXE}" ]]; then
  echo "  (cleaning CMake build dirs: interpreter changed or first build)"
  rm -rf "${YY_SRC}/build" "${CUSPARSE_SRC}/build"
fi
echo "${_PY_EXE}" > "${_EXT_PY_MARKER}"

mkdir -p "${YY_SRC}/build" "${CUSPARSE_SRC}/build"

# Drop stale extensions so the wrong cpython-XY tag cannot shadow the new build.
rm -f "${SPTRSV_DIR}"/yysptrsv_wrapper*.so "${SPTRSV_DIR}"/cusparse_sptrsv*.so

_CMAKE_EXTRA=()
if cmake --help 2>/dev/null | grep -q -- '--fresh'; then
  _CMAKE_EXTRA+=(--fresh)
fi

echo "== yysptrsv_wrapper =="
cmake "${_CMAKE_EXTRA[@]}" -S "${YY_SRC}" -B "${YY_SRC}/build" \
  -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
  -DPython3_EXECUTABLE="${_PY_EXE}" \
  "${_PY_CMAKE_ARGS[@]}" \
  -DPYBIND11_FINDPYTHON=ON \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"
cmake --build "${YY_SRC}/build" -j"$(nproc 2>/dev/null || echo 4)" --target yysptrsv_wrapper

echo "== cusparse_sptrsv =="
cmake "${_CMAKE_EXTRA[@]}" -S "${CUSPARSE_SRC}" -B "${CUSPARSE_SRC}/build" \
  -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
  -DPython3_EXECUTABLE="${_PY_EXE}" \
  "${_PY_CMAKE_ARGS[@]}" \
  -DPYBIND11_FINDPYTHON=ON \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"
cmake --build "${CUSPARSE_SRC}/build" -j"$(nproc 2>/dev/null || echo 4)" --target cusparse_sptrsv

shopt -s nullglob
for so in "${YY_SRC}/build"/yysptrsv_wrapper*.so; do
  cp -f "$so" "${SPTRSV_DIR}/"
  echo "  installed $(basename "$so")"
done
for so in "${CUSPARSE_SRC}/build"/cusparse_sptrsv*.so; do
  cp -f "$so" "${SPTRSV_DIR}/"
  echo "  installed $(basename "$so")"
done

"$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '${SPTRSV_DIR}')
import yysptrsv_wrapper
import cusparse_sptrsv
print('OK: yysptrsv_wrapper and cusparse_sptrsv import from ${SPTRSV_DIR}')
"
