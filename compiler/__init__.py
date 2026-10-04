"""Compiler package initialization."""

import os

# Numba 0.61's legacy ctypes binding is incompatible with newer drivers that
# export ``cuCtxGetDevice_v2`` with a changed ABI.  Select NVIDIA's versioned
# CUDA Python binding before any compiler module can import ``numba.cuda``.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
