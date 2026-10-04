"""Compiler regression tests.

The project's current CUDA environment uses the NVIDIA ``cuda-python``
binding. Selecting it before Numba is imported avoids the legacy ctypes
binding used by default in some Numba installations.
"""

import os


os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
