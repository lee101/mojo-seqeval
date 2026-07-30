"""ctypes loader for the Mojo sequence-evaluation kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "seqeval.mojo")
LIB = os.path.abspath(
    os.environ.get("MOJOSEQEVAL_LIB")
    or os.path.join(ROOT, "dist", "libmojo-seqeval.so")
)

I = ctypes.c_int64

_SIGNATURES = {
    "mse_extract_default": ([I, I, I, I, I], I),
    "mse_extract_strict": ([I, I, I, I, I, I], I),
    "mse_extract_pair_default": ([I] * 11, None),
    "mse_extract_pair_strict": ([I] * 12, None),
    "mse_count_entities": ([I] * 12, None),
    "mse_token_counts": ([I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOSEQEVAL_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.exists(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo source at {SRC} and no library at {LIB}; "
            "set MOJOSEQEVAL_LIB to a prebuilt shared library"
        )
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build` or set MOJOSEQEVAL_LIB")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        path = build()
        try:
            _library = ctypes.CDLL(path)
        except OSError as exc:
            raise BuildError(f"could not load Mojo library {path}: {exc}") from exc
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array) -> int:
    """Return an address only for buffers matching the Mojo ABI contract.

    All current kernels consume native, aligned, contiguous int64 arrays.  Keeping
    this check at the sole address conversion point prevents a future caller from
    accidentally passing a strided view or a narrowing dtype to raw pointer code.
    The caller retains the ndarray reference for the duration of the synchronous
    ctypes call, so NumPy owns the storage throughout the call.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Mojo FFI buffers must be NumPy arrays")
    if array.dtype != np.dtype(np.int64):
        raise TypeError(f"Mojo FFI buffers require int64, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("Mojo FFI buffers must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError("Mojo FFI buffers must be aligned")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("Mojo FFI buffers must have a non-null address")
    return address
