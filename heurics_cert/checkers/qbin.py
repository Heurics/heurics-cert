"""The binary model file of the split certificate layout (``CCPOQ1``).

Layout: the 8-byte magic ``CCPOQ1\\0\\0``, ``n`` as a little-endian int64, then ``n*n`` little-endian float64 in
row-major order. The C checker reads it with ``--qbin``. The fingerprint is over the values, not the file, so a
substituted model is detected whichever layout carries it.
"""
from __future__ import annotations

import os
import struct

import numpy as np

MAGIC = b"CCPOQ1\x00\x00"


def write(Q, path: str) -> str:
    """Write ``Q`` atomically (temporary file, then rename). Returns ``path``."""
    Q = np.asarray(Q, np.float64)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<q", Q.shape[0]))
        fh.write(np.ascontiguousarray(Q, dtype="<f8").tobytes())
    os.replace(tmp, path)
    return path


def read(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        if fh.read(8) != MAGIC:
            raise ValueError(f"{path}: not a CCPOQ1 model file")
        n = struct.unpack("<q", fh.read(8))[0]
        data = fh.read()
    if len(data) != 8 * n * n:
        raise ValueError(f"{path}: expected {8 * n * n} bytes of Q for n = {n}, got {len(data)}")
    return np.frombuffer(data, dtype="<f8").reshape(n, n).astype(np.float64)
