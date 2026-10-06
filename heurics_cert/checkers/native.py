"""Runner for ``ccpocheck``, the C99 checker that shares no code with the Python one.

The binary is found at ``$CCPOCHECK``, else where ``heurics-cert build-checker`` puts it (``~/.heurics-cert/bin``),
else beside its source, else on PATH -- the first that exists and can actually be launched here. Without one, checks
report ``UNAVAILABLE``.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from heurics_cert.checkers import qbin as _qbin

BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c", "ccpocheck")
_EXE = "ccpocheck.exe" if os.name == "nt" else "ccpocheck"
USER = os.path.join(os.path.expanduser("~"), ".heurics-cert", "bin", _EXE)


@functools.lru_cache(maxsize=None)
def runnable(path: str) -> bool:
    """Whether ``path`` can be launched (run without arguments, the checker prints usage and exits)."""
    try:
        subprocess.run([path], capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def binary() -> str | None:
    for p in (os.environ.get("CCPOCHECK"), USER, BUNDLED, BUNDLED + ".exe", shutil.which("ccpocheck")):
        if p and os.path.isfile(p) and os.access(p, os.X_OK) and runnable(p):
            return p
    return None


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run(doc: dict, *, qbin: str | None = None, expect_model: str | None = None, timeout: float = 1800.0) -> dict:
    """Check a CCPO-CERT/1 document with the C checker. Returns its JSON report, or ``{"verdict": "UNAVAILABLE"}`` /
    ``{"verdict": "ERROR", ...}``. With ``qbin``, ``Q`` is read from that model file (written first if missing)."""
    exe = binary()
    if exe is None:
        return {"verdict": "UNAVAILABLE", "reason": "ccpocheck binary not found; build it or set $CCPOCHECK"}
    with tempfile.TemporaryDirectory() as td:
        body = {k: v for k, v in doc.items() if not (qbin and k == "Q")}
        if qbin:
            body["n"] = len(doc["mu"])
            if not os.path.exists(qbin):
                _qbin.write(doc["Q"], qbin)
        path = os.path.join(td, "cert.json")
        with open(path, "w") as fh:
            json.dump({k: _plain(v) for k, v in body.items()}, fh)
        cmd = [exe, path, "--json"] + (["--qbin", qbin] if qbin else []) + \
              (["--expect-model", expect_model] if expect_model else [])
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"verdict": "ERROR", "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
