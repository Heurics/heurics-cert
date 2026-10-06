"""Build the C checker from its bundled source: ``heurics-cert build-checker``.

The binary goes to ``~/.heurics-cert/bin`` (or ``--dest``), where ``heurics_cert.checkers.native`` finds it. The
compiler is ``$CC``, else ``cc``, ``gcc`` or ``clang`` on PATH, else (Windows) MSVC through Visual Studio's
environment script. Floating-point contraction and fast-math are disabled: the checker's error analysis assumes one
correctly rounded operation at a time. The new binary is self-tested before it is reported as built.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c", "ccpocheck.c")
EXE = "ccpocheck.exe" if sys.platform == "win32" else "ccpocheck"


def default_dest() -> str:
    return os.path.join(os.path.expanduser("~"), ".heurics-cert", "bin")


def _unix_command(cc, out):
    return [cc, "-std=c99", "-O2", "-ffp-contract=off", "-fno-fast-math", "-o", out, SOURCE, "-lm"]


def _vcvars() -> str | None:
    """Visual Studio's ``vcvars64.bat``, located through ``vswhere`` or the standard install paths."""
    roots = [os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.environ.get("ProgramFiles",
                                                                                          r"C:\Program Files")]
    vswhere = os.path.join(roots[0], "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.exists(vswhere):
        out = subprocess.run([vswhere, "-latest", "-property", "installationPath"], capture_output=True, text=True)
        path = os.path.join(out.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
        if out.stdout.strip() and os.path.exists(path):
            return path
    for root in roots:
        hits = glob.glob(os.path.join(root, "Microsoft Visual Studio", "*", "*", "VC", "Auxiliary", "Build",
                                      "vcvars64.bat"))
        if hits:
            return sorted(hits)[-1]
    return None


def build(dest: str | None = None, compiler: str | None = None) -> str:
    """Compile the checker into ``dest`` and self-test it. Returns the binary's path; raises ``RuntimeError`` with
    the compiler's output when no compiler is found or the build fails."""
    dest = dest or default_dest()
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, EXE)
    cc = compiler or os.environ.get("CC") or next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cc and os.path.basename(cc).lower() not in ("cl", "cl.exe"):
        proc = subprocess.run(_unix_command(cc, out), capture_output=True, text=True)
    elif sys.platform == "win32":
        flags = f'/nologo /O2 /fp:strict /W3 /Fe:"{out}" "{SOURCE}"'
        if cc or shutil.which("cl"):
            proc = subprocess.run(f"cl {flags}", capture_output=True, text=True, shell=True, cwd=tempfile.gettempdir())
        else:
            vcvars = _vcvars()
            if vcvars is None:
                raise RuntimeError("no C compiler found: install a C compiler (gcc, clang, or Visual Studio's C++ "
                                   "build tools) or set $CC")
            proc = subprocess.run(f'call "{vcvars}" >nul && cl {flags}', capture_output=True, text=True, shell=True,
                                  cwd=tempfile.gettempdir())
    else:
        raise RuntimeError("no C compiler found: install cc, gcc or clang, or set $CC")
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"building the C checker failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    _self_test(out)
    return out


def _self_test(exe: str) -> None:
    """The new binary must prove a valid certificate and refute a forged one."""
    import json
    cert = {"Q": [[2.0, 0.0], [0.0, 2.0]], "mu": [0.1, 0.1], "lo": [0.0, 0.0], "hi": [1.0, 1.0], "K": 2, "rho": 0.0,
            "d": [1.0, 1.0], "w": [0.0, 0.0], "nu": 0.0, "pi": 0.0, "lam": 0.0, "beta_z": -1e-3, "claimed_bound": -1.0}
    with tempfile.TemporaryDirectory() as td:
        for claim, want in ((-1.0, 0), (10.0, 1)):
            path = os.path.join(td, "c.json")
            with open(path, "w") as fh:
                json.dump(dict(cert, claimed_bound=claim), fh)
            code = subprocess.run([exe, path, "--json"], capture_output=True, text=True).returncode
            if code != want:
                raise RuntimeError(f"the built checker failed its self-test (exit {code} for claim {claim}, "
                                   f"expected {want})")
