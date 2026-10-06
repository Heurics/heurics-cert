"""A certificate computation may use the standard library, the array stack and this package -- nothing else.

Run in a fresh interpreter, so a solver imported by some other test cannot hide an import made on the bound path.
This is an allowlist rather than a list of forbidden products: it catches any optimization package, from any
vendor, without having to name one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

#: top-level packages the bound path is allowed to load (the array stack and its runtime dependencies)
ALLOWED = {
    "heurics_cert",
    "numpy", "scipy", "jax", "jaxlib", "ml_dtypes", "opt_einsum", "jax_plugins",
    "jax_cuda12_plugin", "jax_cuda12_pjrt", "nvidia",
    "cython_runtime",  # SciPy's Cython shim
    "jaxtyping",       # array annotations in heurics_cert.extras
    "pygments",        # imported by jax itself
    "typing_extensions", "packaging", "importlib_metadata", "zipp",
}

#: what the allowlisted stack itself loads in this environment: interpreter startup hooks (`sitecustomize`) and the
#: array stack's own optional imports (jax pulls `cloudpickle`) differ between installs, are not on the bound path,
#: and must not be charged to it
BASELINE_SNIPPET = "import numpy, scipy, scipy.linalg, scipy.optimize, jax, jax.numpy, jaxtyping"

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loaded_top_level(snippet: str) -> list[str]:
    """Top-level modules present after running `snippet` in a fresh interpreter."""
    probe = "import sys, json\nprint('__MODS__' + json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    code = snippet + "\n" + probe
    env = dict(os.environ, PYTHONPATH=_REPO + os.pathsep + os.environ.get("PYTHONPATH", ""),
               JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env=env,
                         cwd=_REPO)
    if out.returncode != 0:
        raise RuntimeError(f"snippet failed:\n{out.stderr[-3000:]}")
    line = next(l for l in reversed(out.stdout.splitlines()) if l.startswith("__MODS__"))
    return json.loads(line[len("__MODS__"):])


def foreign_modules(snippet: str) -> list[str]:
    """Modules loaded by `snippet` that are not standard library, not on the allowlist, and not already loaded by
    the allowlisted stack on its own (`BASELINE_SNIPPET`)."""
    std = set(sys.stdlib_module_names)
    baseline = set(loaded_top_level(BASELINE_SNIPPET))
    return sorted(m for m in loaded_top_level(snippet)
                  if m not in ALLOWED and m not in std and m not in baseline and not m.startswith("_"))
