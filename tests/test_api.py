"""The package's structural promises: a cheap import, checkers that never import the prover, a public API that
resolves, a solver-free prove path, and a working command line."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from _solver_free import foreign_modules, loaded_top_level

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _modules_after(snippet: str) -> list[str]:
    code = snippet + "\nimport sys, json\nprint('__M__' + json.dumps(sorted(m for m in sys.modules)))"
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(x for x in reversed(out.stdout.splitlines()) if x.startswith("__M__"))
    return json.loads(line[5:])


def test_importing_the_package_loads_no_numerical_backend():
    mods = loaded_top_level("import heurics_cert")
    assert "numpy" not in mods and "jax" not in mods


def test_checkers_import_nothing_from_the_prover_side():
    mods = _modules_after("import heurics_cert.checkers as c; from heurics_cert.checkers import bound, copositive, "
                          "tree, native, qbin")
    ours = [m for m in mods if m.startswith("heurics_cert.")]
    forbidden = ("heurics_cert.prover", "heurics_cert.search", "heurics_cert.models", "heurics_cert.certificates",
                 "heurics_cert.extras")
    assert not [m for m in ours if m.startswith(forbidden)], ours
    assert "jax" not in {m.split(".")[0] for m in mods}


def test_bound_checker_runs_as_a_standalone_file(tmp_path):
    """The checker a reviewer reads in full is runnable on its own: no package import needed."""
    from heurics_cert.checkers import bound
    cert = {"Q": [[2.0, 0.0], [0.0, 2.0]], "mu": [0.1, 0.1], "lo": [0.0, 0.0], "hi": [1.0, 1.0], "K": 2,
            "rho": 0.0, "d": [1.0, 1.0], "w": [0.0, 0.0], "nu": 0.0, "pi": 0.0, "lam": 0.0, "beta_z": -1e-3,
            "claimed_bound": -1.0}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cert))
    out = subprocess.run([sys.executable, bound.__file__, str(path), "--exact"], capture_output=True, text=True,
                         cwd=str(tmp_path), env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert out.returncode == 0, out.stdout + out.stderr


def test_public_names_resolve():
    import heurics_cert as hc
    for name in ("Problem", "Certificate", "TreeCertificate", "Witness", "load", "prove", "Result", "verify",
                 "Verdict", "check_tree", "Status", "Params", "prove_batch", "prove_copositive", "__version__"):
        assert getattr(hc, name) is not None, name
    for sub in ("models", "certificates", "prover", "checkers", "search", "errors"):
        assert getattr(hc, sub).__name__ == f"heurics_cert.{sub}"
    assert callable(hc.prove) and callable(hc.verify)


def test_status_parsing_and_exit_codes():
    from heurics_cert.status import Status, exit_code
    assert Status.parse("REFUTED: pi < 0") is Status.REFUTED
    assert Status.parse("NOT PROVED: thin margin") is Status.NOT_PROVED
    assert Status.parse("NOT_PROVED") is Status.NOT_PROVED
    assert Status.parse("MODEL_MISMATCH") is Status.MODEL_MISMATCH
    assert [exit_code(s) for s in (Status.PROVED, Status.REFUTED, Status.MODEL_MISMATCH, Status.NOT_PROVED)] == \
        [0, 1, 2, 3]


def test_invalid_problems_raise_a_typed_error():
    import heurics_cert as hc
    from heurics_cert.errors import InvalidProblemError
    with pytest.raises(InvalidProblemError):
        hc.Problem(np.array([[1.0, 0.1], [0.0, 1.0]]), [0.1, 0.1], 0.0, 1.0, 1, 0.0)    # asymmetric
    with pytest.raises(InvalidProblemError):
        hc.Problem(np.eye(2), [0.1, 0.1], 0.5, 0.4, 1, 0.0)                             # lo > hi


def test_prove_path_loads_no_optimization_package():
    pytest.importorskip("jax")
    snippet = ("import numpy as np, heurics_cert as hc\n"
               "rng = np.random.default_rng(0); B = rng.standard_normal((14, 9)) * 0.1; Q = B.T @ B / 14\n"
               "Q = 0.5 * (Q + Q.T); mu = rng.normal(0.01, 0.02, 9)\n"
               "p = hc.Problem(Q, mu, 0.01, 1.0, 3, float(np.quantile(mu, 0.4)))\n"
               "assert hc.prove(p, params=hc.Params(check=('rigorous',))).proved")
    assert foreign_modules(snippet) == []


def test_command_line_info_and_verify(tmp_path):
    from heurics_cert.cli import main
    assert main(["info"]) == 0
    cert = {"Q": [[2.0, 0.0], [0.0, 2.0]], "mu": [0.1, 0.1], "lo": [0.0, 0.0], "hi": [1.0, 1.0], "K": 2,
            "rho": 0.0, "d": [1.0, 1.0], "w": [0.0, 0.0], "nu": 0.0, "pi": 0.0, "lam": 0.0, "beta_z": -1e-3,
            "claimed_bound": -1.0}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cert))
    assert main(["verify", str(path)]) == 0
    assert main(["verify", str(path), "--expect-model", "0000"]) == 2
