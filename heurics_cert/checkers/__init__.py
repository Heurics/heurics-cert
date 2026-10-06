"""Independent checkers. Nothing here imports the prover. The bound checker runs on stdlib + numpy (its exact tier
and the fingerprint on stdlib alone); the C checker shares no code with any of it.

    from heurics_cert.checkers import verify, check_tree
    verify("cert.json", method="c").status                  # PROVED / REFUTED / NOT PROVED / MODEL_MISMATCH
    check_tree(problem, witness, record)["verdict"]          # OPTIMAL / BOUND / REFUTED / NOT PROVED

Modules: ``bound`` (CCPO-CERT/1, rigorous and exact tiers; also runnable as a script), ``copositive``
(CCPO-CERT/2), ``tree`` (CCPO-TREE/0), ``native`` (the C checker, source in ``c/``), ``qbin`` (the binary model file).
"""
from heurics_cert.checkers.api import METHODS, Verdict, verify
from heurics_cert.checkers.bound import model_hash


def check_tree(problem, witness, record, *, method: str = "rigorous", rel_tol: float = 1e-6) -> dict:
    """Check a CCPO-TREE/0 record against the model and its root witness (see ``heurics_cert.checkers.tree``)."""
    from heurics_cert.checkers.tree import check_tree as _check_tree
    return _check_tree(problem, witness, record, method=method, rel_tol=rel_tol)


def c_checker() -> str | None:
    """Path of a launchable ``ccpocheck`` binary, or None."""
    from heurics_cert.checkers.native import binary
    return binary()


__all__ = ["METHODS", "Verdict", "c_checker", "check_tree", "model_hash", "verify"]
