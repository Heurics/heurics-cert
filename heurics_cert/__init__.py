"""heurics-cert: checkable optimality certificates for cardinality-constrained portfolio optimization.

    import heurics_cert as hc

    problem = hc.Problem(Q, mu, lo, hi, K, rho)
    result = hc.prove(problem)                 # a bound certificate, confirmed by independent checkers
    result.status, result.bound                # Status.PROVED, a lower bound on min x'Qx
    result.certificate.save("cert.json")
    hc.verify("cert.json", method="c")         # anyone, later, with nothing but the file and a checker

A certificate is a short witness from which the bound is recomputed and checked with rigorous arithmetic; checking
it requires no trust in the prover. Packages:

* ``hc.models`` -- ``Problem`` and bit-reproducible risk models (factor, Ledoit-Wolf, factor form, sample);
* ``hc.certificates`` -- ``Certificate``, ``TreeCertificate``, ``Witness`` and their file formats;
* ``hc.prover`` -- ``prove``, ``prove_batch`` (GPU), ``prove_copositive``, witness extraction;
* ``hc.checkers`` -- the independent checkers: rigorous and exact (Python), C99, copositive, tree;
* ``hc.search`` -- primal heuristics and the certified branch-and-bound that proves optima;
* ``hc.extras`` -- certificates for mean-CVaR and lot-constrained index tracking.

Names are loaded on first use, so importing the package is cheap and pulls in no numerical backend.
"""
from heurics_cert._version import __version__

_EXPORTS = {
    "Problem": "heurics_cert.models.problem",
    "Certificate": "heurics_cert.certificates.certificate",
    "TreeCertificate": "heurics_cert.certificates.tree",
    "Witness": "heurics_cert.certificates.witness",
    "load": "heurics_cert.certificates.certificate",
    "prove": "heurics_cert.prover.pipeline",
    "Result": "heurics_cert.prover.result",
    "verify": "heurics_cert.checkers.api",
    "Verdict": "heurics_cert.checkers.api",
    "check_tree": "heurics_cert.checkers",
    "Status": "heurics_cert.status",
    "Params": "heurics_cert.params",
}
_SUBPACKAGES = ("models", "certificates", "prover", "checkers", "search", "extras", "errors")


def prove_batch(problems, **kwargs):
    """Certify many problems sharing one singular covariance, batched on the GPU (``jax`` extra)."""
    from heurics_cert.prover.batch import prove_batch as _prove_batch
    return _prove_batch(problems, **kwargs)


def prove_copositive(problem, **kwargs):
    """Improve a diagonal certificate with a copositive one (``jax`` extra)."""
    from heurics_cert.prover.copositive import prove_copositive as _prove_copositive
    return _prove_copositive(problem, **kwargs)


def __getattr__(name):
    import importlib
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    if name in _SUBPACKAGES:
        return importlib.import_module(f"heurics_cert.{name}")
    raise AttributeError(f"module 'heurics_cert' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_EXPORTS) + list(_SUBPACKAGES))


__all__ = ["__version__", "prove_batch", "prove_copositive", *_EXPORTS]
