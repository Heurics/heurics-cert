"""Provers: from a problem to a checked bound certificate.

    from heurics_cert.prover import prove
    result = prove(problem)                      # relaxation -> split ascent -> witness -> checkers
    result.status, result.bound, result.spread

* ``prove`` -- one problem (diagonal split, CCPO-CERT/1);
* ``prove_batch`` -- many problems on one singular covariance, batched on the GPU (``jax`` extra);
* ``prove_copositive`` -- a copositive (CCPO-CERT/2) improvement of a diagonal certificate (``jax`` extra);
* ``extract`` -- the witness at any point, for callers with their own relaxation solver.

No prover is trusted: every certificate is confirmed by the checkers in ``heurics_cert.checkers``.
"""
from heurics_cert.prover.pipeline import check, prove
from heurics_cert.prover.result import Result
from heurics_cert.prover.tangent import extract


def prove_batch(problems, **kwargs):
    """Certify many problems sharing one singular covariance in one batch (see ``heurics_cert.prover.batch``)."""
    from heurics_cert.prover.batch import prove_batch as _prove_batch
    return _prove_batch(problems, **kwargs)


def prove_copositive(problem, **kwargs):
    """Improve a diagonal certificate with a copositive one (see ``heurics_cert.prover.copositive``)."""
    from heurics_cert.prover.copositive import prove_copositive as _prove_copositive
    return _prove_copositive(problem, **kwargs)


__all__ = ["Result", "check", "extract", "prove", "prove_batch", "prove_copositive"]
