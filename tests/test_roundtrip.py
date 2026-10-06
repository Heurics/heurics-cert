"""The seam: the witness the prover emits satisfies the format the independent checkers check.

The prover (``heurics_cert.prover``) and the checkers (``heurics_cert.checkers``) share no code. A certificate from
one must be proved by the other on every tier, and both must fingerprint the model identically.
"""
import numpy as np
import pytest

import heurics_cert as hc
from heurics_cert.checkers import bound


def _problem(seed, n, K):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q = (A @ A.T) / n + np.eye(n)
    Q = 0.5 * (Q + Q.T)
    mu = rng.normal(scale=0.05, size=n)
    return hc.Problem(Q, mu, np.zeros(n), np.full(n, 0.6), K, float(np.quantile(mu, 0.3)))


def _params():
    try:
        import jax  # noqa: F401
        return hc.Params(check=())
    except ImportError:
        pytest.importorskip("cvxpy")
        return hc.Params(solver="clarabel", check=())


def test_prover_output_is_proved_by_every_checker_tier():
    p = _problem(0, 12, 4)
    doc = hc.prove(p, params=_params()).certificate.document()
    plain = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in doc.items()}
    assert bound.check_exact(plain)["verdict"] == "PROVED"
    assert bound.check_rigorous(doc)["verdict"] == "PROVED"


def test_fingerprints_agree_between_prover_and_checker_and_across_representations():
    """numpy arrays and JSON lists hash to the same digest; otherwise ``--expect-model`` would be a lie."""
    p = _problem(1, 10, 3)
    cert = hc.prove(p, params=_params()).certificate
    doc = cert.document()
    plain = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in doc.items()}
    assert doc["model_hash"] == p.fingerprint == bound.model_hash(plain) == bound.model_hash(doc)
