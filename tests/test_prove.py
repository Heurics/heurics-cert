"""The prover end to end, on problems small enough to solve exactly by enumeration.

Validity is tested against ground truth, not against another bound: every certificate must be proved by the checkers
and sit at or below the true optimum found by enumerating every support of size <= K.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

import heurics_cert as hc                                  # noqa: E402
from heurics_cert.checkers import qbin                     # noqa: E402

C_CHECKER = hc.checkers.c_checker()
CHECKS = ("rigorous", "exact") + (("c",) if C_CHECKER else ())
try:
    import jax  # noqa: F401
    SOLVER = "fista"
except ImportError:                                        # without JAX, exercise the pipeline on Clarabel
    SOLVER = "clarabel"


def _problem(n, K, rank=None, seed=0, hi=1.0, lo=0.01):
    rng = np.random.default_rng(seed)
    r = rank or n + 5
    B = rng.standard_normal((r, n)) * 0.1
    Q = B.T @ B / r
    Q = 0.5 * (Q + Q.T)
    mu = rng.normal(0.01, 0.02, n)
    rho = float(np.quantile(mu, 0.4))
    return hc.Problem(Q=Q, mu=mu, lo=np.full(n, lo), hi=np.full(n, hi), K=K, rho=rho, name=f"syn{n}k{K}r{rank}")


def _brute_force(p) -> float:
    best = np.inf
    for k in range(1, p.K + 1):
        for S in itertools.combinations(range(p.n), k):
            S = list(S)
            x = cp.Variable(k)
            Qs = 0.5 * (p.Q[np.ix_(S, S)] + p.Q[np.ix_(S, S)].T)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(Qs))),
                              [cp.sum(x) == 1, p.mu[S] @ x >= p.rho, x >= p.lo[S], x <= p.hi[S]])
            try:
                prob.solve(solver=cp.CLARABEL)
            except Exception:                                                  # noqa: BLE001
                continue
            if prob.status == "optimal":
                best = min(best, prob.value)
    return best


def _params(*checks):
    return hc.Params(solver=SOLVER, check=checks or ("rigorous",))


@pytest.mark.parametrize("n,K,rank", [(8, 2, None), (9, 3, None), (10, 3, 4)])
def test_bound_is_proved_and_below_the_optimum(n, K, rank):
    p = _problem(n, K, rank)
    res = hc.prove(p, params=_params(*CHECKS))
    assert res.status is hc.Status.PROVED, {k: v.status for k, v in res.verdicts.items()}
    opt = _brute_force(p)
    assert res.bound <= opt * (1 + 1e-9), (res.bound, opt)
    assert res.bound > 0.0


def test_singular_covariance_gives_a_positive_bound():
    p = _problem(12, 3, rank=5, seed=3)
    assert float(np.linalg.eigvalsh(p.Q)[0]) < 1e-12
    res = hc.prove(p, params=_params())
    assert res.proved and res.bound > 0.0
    assert res.split.startswith(("uniform", "diag", "factor"))


def test_tampered_claim_is_not_proved():
    p = _problem(8, 2)
    res = hc.prove(p, params=_params())
    w = res.certificate.witness
    forged = hc.Certificate(p, type(w)(**{**w.__dict__, "claimed_bound": w.claimed_bound * 1.5 + 1e-6}))
    assert not hc.verify(forged, "rigorous").proved


def test_model_substitution_is_detected():
    p = _problem(8, 2)
    res = hc.prove(p, params=_params())
    other = hc.Problem(p.Q, p.mu, p.lo, p.hi, p.K, p.rho * 1.01)
    assert other.fingerprint != p.fingerprint
    v = hc.verify(res.certificate, "rigorous", expect_model=other.fingerprint[:16])
    assert v.status is hc.Status.MODEL_MISMATCH


def test_roundtrip_self_contained_and_split(tmp_path):
    p = _problem(9, 3)
    res = hc.prove(p, params=_params())
    one = os.path.join(tmp_path, "c.json")
    res.certificate.save(one)
    split, qb = os.path.join(tmp_path, "w.json"), os.path.join(tmp_path, "m.qbin")
    res.certificate.save(split, qbin=qb)
    for c in (hc.load(one), hc.load(split, qbin=qb)):
        assert c.fingerprint == p.fingerprint
        assert c.bound == res.bound
        assert hc.verify(c, "rigorous").proved
    assert hc.verify(one).proved and hc.verify(split, qbin=qb).proved      # straight from the files
    assert np.array_equal(qbin.read(qb), p.Q)
    assert os.path.getsize(split) < os.path.getsize(one)


@pytest.mark.skipif(C_CHECKER is None, reason="ccpocheck not built for this platform")
def test_c_checker_reads_the_split_layout(tmp_path):
    p = _problem(9, 3)
    res = hc.prove(p, params=_params())
    qb = os.path.join(tmp_path, "m.qbin")
    assert hc.verify(res.certificate, "c", qbin=qb).proved
    assert hc.verify(res.certificate, "c").proved


def test_gap_rejects_an_infeasible_candidate():
    p = _problem(8, 2)
    res = hc.prove(p, params=_params())
    x = np.zeros(p.n)
    x[:3] = 1.0 / 3.0                                       # three names against K = 2
    g = res.gap(x)
    assert not g["feasible"] and np.isnan(g["gap"])


def test_spread_is_reported():
    res = hc.prove(_problem(9, 3), params=_params())
    assert res.spread is not None and -1e-9 <= res.spread <= 1.0


@pytest.mark.skipif(SOLVER != "fista", reason="needs jax")
@pytest.mark.parametrize("n,K,rank", [(9, 3, None), (12, 3, 5)])
def test_fista_matches_the_interior_point_reference(n, K, rank):
    """The two solvers propose different points; both witnesses must prove, and FISTA's must come within a small
    tolerance of Clarabel's (tightness, not validity, is what the solver decides)."""
    p = _problem(n, K, rank, seed=7)
    ours = hc.prove(p, params=hc.Params(solver="fista", check=("rigorous",)))
    ref = hc.prove(p, params=hc.Params(solver="clarabel", check=("rigorous",)))
    assert ours.proved and ref.proved
    assert ours.bound >= ref.bound - 1e-3 * abs(ref.bound)
