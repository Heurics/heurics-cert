"""``prove``: a problem in, a checked certificate out.

    splits -> relaxation per distinct split -> ascent on the best split -> one witness -> checkers

Only the winning split is certified: extraction dominates the cost and loses almost nothing at any split, so the best
relaxation gives the best certificate. The witness is retried up the margin ladder until every requested checker
proves it (a NOT PROVED at a thin margin is a precision event, not a verdict on the bound).
"""
from __future__ import annotations

import time

import numpy as np

from heurics_cert.certificates.certificate import Certificate
from heurics_cert.errors import SolverError
from heurics_cert.params import DEFAULT, Params
from heurics_cert.prover import relaxation, splits as splits_mod, tangent
from heurics_cert.prover.result import Result
from heurics_cert.checkers.api import verify


def _candidates(problem):
    """Candidate splits. On a clearly rank-deficient ``Q`` every diagonal split is ``d = 0``, so the O(n^3) search is
    skipped (only when the rank is at most n/2: skipping can cost tightness, never validity)."""
    r = splits_mod.numerical_rank_below(problem.Q)
    if r is not None and r <= problem.n // 2:
        return {f"zero (rank {r} < n)": np.zeros(problem.n)}
    return splits_mod.candidate_splits(problem.Q)


def check(certificate: Certificate, methods, *, qbin: str | None = None, timeout: float = 1800.0) -> dict:
    """Run each checker in ``methods`` on ``certificate``; ``qbin`` is used by the C checker."""
    return {m: verify(certificate, m, qbin=qbin if m == "c" else None, timeout=timeout) for m in methods}


def prove(problem, *, params: Params = DEFAULT, splits: dict | None = None, qbin: str | None = None) -> Result:
    """Certify a lower bound on ``problem``'s optimum.

    ``splits`` maps names to diagonal splits (default: computed from ``Q``; pass them when certifying many problems
    on one covariance). ``qbin`` lets the C checker read ``Q`` from a binary model file. See ``Params`` for the
    solver, the checkers and the effort; none of them affects soundness."""
    def solve(pr, d, pi_hint=None):
        return relaxation.solve(pr, d, params=params, pi_hint=pi_hint)

    t = {}
    t0 = time.perf_counter()
    cand = splits if splits is not None else _candidates(problem)
    t["splits_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    solved = {}
    for name, d in cand.items():
        if d is None or any(np.array_equal(np.asarray(d, np.float64), dd) for dd, _ in solved.values()):
            continue                           # on a singular Q every split is d = 0: solve it once
        r = solve(problem, d)
        if r is not None:
            solved[name] = (np.asarray(d, np.float64), r)
    if not solved:
        raise SolverError("the relaxation solver returned no point for any split")
    name = max(solved, key=lambda k: solved[k][1].value)
    d, relax = solved[name]
    t["solve_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if params.solver == "fista":                          # ascent steps warm-start the return multiplier
        def solve_asc(pr, dd):
            return solve(pr, dd, pi_hint=relax.duals["pi"])
    else:
        solve_asc = solve
    d, relax, accepted = splits_mod.ascend(problem, d, relax, params.ascent_steps, solve=solve_asc)
    t["ascent_s"] = time.perf_counter() - t0
    t["extract_s"] = t["check_s"] = 0.0
    points = [(relax.x, relax.duals)]
    if not np.any(d > 0):
        # d = 0: the relaxation is a QP on the capped simplex; its exact optimum usually gives a tighter tangent
        from heurics_cert.prover import active_set
        xr, nu_r, pi_r, it = active_set.solve(problem.Q, problem.mu, problem.hi, problem.rho, relax.x)
        if it < active_set.MAX_ITER and np.isfinite(xr).all():
            points.append((xr, dict(nu=-nu_r, pi=pi_r, lam=0.0)))
    for margin in params.margin_ladder:
        t0 = time.perf_counter()
        wit = max((tangent.extract(problem, d, x, du, margin) for x, du in points),
                  key=lambda w: w.claimed_bound)
        t["extract_s"] += time.perf_counter() - t0
        cert = Certificate(problem, wit)
        t0 = time.perf_counter()
        verdicts = check(cert, params.check, qbin=qbin, timeout=params.check_timeout)
        t["check_s"] += time.perf_counter() - t0
        if all(v.proved for v in verdicts.values()):
            break
    return Result(cert, verdicts, name + ("+ascent" if accepted else ""), relax.value,
                  relaxation_point=np.asarray(relax.x, np.float64), ascent_accepted=accepted, margin=margin, timings=t)
