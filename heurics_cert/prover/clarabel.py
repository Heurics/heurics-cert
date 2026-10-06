"""The perspective relaxation by an interior-point method (Clarabel, through CVXPY). Requires the ``reference``
extra. Kept as an independent cross-check of the default solver."""
from __future__ import annotations

import numpy as np

from heurics_cert.prover.relaxation import Relaxation


def solve(problem, d, *, scale: float | None = None, time_limit: float = 900.0) -> Relaxation | None:
    """Solve for split ``d``; None if the solver produced no point.

    The objective is divided by ``scale`` (default: the largest ``|Q_ij|``): these objectives are ~1e-6, and the
    solver's absolute tolerances would otherwise dominate them."""
    import cvxpy as cp
    Q, mu, lo, hi, K, rho = problem.Q, problem.mu, problem.lo, problem.hi, problem.K, problem.rho
    d = np.asarray(d, np.float64)
    s = float(scale) if scale else max(float(np.max(np.abs(Q))), 1e-300)
    n = problem.n
    A = (Q - np.diag(d)) / s
    A = 0.5 * (A + A.T)
    ev, U = np.linalg.eigh(A)
    keep = ev > 1e-12 * max(float(ev.max()), 1e-300)
    x = cp.Variable(n)
    y = cp.Variable(n)
    sc = 1.0 / max(float(np.abs(mu).max()), 1e-12)
    cons = [cp.sum(x) == 1.0, sc * (mu @ x) >= sc * rho, x >= cp.multiply(lo, y), x <= cp.multiply(hi, y),
            y >= 0.0, y <= 1.0, cp.sum(y) <= K]
    if keep.sum() <= n // 2:                   # low rank: a factor is far cheaper than a dense quadratic form
        obj = cp.sum_squares((U[:, keep] * np.sqrt(ev[keep])).T @ x)
    else:
        obj = cp.quad_form(x, cp.psd_wrap(A))
    idx = np.flatnonzero(d > 0)
    if idx.size:
        t = cp.Variable(idx.size)
        # x_i^2 <= t_i y_i  <=>  ||(2 x_i, t_i - y_i)|| <= t_i + y_i
        cons.append(cp.SOC(t + y[idx], cp.vstack([2.0 * x[idx], t - y[idx]]), axis=0))
        obj = obj + (d[idx] / s) @ t
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL, time_limit=time_limit, max_iter=500)
    except Exception:                                                          # noqa: BLE001
        return None
    if x.value is None:
        return None

    def dv(c):
        return 0.0 if c.dual_value is None else float(np.ravel(c.dual_value)[0])
    duals = dict(nu=dv(cons[0]) * s, pi=dv(cons[1]) * sc * s, lam=dv(cons[6]) * s)
    return Relaxation(x=np.asarray(x.value, np.float64), y=np.asarray(y.value, np.float64),
                      value=float(prob.value) * s, status=str(prob.status), duals=duals)
