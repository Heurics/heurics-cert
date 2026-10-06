"""Feasible portfolios of our own: round the relaxation to a support, solve the restricted QP exactly, swap.

A certificate is a statement about the model, but the gap it is quoted against needs a feasible point, and a gap is
only as good as that point. This gives the certifier its own: if rounding the relaxation already closes a gap, the
bound was never the problem. The restricted QP respects the buy-in bounds; a drop-and-repair route (``lo`` relaxed to
0, then assets below it removed) is kept as a second candidate. Nothing here is trusted: results are validated with
``Problem.feasibility`` like any other candidate.
"""
from __future__ import annotations

import time

import numpy as np

from heurics_cert.prover import active_set


def project_box_simplex(v, lo, hi):
    """Euclidean projection onto {lo <= x <= hi, 1'x = 1} by bisection on the shift."""
    a, b = float(np.min(v - hi)), float(np.max(v - lo))
    for _ in range(100):
        t = 0.5 * (a + b)
        if np.clip(v - t, lo, hi).sum() > 1.0:
            a = t
        else:
            b = t
    x = np.clip(v - 0.5 * (a + b), lo, hi)
    free = (x > lo) & (x < hi)
    if free.any():
        x[free] += (1.0 - x.sum()) / free.sum()
    return x


def max_return_point(mu, lo, hi):
    x, left = lo.copy(), 1.0 - float(lo.sum())
    for j in np.argsort(-mu):
        add = min(hi[j] - lo[j], left)
        x[j] += add
        left -= add
    return x


def box_qp_approx(Q, mu, lo, hi, rho, iters=300, deadline=np.inf):
    """Approximate ``min x'Qx`` over ``{lo <= x <= hi, 1'x = 1, mu'x >= rho}``: FISTA on ``x'Qx - pi mu'x`` with
    bisection on ``pi``. A point meeting the return row, or None when no point of the box-simplex does."""
    k = len(mu)
    if lo.sum() > 1 + 1e-12 or hi.sum() < 1 - 1e-12:
        return None
    xm = max_return_point(mu, lo, hi)
    if mu @ xm < rho - 1e-12 * max(1.0, abs(rho)):
        return None
    L = 2.0 * float(np.linalg.eigvalsh(Q)[-1]) + 1e-300

    def argmin(pi, x0):
        x, y, t = x0.copy(), x0.copy(), 1.0
        for _ in range(iters):
            xn = project_box_simplex(y - (2.0 * (Q @ y) - pi * mu) / L, lo, hi)
            tn = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
            y = xn + (t - 1) / tn * (xn - x)
            x, t = xn, tn
        return x

    x = argmin(0.0, project_box_simplex(np.full(k, 1.0 / k), lo, hi))
    if mu @ x >= rho:
        return x
    p_lo, p_hi, xf = 0.0, 1e-3 * L / max(float(np.abs(mu).max()), 1e-12), xm
    for _ in range(60):
        if time.perf_counter() > deadline:
            return xf
        xt = argmin(p_hi, x)
        if mu @ xt >= rho:
            xf = xt
            break
        p_lo, p_hi = p_hi, 2.0 * p_hi
    for _ in range(40):
        if time.perf_counter() > deadline:
            break
        pm = 0.5 * (p_lo + p_hi)
        xt = argmin(pm, xf)
        if mu @ xt >= rho:
            p_hi, xf = pm, xt
        else:
            p_lo = pm
    return xf


def box_qp_exact(Q, mu, lo, hi, rho, x, max_iter=200, multipliers=False):
    """Primal-dual active set for ``min x'Qx`` over ``{lo <= x <= hi, 1'x = 1, mu'x >= rho}``, warm-started from
    the bound pattern of ``x``. Each step solves the KKT system on the free weights, fixes weights that leave their
    box, and releases bound weights whose reduced cost ``2(Qx)_i + nu - pi mu_i`` has the wrong sign. Returns the
    optimum, or None when it does not settle; with ``multipliers=True``, ``(x, nu, pi)``."""
    state = np.where(x <= lo * (1 + 1e-10), -1, np.where(x >= hi * (1 - 1e-10), 1, 0))
    ret = True
    scale = max(float(np.abs(np.diag(Q)).max()), 1e-300)
    for _ in range(max_iter):
        F = np.flatnonzero(state == 0)
        B = np.flatnonzero(state != 0)
        xB = np.where(state[B] < 0, lo[B], hi[B])
        m = F.size + 1 + ret
        M = np.zeros((m, m))
        r = np.zeros(m)
        M[:F.size, :F.size] = 2.0 * Q[np.ix_(F, F)]
        M[:F.size, F.size] = 1.0
        M[F.size, :F.size] = 1.0
        r[:F.size] = -2.0 * Q[np.ix_(F, B)] @ xB
        r[F.size] = 1.0 - xB.sum()
        if ret:
            M[:F.size, F.size + 1] = -mu[F]
            M[F.size + 1, :F.size] = mu[F]
            r[F.size + 1] = rho - mu[B] @ xB
        try:
            z = np.linalg.solve(M, r)
        except np.linalg.LinAlgError:
            return None
        xn = np.empty_like(x)
        xn[F], xn[B] = z[:F.size], xB
        nu = z[F.size]
        pi = z[F.size + 1] if ret else 0.0
        if ret and pi < 0:                                    # return row not binding: release it
            ret = False
            continue
        if not ret and mu @ xn < rho - 1e-14 * max(1.0, abs(rho)):
            ret = True                                        # released too early: bind it again
            continue
        out_lo = F[xn[F] < lo[F] - 1e-14]
        out_hi = F[xn[F] > hi[F] + 1e-14]
        if out_lo.size or out_hi.size:                        # fix the violators at the bound they cross
            state[out_lo] = -1
            state[out_hi] = 1
            continue
        red = 2.0 * (Q @ xn) + nu - pi * mu
        tol = 1e-12 * scale
        rel_lo = B[(state[B] < 0) & (red[B] < -tol)]
        rel_hi = B[(state[B] > 0) & (red[B] > tol)]
        if not rel_lo.size and not rel_hi.size:
            return (xn, float(nu), float(pi)) if multipliers else xn
        worst = max(list(rel_lo), key=lambda i: -red[i], default=None)     # release one at a time: no cycling
        worst_hi = max(list(rel_hi), key=lambda i: red[i], default=None)
        if worst is None or (worst_hi is not None and red[worst_hi] > -red[worst]):
            worst = worst_hi
        state[worst] = 0
    return None


def _restricted_lo(problem, S, x0=None, deadline=np.inf):
    """``min x'Qx`` on the support ``S`` with ``lo_i <= x_i <= hi_i`` there. Returns ``(x, objective)`` or
    ``(None, inf)``: the active set from ``x0`` first, FISTA then the active set if that does not settle."""
    S = np.asarray(sorted(S), dtype=np.int64)
    if S.size == 0:
        return None, np.inf
    QS, muS, loS, hiS = problem.Q[np.ix_(S, S)], problem.mu[S], problem.lo[S], problem.hi[S]
    rho_tol = 1e-12 * max(1.0, abs(problem.rho))
    if loS.sum() > 1 + 1e-12 or hiS.sum() < 1 - 1e-12 or mu_max(muS, loS, hiS) < problem.rho - rho_tol:
        return None, np.inf
    xs = None
    if x0 is not None:
        xs = box_qp_exact(QS, muS, loS, hiS, problem.rho, project_box_simplex(np.asarray(x0, np.float64)[S], loS, hiS))
    if xs is None:
        xf = box_qp_approx(QS, muS, loS, hiS, problem.rho, deadline=deadline)
        if xf is None:
            return None, np.inf
        xs = box_qp_exact(QS, muS, loS, hiS, problem.rho, xf)
        xs = xf if xs is None else xs
    x = np.zeros(problem.n)
    x[S] = xs
    return x, problem.objective(x)


def mu_max(mu, lo, hi):
    """Largest return any point of {lo <= x <= hi, 1'x = 1} reaches (the greedy fill)."""
    return float(mu @ max_return_point(mu, lo, hi))


def _restricted_drop(problem, S, x0=None, deadline=np.inf):
    """Drop and repair: the QP on ``S`` with ``lo`` relaxed to 0, then assets below ``lo`` leave and it is re-solved
    (every asset under half its ``lo`` at once, otherwise the smallest)."""
    Q, mu, lo, hi, rho = problem.Q, problem.mu, problem.lo, problem.hi, problem.rho
    S = np.asarray(sorted(S), dtype=np.int64)
    while S.size and time.perf_counter() < deadline:
        if float(hi[S[0]]) * S.size < 1.0 - 1e-12:        # the caps cannot hold the budget
            return None, np.inf
        start = x0[S] if x0 is not None else np.full(S.size, 1.0 / S.size)
        xs, _, _, it = active_set.solve(Q[np.ix_(S, S)], mu[S], hi[S], rho, start + 1e-12)
        if not np.isfinite(xs).all() or float(mu[S] @ xs) < rho - 1e-12 * max(1.0, abs(rho)):
            return None, np.inf
        low = (xs > 0) & (xs < lo[S] * (1 - 1e-12))
        if not low.any():
            x = np.zeros(problem.n)
            x[S] = np.where(xs > 0, xs, 0.0)
            x /= x.sum()
            return x, problem.objective(x)
        dust = low & (xs < 0.5 * lo[S])
        S = S[~dust] if dust.any() else S[~(low & (xs == xs[low].min()))]
    return None, np.inf


def restricted_qp(problem, S, x0=None, deadline=np.inf):
    """The better feasible result of the buy-in-respecting QP on ``S`` and drop-and-repair. Returns
    ``(x, objective)`` or ``(None, inf)``."""
    best = (None, np.inf)
    for x, f in (_restricted_lo(problem, S, x0, deadline), _restricted_drop(problem, S, x0, deadline)):
        if x is not None and problem.feasibility(x)["feasible"] and f < best[1]:
            best = (x, f)
    return best


def polish(problem, x_start, *, time_limit: float = 5.0, pool: int = 24, max_rounds: int = 200):
    """Round ``x_start`` (typically the relaxation point) to its top-K support and improve it by local moves:
    dropping a held asset, adding one of the ``pool`` best entrants by reduced cost, or swapping one for a held
    asset, first improvement taken. Returns ``(x, objective, info)``; ``x`` is None when no feasible support was
    found."""
    t0 = time.perf_counter()
    dl = t0 + time_limit
    K = problem.K
    x_start = np.asarray(x_start, np.float64)
    order = np.argsort(-x_start)
    held = np.flatnonzero(x_start > 0)
    first = [set(int(i) for i in order[:K])]
    if 0 < held.size <= K:                                  # already a support: also try it as is
        first.append(set(held.tolist()))
    x, f = None, np.inf
    for S in first:
        xt, ft = restricted_qp(problem, S, x_start, dl)
        if ft < f:
            x, f = xt, ft
    k = K
    while x is None and k < problem.n and time.perf_counter() < dl:   # rounding infeasible (return row): widen the pool
        k = min(problem.n, 2 * k)
        cand = order[:k]
        best = cand[np.argsort(-problem.mu[cand])][:K]
        x, f = restricted_qp(problem, set(int(i) for i in best), None, dl)
    if x is None:
        return None, np.inf, {"rounds": 0, "swaps": 0, "seconds": time.perf_counter() - t0}
    swaps = rounds = 0
    Q, mu, lo = problem.Q, problem.mu, problem.lo
    while rounds < max_rounds and time.perf_counter() < dl:
        rounds += 1
        S = np.flatnonzero(x > 0)
        g = 2.0 * (Q @ x)
        # multipliers of the restricted QP from its support: least squares on 2Qx = nu + pi mu
        R = np.stack([np.ones(S.size), mu[S]], axis=1)
        (nu, pi), *_ = np.linalg.lstsq(R, g[S], rcond=None)
        pi = max(pi, 0.0)
        red = g - nu - pi * mu
        red_out = red.copy()
        red_out[S] = np.inf
        entrants = np.argsort(red_out)[:pool]
        at_lo = S[x[S] <= lo[S] * (1 + 1e-9)]
        leavers = list(at_lo[np.argsort(-red[at_lo])][:4]) + list(S[np.argsort(x[S])][:4])
        leavers = list(dict.fromkeys(int(j) for j in leavers))
        Sset = set(S.tolist())
        trials = [Sset - {j} for j in leavers]
        if S.size < K:
            trials += [Sset | {int(i)} for i in entrants]
        trials += [(Sset - {j}) | {int(i)} for i in entrants for j in leavers]
        improved = False
        for T in trials:
            if time.perf_counter() >= dl:
                break
            xt, ft = restricted_qp(problem, T, x, dl)
            if ft < f * (1 - 1e-12):
                x, f, improved = xt, ft, True
                break
        if not improved:
            break
        swaps += 1
    return x, f, {"rounds": rounds, "swaps": swaps, "seconds": time.perf_counter() - t0}
