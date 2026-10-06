"""Exact tangent point for the capped-simplex relaxation: a ratio-tested primal active set.

    min x'Qx   s.t.  1'x = 1,  mu'x >= rho,  0 <= x <= hi          (hi uniform; Q PSD, often singular)

The tangent is first-order sensitive to the sign pattern of ``2Qx + nu``, so KKT accuracy, not objective accuracy,
decides the bound. Every iterate stays feasible (a ratio test caps each step at the first bound or the return row it
would cross), so the objective is non-increasing and the method ends at the exact optimum of the convex QP (Nocedal &
Wright, Algorithm 16.3). Bounds are released in blocks after pricing all assets with one product.

On a singular ``Q`` the optimal set can be a flat face where reduced costs are zero only to rounding. Three guards
bound the cost: ``PRICE_TOL`` keeps rounding-level reduced costs from being released, ``FLAT_ITERS`` returns an
iterate that has stopped improving, and ``WORK_BUDGET`` caps the dense KKT work. A failure costs tightness only; the
iteration count is returned so the caller can fall back.

Multipliers: at the optimum ``2Qx = nu 1 + pi mu`` on the free set.
"""
from __future__ import annotations

import numpy as np

MAX_ITER = 400
#: pricing tolerance, relative to |nu|
PRICE_TOL = 1e-9
#: this many iterations without a relative decrease of FLAT_TOL: optimal to working precision
FLAT_ITERS, FLAT_TOL = 25, 1e-13
#: cost ceiling: the sum over iterations of |F|^3 (the KKT solves); ~2e10 is a couple of seconds on one core
WORK_BUDGET = 2e10


def cap_project(v, hi0):
    """Euclidean projection onto ``{0 <= x <= hi0, 1'x = 1}`` by bisection on the shift."""
    a, b = float(v.min()) - 1.0, float(v.max())
    for _ in range(200):
        m = 0.5 * (a + b)
        if np.clip(v - m, 0.0, hi0).sum() > 1.0:
            a = m
        else:
            b = m
    x = np.clip(v - 0.5 * (a + b), 0.0, hi0)
    return x / x.sum()


def _phase1(x0, mu, hi0, rho):
    """A feasible start near ``x0``: project onto the capped simplex, then mix toward the max-return vertex just
    enough to meet the return row (both ends satisfy the budget and the box, so every mix does)."""
    x = cap_project(x0, hi0)
    if mu @ x >= rho:
        return x
    xm = np.zeros(len(x))
    left = 1.0
    for j in np.argsort(-mu):
        take = min(hi0, left)
        xm[j] = take
        left -= take
        if left <= 0:
            break
    gap = float(mu @ xm - mu @ x)
    t = 1.0 if gap <= 0 else min(1.0, (rho - mu @ x) / gap * (1 + 1e-12))
    return (1 - t) * x + t * xm


def _eq_step(Q, F, x, ret_on, mu):
    """Newton step on the free set ``F`` and its multipliers. Least squares, because ``Q`` restricted to ``F`` is
    singular once ``|F|`` exceeds the covariance rank; the min-norm step keeps the iterate off the null space."""
    rows = [np.ones(F.size)]
    if ret_on:
        rows.append(mu[F])
    R = np.array(rows)
    m = R.shape[0]
    g = 2.0 * (Q[F] @ x)
    KKT = np.block([[2.0 * Q[np.ix_(F, F)], R.T], [R, np.zeros((m, m))]])
    try:
        sol = np.linalg.lstsq(KKT, np.concatenate([-g, np.zeros(m)]), rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None
    return sol[:F.size], -sol[F.size:]


def _obj(Q, x):
    """``x'Qx`` on ``x``'s support (O(k^2) for k nonzeros)."""
    S = np.flatnonzero(x)
    xs = x[S]
    return float(xs @ Q[np.ix_(S, S)] @ xs)


def solve(Q, mu, hi, rho, x0, *, max_iter=MAX_ITER, tol=1e-13):
    """Returns ``(x, nu, pi, iterations)``; ``iterations == max_iter`` means it did not converge."""
    hi0 = float(np.asarray(hi).ravel()[0])
    mu = np.asarray(mu, np.float64)
    x = _phase1(np.asarray(x0, np.float64), mu, hi0, rho)
    lower = x <= 1e-15
    upper = x >= hi0 - 1e-15
    ret_on = abs(mu @ x - rho) <= 1e-12 * max(1.0, abs(rho))
    work = 0.0
    best_obj, flat = _obj(Q, x), 0
    for it in range(max_iter):
        F = np.flatnonzero(~lower & ~upper)
        if F.size == 0:
            break
        work += float(F.size) ** 3
        if work > WORK_BUDGET:
            return x, 0.0, 0.0, max_iter
        p, lam = _eq_step(Q, F, x, ret_on, mu)
        if p is None:
            break
        if np.max(np.abs(p)) <= tol * max(1.0, float(np.max(np.abs(x)))):
            nu = float(lam[0])
            pi = float(lam[1]) if ret_on else 0.0
            if ret_on and pi < -1e-14:          # the return row is not really binding: release it
                ret_on = False
                continue
            r = 2.0 * (Q @ x) - nu - pi * mu    # reduced costs, all n priced by one matvec
            scale = max(abs(nu), 1e-300)
            rel = (lower & (r < -PRICE_TOL * scale)) | (upper & (r > PRICE_TOL * scale))
            if not rel.any():
                return x, nu, pi, it
            lower[rel] = False
            upper[rel] = False
            continue
        # ratio test: the largest step in [0, 1] that keeps 0 <= x <= hi and, if slack, the return row
        xF = x[F]
        a, block = 1.0, None
        neg, pos = p < 0, p > 0
        if neg.any():
            s = -xF[neg] / p[neg]
            k = int(np.argmin(s))
            if s[k] < a:
                a, block = float(s[k]), ("lo", F[np.flatnonzero(neg)[k]])
        if pos.any():
            s = (hi0 - xF[pos]) / p[pos]
            k = int(np.argmin(s))
            if s[k] < a:
                a, block = float(s[k]), ("hi", F[np.flatnonzero(pos)[k]])
        if not ret_on:
            dret = float(mu[F] @ p)
            if dret < 0:
                s = float(mu @ x - rho) / -dret
                if s < a:
                    a, block = s, ("ret", None)
        x = x.copy()
        x[F] = xF + max(a, 0.0) * p
        if block is not None:
            kind, j = block
            if kind == "lo":
                x[j] = 0.0
                lower[j] = True
            elif kind == "hi":
                x[j] = hi0
                upper[j] = True
            else:
                ret_on = True
        obj = _obj(Q, x)
        if obj < best_obj * (1.0 - FLAT_TOL):
            best_obj, flat = obj, 0
        else:
            flat += 1
            if flat >= FLAT_ITERS:
                # optimal to working precision on a flat face: return it as converged, with its multipliers
                Fz = np.flatnonzero(~lower & ~upper)
                if Fz.size:
                    _, lam = _eq_step(Q, Fz, x, ret_on, mu)
                    if lam is not None:
                        return x, float(lam[0]), (float(lam[1]) if ret_on else 0.0), it
    F = np.flatnonzero(~lower & ~upper)
    if F.size:
        _, lam = _eq_step(Q, F, x, ret_on, mu)
        if lam is not None:
            return x, float(lam[0]), (float(lam[1]) if ret_on else 0.0), max_iter
    return x, 0.0, 0.0, max_iter
