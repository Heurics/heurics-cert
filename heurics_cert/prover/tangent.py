"""Witness extraction: any point ``x_hat`` becomes a bound certificate.

For a split ``Q = A + diag(d)`` with ``A`` positive definite, the tangent of ``x'Ax`` at ``x_hat`` is a global
underestimator, ``x'Ax >= w'x + beta_z`` with ``w = 2 A x_hat`` and ``beta_z = -x_hat'A x_hat``. What remains is
separable per asset, so the dual is closed form in three scalars: ``lam`` exactly given ``(nu, pi)``, and ``(nu, pi)``
maximised numerically. Every ``(x_hat, nu, pi)`` gives a valid bound; the point and the search decide only how tight.
``w`` lies in ``range(A)`` by construction, so a singular ``Q`` needs no pseudo-inverse.
"""
from __future__ import annotations

import numpy as np

from heurics_cert.certificates.witness import Witness
from heurics_cert.params import DEFAULT

U = 2.0 ** -53
MARGIN_LADDER = DEFAULT.margin_ladder
#: above this size ``lambda_min`` for the margin comes from Lanczos rather than a full eigendecomposition
LANCZOS_MIN_N = 400


def per_asset_min(d, c, lo, hi):
    """``min_{s in {lo, hi, vertex}} d s^2 + c s`` per asset (the vertex only where ``d > 0``)."""
    safe = np.where(d > 0, d, 1.0)
    vert = np.where(d > 0, np.clip(-c / (2.0 * safe), lo, hi), lo)
    cands = np.stack([lo, hi, vert])
    return np.min(d * cands * cands + c * cands, axis=0)


def dual_value(w, nu, pi, beta_z, d, mu, lo, hi, K, rho):
    """The certificate's dual at ``(nu, pi)`` with ``lam`` at its exact maximiser. Returns ``(value, lam)``."""
    m = per_asset_min(d, w + nu - pi * mu, lo, hi)
    lam = max(0.0, -float(np.partition(m, K)[K])) if K < len(m) else 0.0
    return beta_z - nu + pi * rho - lam * K + float(np.sum(np.minimum(m + lam, 0.0))), lam


def ternary_max(g, a, b, it=80):
    """Maximiser of a concave scalar function on ``[a, b]`` by ternary search."""
    for _ in range(it):
        p, q = a + (b - a) / 3, b - (b - a) / 3
        if g(p) < g(q):
            a = p
        else:
            b = q
    return 0.5 * (a + b)


def maximize_scalars(w, beta_z, d, mu, lo, hi, K, rho, nu_span, pi_span, start=(0.0, 0.0), shrink_after=3):
    """Maximise the concave, non-smooth dual over ``(nu, pi >= 0)``: ternary search along both axes and both
    diagonals with shrinking brackets. Returns ``(value, nu, pi)``; every point visited is a valid bound."""
    def f(nu, pi):
        return dual_value(w, nu, max(pi, 0.0), beta_z, d, mu, lo, hi, K, rho)[0]
    nu, pi = float(start[0]), max(float(start[1]), 0.0)
    best = (f(nu, pi), nu, pi)
    rn, rp = nu_span, pi_span
    for r in range(24):
        nu = ternary_max(lambda t: f(t, pi), nu - rn, nu + rn)
        pi = max(ternary_max(lambda t: f(nu, t), max(pi - rp, 0.0), pi + rp), 0.0)
        for sgn in (1.0, -1.0):
            tt = ternary_max(lambda t: f(nu + t * rn, max(pi + sgn * t * rp, 0.0)), -1.0, 1.0)
            nu, pi = nu + tt * rn, max(pi + sgn * tt * rp, 0.0)
        v = f(nu, pi)
        if v > best[0]:
            best = (v, nu, pi)
        if r >= shrink_after:
            rn, rp = 0.5 * rn, 0.5 * rp
    return best


def smallest_eigenvalue(A, *, n=None, zero_split=False, rounding=0.0):
    """``lambda_min(A)``, used only to size the margin in ``extract`` (a low estimate costs tightness, never validity).

    With ``zero_split`` (``d = 0``) the bound ``-rounding`` is known without a solve, and an iterative solver must not
    be used: the bottom of a sample covariance's spectrum is a large cluster near zero where Lanczos stalls. Small
    ``n`` uses a dense ``eigvalsh``; otherwise capped Lanczos with an ``eigvalsh`` fallback."""
    n = A.shape[0] if n is None else n
    if zero_split:
        return -float(rounding)
    if n < LANCZOS_MIN_N:
        return float(np.linalg.eigvalsh(A)[0])
    try:
        from scipy.sparse.linalg import ArpackNoConvergence, eigsh
    except ImportError:
        return float(np.linalg.eigvalsh(A)[0])
    try:
        v = float(eigsh(A, k=1, which="SA", tol=1e-7, maxiter=600, return_eigenvectors=False)[0])
    except ArpackNoConvergence:
        return float(np.linalg.eigvalsh(A)[0])
    scale = float(np.max(np.abs(np.diag(A)))) if n else 0.0
    return v - 1e-8 * max(abs(v), scale)


def extract(problem, d_split, x_hat, duals=None, margin: float = MARGIN_LADDER[0]) -> Witness:
    """The tangent witness at ``x_hat`` for split ``d_split``.

    The split is backed off by ``margin`` multiples of the rigorous checker's Cholesky error scale
    ``(n + 2) u trace(Q)``, so that ``Q - diag(d)`` is positive definite by an amount a verified Cholesky can see,
    and ``beta_z`` by a matching slack. ``duals`` (the relaxation's KKT multipliers) seed the scalar search; both
    sign conventions of the budget multiplier are tried, and a blind search from zero is kept."""
    Q, mu, lo, hi, K, rho = problem.Q, problem.mu, problem.lo, problem.hi, problem.K, problem.rho
    n = problem.n
    err_scale = (n + 2) * U * float(np.sum(np.abs(np.diag(Q))))
    d = np.asarray(d_split, np.float64) - margin * err_scale
    A = 0.5 * ((Q - np.diag(d)) + (Q - np.diag(d)).T)
    zero_split = bool(np.max(np.abs(np.asarray(d_split, np.float64)))
                      <= 1e-12 * max(float(np.max(np.abs(np.diag(Q)))), 1e-300))
    # with d = 0, A = Q + margin*err*I, and lambda_min(A) >= (margin - 1)*err covers a covariance's rounding
    lmin = (margin * err_scale - err_scale) if zero_split else smallest_eigenvalue(A, n=n)
    floor = margin * err_scale
    if lmin < floor:
        d, A = d - (floor - lmin), A + (floor - lmin) * np.eye(n)
    x_hat = np.asarray(x_hat, np.float64)
    Ax = A @ x_hat
    w = 2.0 * Ax
    z = -float(x_hat @ Ax)
    beta_z = z - (1e-12 * max(abs(z), 1e-300) + margin * (n + 2) * U * float(np.sum(np.abs(np.diag(A)))))
    scale = float(np.max(np.abs(w))) + 2.0 * float(np.max(np.abs(d))) * max(float(np.max(hi)), 1.0)
    nu_span = 2.0 * max(scale, 1e-12)
    pi_span = nu_span / max(float(np.max(np.abs(mu))), 1e-12)
    runs = [maximize_scalars(w, beta_z, d, mu, lo, hi, K, rho, nu_span, pi_span)]
    if duals is not None:
        for sgn in (1.0, -1.0):
            st = (sgn * duals["nu"], abs(duals["pi"]))
            runs.append((dual_value(w, st[0], st[1], beta_z, d, mu, lo, hi, K, rho)[0],) + st)
            runs.append(maximize_scalars(w, beta_z, d, mu, lo, hi, K, rho, 1e-3 * nu_span, 1e-3 * pi_span,
                                         start=st, shrink_after=0))
    _, nu, pi = max(runs, key=lambda r: r[0])
    val, lam = dual_value(w, nu, pi, beta_z, d, mu, lo, hi, K, rho)
    return Witness(d=d, w=w, nu=nu, pi=pi, lam=lam, beta_z=beta_z, claimed_bound=val - 1e-9 * abs(val))


def tangent_witness(problem, d, x_hat, margin) -> Witness:
    """The witness at ``x_hat`` on an already-margined split ``d`` (no eigenvalue shift): used to rebuild a root
    witness on a recorded split. Valid once a checker proves it."""
    n = problem.n
    A = 0.5 * ((problem.Q - np.diag(d)) + (problem.Q - np.diag(d)).T)
    Ax = A @ x_hat
    w = 2.0 * Ax
    z = -float(x_hat @ Ax)
    beta_z = z - (1e-12 * max(abs(z), 1e-300) + margin * (n + 2) * U * float(np.sum(np.abs(np.diag(A)))))
    scale = float(np.max(np.abs(w))) + 2.0 * float(np.max(np.abs(d))) * max(float(np.max(problem.hi)), 1.0)
    nu_span = 2.0 * max(scale, 1e-12)
    pi_span = nu_span / max(float(np.max(np.abs(problem.mu))), 1e-12)
    args = (d, problem.mu, problem.lo, problem.hi, problem.K, problem.rho)
    _, nu, pi = maximize_scalars(w, beta_z, *args, nu_span, pi_span)
    val, lam = dual_value(w, nu, pi, beta_z, *args)
    return Witness(d=d, w=w, nu=nu, pi=pi, lam=lam, beta_z=beta_z, claimed_bound=val - 1e-9 * abs(val))
