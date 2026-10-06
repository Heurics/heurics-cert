"""Diagonal splits ``Q = A + diag(d)`` with ``A >= 0``: the one free choice in the perspective relaxation.

Every split gives a valid relaxation; they differ in tightness only. Candidates depend on ``Q`` alone, so callers
certifying many problems on one covariance should compute ``candidate_splits`` once and pass them in. On a singular
``Q`` every split is ``d = 0``.

``ascend`` improves the best candidate per problem: the relaxation value ``R(d)`` is a minimum of functions affine in
``d``, hence concave, with supergradient ``g_i = x_i^2/y_i - x_i^2`` at the relaxation's minimiser ``(x, y)``.
"""
from __future__ import annotations

import numpy as np


def psd_guard(Q, d, target_rel: float = 1e-10):
    """The largest nonnegative split at or below ``d`` with ``lambda_min(Q - diag(d)) >= target`` (a uniform downward
    shift first, bisection on a scale factor otherwise). A singular ``Q`` with ``d = 0`` is returned as is."""
    q_scale = float(np.max(np.sum(np.abs(Q), axis=1)))
    target = target_rel * q_scale
    d = np.maximum(np.asarray(d, np.float64), 0.0)
    lam = float(np.linalg.eigvalsh(Q - np.diag(d))[0])
    if lam >= target or not np.any(d > 0):
        return d
    if np.all(d + lam - target >= 0.0):
        return d + lam - target
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        if np.linalg.eigvalsh(Q - np.diag(t * d))[0] >= target:
            lo = t
        else:
            hi = t
    return lo * d


def direction_split(Q, direction):
    """The largest ``t * direction`` with ``Q - diag(t * direction)`` PSD (a generalized eigenvalue)."""
    w = np.maximum(np.asarray(direction, np.float64), 1e-12 * float(np.max(np.diag(Q))))
    s = 1.0 / np.sqrt(w)
    S = Q * s[:, None] * s[None, :]
    t = max(float(np.linalg.eigvalsh(0.5 * (S + S.T))[0]), 0.0)
    return psd_guard(Q, t * w)


def pca_direction(Q, k: int | None = None):
    """Residual variance after a k-factor principal-component fit."""
    n = Q.shape[0]
    k = k or max(1, min(10, n // 10))
    ev, U = np.linalg.eigh(Q)
    B = U[:, -k:] * np.sqrt(np.maximum(ev[-k:], 0.0))
    return np.maximum(np.diag(Q) - np.sum(B * B, axis=1), 0.0)


def numerical_rank_below(Q, k: int = 256, seed: int = 0, rtol: float = 1e-11):
    """``rank(Q)`` if it is below ``k``, else None: a randomized range test (one n x k product, no eigensolve). A
    sample covariance from T observations has rank at most T - 1."""
    n = Q.shape[0]
    if n <= k:
        return None
    rng = np.random.default_rng(seed)
    Y = Q @ rng.standard_normal((n, k))
    s = np.linalg.svd(Y, compute_uv=False)
    r = int(np.sum(s > s[0] * rtol))
    return r if r < k else None


def candidate_splits(Q) -> dict[str, np.ndarray]:
    """The ``Q``-only candidates: ``uniform`` (``lambda_min I``), ``diag`` (scaled diagonal), ``factor`` (principal-
    component residual). O(n^3) each; cache per covariance."""
    Q = np.asarray(Q, np.float64)
    ev = float(np.linalg.eigvalsh(Q)[0])
    return {
        "uniform": psd_guard(Q, np.full(Q.shape[0], max(ev, 0.0))),
        "diag": direction_split(Q, np.diag(Q)),
        "factor": direction_split(Q, pca_direction(Q)),
    }


def ascend(problem, d, relax, steps: int = 8, *, solve, min_gain: float = 1e-6):
    """Supergradient ascent on the split from ``(d, relax)``. Returns ``(d, relax, accepted_steps)``.

    Only improving steps are accepted (the result is never below the start); the step grows 1.5x on success and
    halves on failure, and the ascent ends when a step's predicted gain falls below ``min_gain`` of the value.
    ``solve(problem, d)`` returns a ``Relaxation`` or None."""
    Q = problem.Q
    if not np.any(np.asarray(d) > 0):
        return d, relax, 0                     # singular: nothing to ascend
    step = 0.25 * float(np.mean(np.diag(Q)))
    accepted = 0
    for _ in range(steps):
        x, y = relax.x, relax.y
        g = np.where(x > 1e-10, x * x / np.maximum(y, 1e-12) - x * x, 0.0)
        gmax = float(np.abs(g).max())
        if gmax <= 0.0 or step * float(np.sum(g * g)) / gmax < min_gain * abs(relax.value):
            break
        dc = psd_guard(Q, d + step * g / gmax)
        rc = solve(problem, dc)
        if rc is not None and rc.value > relax.value:
            d, relax = dc, rc
            step *= 1.5
            accepted += 1
        else:
            step *= 0.5
    return d, relax, accepted
