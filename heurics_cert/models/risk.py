"""Risk models built bit-reproducibly from a returns matrix ``R`` (T x n): the same returns give the same model bytes,
and so the same fingerprint, on any machine (see ``heurics_cert.models._exact``).

* ``factor_model`` -- statistical factor model: top-k principal directions plus floored specific variance;
* ``ledoit_wolf`` -- Ledoit-Wolf shrinkage toward the identity (J. Multivariate Anal. 2004) or toward constant
  correlation (J. Portfolio Manag. 2004);
* ``factor_form`` -- ``X F X' + diag(spec)`` for given exposures, factor covariance and specific variances;
* ``sample_covariance`` -- the ddof=1 sample covariance.

Each returns the split the certificate uses alongside ``Q``: the model's own diagonal (``Q - diag(d)`` is PSD).
"""
from __future__ import annotations

import math

import numpy as np

from heurics_cert.errors import InvalidProblemError
from heurics_cert.models._exact import colsum, dot, dots, fsum_all, jacobi_eigh, outer_sum


def statistics(R):
    """``(mu, Xc)``: the mean return of each asset and the centred returns."""
    R = np.asarray(R, np.float64)
    mu = colsum(R) / R.shape[0]
    return mu, R - mu


def factor_model(R, kfac: int = 10, floor: float = 0.05):
    """Statistical factor model ``Q = B B' + diag(spec)``. Returns ``(mu, Q, spec)``; ``spec`` is the split.

    ``B`` holds the top ``kfac`` principal directions of the ddof=1 sample covariance scaled by sqrt(eigenvalue);
    ``spec`` is the residual variance, floored at ``floor`` of each asset's total. Computed through the T x T Gram
    matrix ``G = Xc Xc'``: if ``G v = g v`` then ``B_k = Xc' v / sqrt(T - 1)``."""
    mu, Xc = statistics(R)
    T, n = Xc.shape
    G = np.empty((T, T))
    for a in range(T):
        G[a, a:] = dots(Xc[a:], Xc[a])
        G[a:, a] = G[a, a:]
    _, V = jacobi_eigh(G)
    sq = math.sqrt(T - 1.0)
    B = np.stack([dots(Xc.T, V[:, k]) / sq for k in range(kfac)], axis=1)
    var = np.array([dot(Xc[:, i], Xc[:, i]) for i in range(n)]) / (T - 1.0)
    bb = np.array([dot(B[i], B[i]) for i in range(n)])
    spec = np.maximum(np.maximum(var - bb, 0.0), floor * var)
    Q = np.zeros((n, n))
    for k in range(kfac):                                  # fixed order, elementwise
        Q += np.multiply.outer(B[:, k], B[:, k])
    Q[np.diag_indices(n)] += spec
    return mu, Q, spec


def ledoit_wolf(R, target: str):
    """Ledoit-Wolf shrinkage of the 1/T sample covariance ``S``. Returns ``(Q, d, delta)``; ``d`` is the split.

    ``target="identity"``: ``Q = (1 - delta) S + delta m I``, ``d = delta m`` (J. Multivariate Anal. 88, 2004).
    ``target="cc"``: ``Q = (1 - delta) S + delta F`` with ``F = rbar s s' + (1 - rbar) diag(s^2)``,
    ``d = delta (1 - rbar) s^2`` (J. Portfolio Manag. 30(4), 2004); needs a mean correlation ``rbar >= 0``.
    ``delta`` is the published closed-form intensity, clipped to [0, 1]."""
    R = np.asarray(R, np.float64)
    T, n = R.shape
    _, X = statistics(R)
    S = outer_sum(X) / T
    var = np.diag(S).copy()
    if target == "identity":
        m = math.fsum(var.tolist()) / n
        E = S.copy()
        E[np.diag_indices(n)] -= m
        d2 = fsum_all(E * E) / n
        x2 = np.array([dot(X[t], X[t]) for t in range(T)])
        b2 = (math.fsum((x2 * x2).tolist()) / T - fsum_all(S * S)) / T / n
        delta = min(b2, d2) / d2
        Q = (1.0 - delta) * S
        Q[np.diag_indices(n)] += delta * m
        return Q, np.full(n, delta * m), delta
    if target == "cc":
        sd = np.sqrt(var)
        SD = np.multiply.outer(sd, sd)
        rbar = (fsum_all(S / SD) - n) / (n * (n - 1))
        if rbar < 0:
            raise InvalidProblemError(f"constant-correlation split needs mean correlation >= 0 (got {rbar:.3g})")
        F = rbar * SD
        F[np.diag_indices(n)] = var
        Y = X * X
        P = outer_sum(Y) / T - S * S
        phi = fsum_all(P)
        theta = outer_sum(Y * X, X) / T - var[:, None] * S
        theta[np.diag_indices(n)] = 0.0
        rho = math.fsum(np.diag(P).tolist()) + rbar * fsum_all((sd[None, :] / sd[:, None]) * theta)
        gamma = fsum_all((F - S) * (F - S))
        delta = max(0.0, min(1.0, (phi - rho) / gamma / T))
        Q = (1.0 - delta) * S + delta * F
        return Q, delta * (1.0 - rbar) * var, delta
    raise ValueError(f"unknown target {target!r}; expected 'identity' or 'cc'")


def factor_form(X, F, spec):
    """``X F X' + diag(spec)`` for exposures ``X`` (n x k), factor covariance ``F`` (k x k) and specific variances
    (the split), e.g. a published fundamental model."""
    X, F = np.asarray(X, np.float64), np.asarray(F, np.float64)
    k = X.shape[1]
    XF = np.zeros_like(X)
    for j in range(k):
        for i in range(k):
            XF[:, j] += X[:, i] * F[i, j]
    Q = np.zeros((X.shape[0], X.shape[0]))
    for j in range(k):
        Q += np.multiply.outer(XF[:, j], X[:, j])
    Q = 0.5 * (Q + Q.T)
    Q[np.diag_indices(X.shape[0])] += spec
    return Q


def sample_covariance(R):
    """``(mu, Q)``: the mean and the ddof=1 sample covariance. Singular when there are fewer observations than
    assets; its split is then ``d = 0``."""
    mu, Xc = statistics(R)
    return mu, outer_sum(Xc) / (Xc.shape[0] - 1.0)
