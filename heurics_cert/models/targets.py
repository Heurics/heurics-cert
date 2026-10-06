"""Return targets: ``rho`` as a quantile of the return range a cardinality-``K`` portfolio can reach."""
from __future__ import annotations

import math

import numpy as np

from heurics_cert.models._exact import dot


def extreme_return(mu, lo, hi, K: int, *, maximise: bool) -> float:
    """The largest (smallest) ``mu'x`` over ``{1'x = 1, lo_i y_i <= x_i <= hi_i y_i, 1'y <= K}``, in closed form:
    the ``K`` best assets by ``mu``, each at its floor, then the remaining budget to the best of them up to ``hi``."""
    sign = -1.0 if maximise else 1.0
    order = np.argsort(sign * mu, kind="stable")[:K]
    m, l, h = mu[order], lo[order], hi[order]
    inner = np.argsort(sign * m, kind="stable")
    cap = (h - l)[inner]
    room = np.cumsum(cap) - cap
    budget = 1.0 - math.fsum(l.tolist())
    take = np.clip(budget - room, 0.0, cap)
    x = l.copy()
    x[inner] += take
    return dot(m, x)


def return_target(mu, lo, hi, K: int, q: float) -> float:
    """``rho`` at quantile ``q`` of the return range a cardinality-``K`` portfolio can reach."""
    r_hi = extreme_return(mu, lo, hi, K, maximise=True)
    r_lo = extreme_return(mu, lo, hi, K, maximise=False)
    return float(r_lo + q * (r_hi - r_lo))
