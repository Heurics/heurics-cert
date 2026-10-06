"""Bit-reproducible arithmetic: results fixed by IEEE-754 alone, independent of BLAS, threads and CPU.

Every operation is an elementwise ufunc (one call each: no reassociation, no fused multiply-add), an error-free
product (Dekker's TwoProduct) summed by ``math.fsum`` (correctly rounded), or a fixed-order accumulation. Models built
from these give the same bytes, and so the same fingerprint, on any machine.
"""
from __future__ import annotations

import math

import numpy as np

_SPLIT = 134217729.0                                   # 2^27 + 1: Dekker's splitter for float64


def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """Elementwise `p + e == a * b` exactly (no overflow or underflow in our ranges)."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def dot(a, b) -> float:
    """Correctly rounded `sum(a * b)`."""
    p, e = two_prod(np.asarray(a, np.float64), np.asarray(b, np.float64))
    return math.fsum(np.concatenate([p, e]).tolist())


def dots(M, v):
    """Correctly rounded `M @ v` row by row."""
    p, e = two_prod(np.asarray(M, np.float64), np.asarray(v, np.float64)[None, :])
    return np.array([math.fsum(r) for r in np.concatenate([p, e], axis=1).tolist()])


def colsum(R):
    return np.array([math.fsum(c) for c in np.asarray(R, np.float64).T.tolist()])


def fsum_all(A) -> float:
    """Correctly rounded sum of every entry."""
    return math.fsum(np.asarray(A, np.float64).ravel().tolist())


def outer_sum(Y, Z=None):
    """``Y' Z`` (T x n each) as the sum of the T outer products ``y_t z_t'``, accumulated elementwise in t order:
    deterministic (not correctly rounded), and fast enough at n = 2,901 where a per-entry exact dot is not."""
    Y = np.asarray(Y, np.float64)
    Z = Y if Z is None else np.asarray(Z, np.float64)
    M = np.zeros((Y.shape[1], Z.shape[1]))
    for t in range(Y.shape[0]):
        M += np.multiply.outer(Y[t], Z[t])
    return M


def jacobi_eigh(G, *, sweeps: int = 60, tol: float = 1e-15):
    """Eigenpairs of a small symmetric matrix by cyclic Jacobi rotations, deterministic (fixed order, elementwise ops,
    stops after the first sweep whose largest off-diagonal entry is <= tol * max|diag|). Returns (w, V), w descending,
    V's columns unit eigenvectors."""
    A = np.array(G, np.float64)
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(sweeps):
        off = np.abs(A - np.diag(np.diag(A))).max()
        if off <= tol * np.abs(np.diag(A)).max():
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if apq == 0.0:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                ap, aq = A[:, p].copy(), A[:, q].copy()
                A[:, p] = c * ap - s * aq
                A[:, q] = s * ap + c * aq
                rp, rq = A[p, :].copy(), A[q, :].copy()
                A[p, :] = c * rp - s * rq
                A[q, :] = s * rp + c * rq
                A[p, q] = A[q, p] = 0.0
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
    w = np.diag(A).copy()
    order = np.argsort(-w, kind="stable")
    return w[order], V[:, order]
