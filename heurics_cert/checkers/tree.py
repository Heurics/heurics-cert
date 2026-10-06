"""Checker for CCPO-TREE/0 search-tree certificates (docs/TREE_CERTIFICATE.md).

Shares no code with the search (``heurics_cert.search``); it re-derives every quantity from the record.

Trusted: the model and the root witness's split ``d``, whose ``A = Q - diag(d)`` the bound checker proves PSD when it
proves the root certificate. Everything in the record is untrusted:

structure
    every internal node has exactly two children on one asset (sides 1 and 0), the asset is not already fixed on its
    path, and every node is internal or terminal, never both -- so the terminals partition all supports;
sets
    each node's IN/OUT is rebuilt from the parent chain;
bound
    for a ``bound`` terminal, ``at`` must be an ancestor-or-self; ``w = 2 A x_hat`` and ``beta = -x_hat'A x_hat`` are
    recomputed from the recorded point, and the node bound
    ``beta - nu + pi rho - lam (K - |IN|) + sum_IN m_i + sum_FREE min(m_i + lam, 0)`` is evaluated as a rigorous
    lower bound (a-priori error bounds, ``gamma_k = k u / (1 - k u)``);
card / support
    ``|IN| > K``; or ``IN u OUT`` covers all assets and IN cannot meet the budget or the return row (exact rationals);
incumbent
    feasible with scale-relative tolerances; its objective bounded above rigorously.

Verdict: ``OPTIMAL`` (certified gap <= the reader's ``rel_tol``), ``BOUND`` (a certified lower bound only),
``REFUTED`` (malformed or false), ``NOT PROVED`` (the root certificate was not proved).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

FORMAT = "CCPO-TREE/0"
U = 2.0 ** -53


def _gamma(k):
    """``gamma_k = k u / (1 - k u)``: relative error bound of a k-operation float64 computation."""
    return k * U / (1 - k * U)


class Refuted(Exception):
    """The record is malformed or its claim is false; the message names the failed check."""


def _sets(nodes, split, n_assets):
    """Every node's IN/OUT from the parent chain, checking the tree's shape on the way."""
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise Refuted("the record has no nodes: nothing covers the root")
    IN, OUT, children = [None] * n_nodes, [None] * n_nodes, {}
    for i, (p, var, side) in enumerate(nodes):
        if i == 0:
            if p != -1:
                raise Refuted("node 0 is not the root")
            IN[0], OUT[0] = frozenset(), frozenset()
            continue
        if not (0 <= p < i):
            raise Refuted(f"node {i}: parent {p} not earlier in the record")
        if side not in (0, 1):
            raise Refuted(f"node {i}: side {side}")
        if not (0 <= var < n_assets):
            raise Refuted(f"node {i}: asset {var} is not one of the model's {n_assets}")
        if split.get(p) != var:
            raise Refuted(f"node {i}: parent {p} does not split on {var}")
        if var in IN[p] or var in OUT[p]:
            raise Refuted(f"node {i}: asset {var} already fixed on its path")
        IN[i] = IN[p] | {var} if side == 1 else IN[p]
        OUT[i] = OUT[p] | {var} if side == 0 else OUT[p]
        children.setdefault(p, []).append((side, i))
    for p, ch in children.items():
        if sorted(s for s, _ in ch) != [0, 1]:
            raise Refuted(f"node {p}: children sides {ch}")
    return IN, OUT, children


def _tangent(P, d, xS_idx, xS_val):
    """``w = 2 A x_hat`` with a-priori error radii, and a rigorous lower bound on ``beta = -x_hat'A x_hat``.

    The recorded point must be a point: distinct in-range integer indices, one value each. A repeated index would
    make ``w`` and ``beta`` come from different vectors."""
    S = np.asarray(xS_idx)
    xs = np.asarray(xS_val, np.float64)
    if S.ndim != 1 or xs.shape != S.shape or (S.size and not np.issubdtype(S.dtype, np.integer)):
        raise Refuted("a tangent point needs one integer index per value")
    S = S.astype(np.int64)
    if np.any(S < 0) or np.any(S >= P.n) or np.unique(S).size != S.size:
        raise Refuted("a tangent point's indices must be distinct and in range")
    if not np.all(np.isfinite(xs)):
        raise Refuted("non-finite tangent point")
    xfull = np.zeros(P.n)
    xfull[S] = xs
    QS = P.Q[:, S]
    y = QS @ xs - d * xfull                                   # (A x)
    ey = _gamma(S.size + 2) * (np.abs(QS) @ np.abs(xs) + np.abs(d * xfull))
    p = float(xs @ y[S])
    ep = _gamma(S.size) * float(np.abs(xs) @ np.abs(y[S])) + float(np.abs(xs) @ ey[S])
    beta_lo = -(p + ep) - 2 * U * abs(p + ep)
    return 2.0 * y, 2.0 * ey, beta_lo


def _lower_bound(P, d, tan, INs, OUTs, nu, pi, lam, lo0):
    """A rigorous lower bound on the node bound for held set INs and excluded set OUTs at ``(nu, pi, lam)``, with the
    precomputed tangent ``tan = (w, error radii of w, beta_lo)``. ``+inf`` when IN alone exceeds K."""
    n = P.n
    IN = np.array(sorted(INs), np.int64)
    mask = np.ones(n, bool)
    mask[IN] = False
    mask[list(OUTs)] = False
    FREE = np.flatnonzero(mask)
    if pi < 0 or lam < 0 or not all(np.isfinite([nu, pi, lam])):
        raise Refuted("multipliers out of domain")
    slots = P.K - IN.size
    if slots < 0:
        return np.inf
    wf, ewf, beta_lo = tan
    J = np.concatenate([IN, FREE])
    w, ew = wf[J], ewf[J]
    c = w + nu - pi * P.mu[J]
    ec = ew + _gamma(3) * (np.abs(w) + abs(nu) + np.abs(pi * P.mu[J]))
    cl = c - ec
    cl = cl - 2 * U * np.abs(cl)                              # a rigorous lower bound on c
    dJ = d[J]
    a = np.zeros(J.size) if lo0 else P.lo[J]
    b = P.hi[J]
    if np.any(a < 0):
        raise Refuted("negative lower bound on a weight")

    def gl(s):                                                 # lower bound of d s^2 + cl s at s >= 0
        v = dJ * s * s + cl * s
        return v - _gamma(3) * (np.abs(dJ) * s * s + np.abs(cl) * s)
    ga, gb = gl(a), gl(b)
    m = np.minimum(ga, gb)                                     # d <= 0: the minimum is at an endpoint
    pos_d = dJ > 0
    if pos_d.any():
        dd = np.where(pos_d, dJ, 1.0)
        v = -cl / (2 * dd)
        q = cl * cl / (4 * dd)
        uncon = -q * (1 + _gamma(3))                           # the unconstrained minimum: always a lower bound
        slack = 1e-9 * (np.abs(a) + np.abs(b)) + 1e-300
        mm = np.where(v < a - slack, ga, np.where(v > b + slack, gb, uncon))
        m = np.where(pos_d, mm, m)
    mi, mf = m[: IN.size], m[IN.size:]
    t = mf + lam
    t = t - U * np.abs(t)
    terms = np.concatenate([[beta_lo, -nu, pi * P.rho, -lam * slots], mi, np.minimum(t, 0.0)])
    total = float(np.sum(terms))
    return total - _gamma(terms.size + 2) * float(np.sum(np.abs(terms)))


def _support_infeasible(P, INs):
    """True when no weights on exactly the support INs meet the budget, box and return rows (exact rationals: the
    greedy fill by return is the support's maximum-return point)."""
    idx = sorted(INs)
    lo = [Fraction(float(P.lo[i])) for i in idx]
    hi = [Fraction(float(P.hi[i])) for i in idx]
    mu = [Fraction(float(P.mu[i])) for i in idx]
    if sum(lo) > 1 or sum(hi) < 1:
        return True
    x, left = list(lo), 1 - sum(lo)
    for k in sorted(range(len(idx)), key=lambda k: -mu[k]):
        add = min(hi[k] - lo[k], left)
        x[k] += add
        left -= add
    return sum(m * xi for m, xi in zip(mu, x)) < Fraction(float(P.rho))


def _incumbent(P, inc):
    """A rigorous upper bound on the incumbent's objective, or None if it is not a feasible portfolio (indices must
    be distinct and in range, or one asset could be held above its cap across several entries)."""
    S = np.asarray(inc["idx"], np.int64)
    x = np.asarray(inc["val"], np.float64)
    if S.shape != x.shape or S.size == 0 or S.size > P.K or np.unique(S).size != S.size:
        return None
    if np.any(S < 0) or np.any(S >= P.n) or not np.all(np.isfinite(x)) or np.any(x < 0):
        return None
    tol = 1e-9
    if abs(float(x.sum()) - 1.0) > tol or np.any(x < P.lo[S] * (1 - 1e-12)) or np.any(x > P.hi[S] * (1 + 1e-12)):
        return None
    if float(P.mu[S] @ x) < P.rho - tol * max(1.0, float(np.abs(P.mu).max())):
        return None
    QS = P.Q[np.ix_(S, S)]
    f = float(x @ QS @ x)
    return f + _gamma(S.size + 1) * float(np.abs(x) @ np.abs(QS) @ np.abs(x))


def check_tree(problem, witness, record, *, method: str = "rigorous", rel_tol: float = 1e-6) -> dict:
    """The full check: prove the root certificate (``A = Q - diag(d)`` PSD for the witness's split), bind the record
    to the model and to that split, then ``check``. ``problem`` is the model the reader holds (anything with
    ``.fields()``, ``.Q``, ``.mu``, ...), ``witness`` the root witness (``.d``, ``.fields()``), ``rel_tol`` the
    reader's acceptance tolerance for OPTIMAL."""
    from heurics_cert.checkers import bound
    from heurics_cert.checkers.api import verify
    if record.get("format") != FORMAT:
        return dict(verdict="REFUTED", reason=f"format {record.get('format')!r} is not {FORMAT}")
    model = problem.fields()
    if record.get("model_sha256") != bound.model_hash(model):
        return dict(verdict="REFUTED", reason="the record is about a different model (SHA-256 mismatch)")
    d = np.asarray(witness.d, np.float64)
    if "root_d" in record and not np.array_equal(np.asarray(record["root_d"], np.float64), d):
        return dict(verdict="REFUTED", reason="the record's split differs from the root witness's")
    root = verify(dict(model, **witness.fields()), method).proved
    return check(problem, d, record, root, rel_tol=rel_tol)


def check(P, d, record, root_proved: bool, *, rel_tol: float = 1e-6) -> dict:
    """The tree check proper, given whether the root split ``d`` is proved.

    ``rel_tol`` is the reader's tolerance; the record's own ``rel_tol`` (the search's stopping rule) plays no part.
    Returns ``dict(verdict, LB, U, gap_pct, terminals, reason, rel_tol)``."""
    if not root_proved:
        return dict(verdict="NOT PROVED", reason="root certificate not proved (A = Q - diag(d) PSD unestablished)")
    d = np.asarray(d, np.float64)
    try:
        nodes = record["nodes"]
        split = {int(k): int(v) for k, v in record["split"].items()}
        term = {int(k): v for k, v in record["term"].items()}
        IN, OUT, children = _sets(nodes, split, P.n)
        stray = [k for k in term if not 0 <= k < len(nodes)]
        if stray:
            raise Refuted(f"terminal proofs for nodes {stray[:5]} that are not in the record")
        for i in range(len(nodes)):
            internal, terminal = i in children, i in term
            if internal == terminal:
                what = "both internal and terminal" if internal else "neither internal nor terminal"
                raise Refuted(f"node {i}: {what}")
            if internal and i not in split:
                raise Refuted(f"node {i}: children without a split record")
        LB, cache = np.inf, {}
        for i, pr in term.items():
            if pr["kind"] == "card":
                if len(IN[i]) <= P.K:
                    raise Refuted(f"node {i}: 'card' but |IN| = {len(IN[i])} <= K")
                continue
            if pr["kind"] == "support":
                if len(IN[i]) + len(OUT[i]) != P.n or not _support_infeasible(P, IN[i]):
                    raise Refuted(f"node {i}: 'support' proof does not hold")
                continue
            if pr["kind"] != "bound":
                raise Refuted(f"node {i}: unknown proof kind")
            a = int(pr["at"])
            # refused here, not by the walk: the walk ends at -1 (the root's parent), and IN[-1] is the last node's
            if not 0 <= a < len(nodes):
                raise Refuted(f"node {i}: proof node {a} is not in the record")
            k = i
            while k != -1 and k != a:
                k = nodes[k][0]
            if k != a:
                raise Refuted(f"node {i}: proof node {a} is not an ancestor")
            ti = int(pr["t"])
            if not 0 <= ti < len(record["tangents"]):
                raise Refuted(f"node {i}: tangent {ti} missing")
            if ti not in cache:
                cache[ti] = _tangent(P, d, *record["tangents"][ti])
            LB = min(LB, _lower_bound(P, d, cache[ti], IN[a], OUT[a], pr["nu"], pr["pi"], pr["lam"], pr["lo0"]))
    except Refuted as exc:
        return dict(verdict="REFUTED", reason=str(exc))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return dict(verdict="REFUTED", reason=f"malformed record: {type(exc).__name__}: {exc}")
    try:
        inc = record.get("incumbent")
        Uc = _incumbent(P, inc) if inc and inc.get("idx") else None
    except (KeyError, IndexError, TypeError, ValueError):
        Uc = None
    if Uc is None:
        return dict(verdict="BOUND", LB=LB, U=None, gap_pct=None, terminals=len(term), reason="no valid incumbent")
    gap = (Uc - LB) / Uc
    tol = float(rel_tol)
    ok = gap <= tol * (1 + 1e-6)
    return dict(verdict="OPTIMAL" if ok else "BOUND", LB=LB, U=Uc, gap_pct=100 * gap, terminals=len(term), reason="",
                rel_tol=tol)
