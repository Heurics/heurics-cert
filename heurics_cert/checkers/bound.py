"""Checker for CCPO-CERT/1 bound certificates. Standalone: stdlib, plus numpy for the rigorous tier.

    python -m heurics_cert.checkers.bound certificate.json [--exact] [--expect-model PREFIX]

A certificate claims a lower bound on

    min x'Qx  s.t.  1'x = 1,  mu'x >= rho,  lo_i y_i <= x_i <= hi_i y_i,  1'y <= K,  y binary

and carries a witness ``(d, w, nu, pi, lam, beta_z)``. The checker re-derives the bound from the certificate's own
numbers and takes nothing on trust:

* ``pi, lam >= 0`` (weak duality needs them);
* ``A = Q - diag(d)`` is positive semidefinite (otherwise ``min_z z'Az - w'z = -inf``);
* ``beta_z`` is a lower bound on that inner minimum;
* each per-asset term is the minimum over the whole interval, not over sampled points.

Two tiers:

``check_exact``
    ``fractions.Fraction`` throughout, stdlib only. Every float is a dyadic rational, so a rational ``LDL'`` proves
    or refutes ``A >= 0`` with no tolerance and solves ``A y = w`` exactly. Decides every case; cost grows like
    ``n^4`` (0.2 s at n = 40, 7 s at 100, 170 s at 200).

``check_rigorous``
    float64 with outward rounding. Positive semidefiniteness by a verified Cholesky (a factorization of ``A - cI``
    whose backward error is bounded rigorously); every later step rounds downward, so the result is a certified
    underestimate of the dual value. 17 ms at n = 400.

Verdicts: ``PROVED``; ``REFUTED`` (a definite violation, a certified direction of negative curvature, or a claim
above a rigorous upper bound on what the witness supports); ``NOT PROVED`` (nothing refuted, something undecided).
A verified Cholesky fails at degeneracy by construction, so a valid certificate with zero slack is ``NOT PROVED``
on the rigorous tier, never ``REFUTED``. The rigorous tier never contradicts the exact one.

Every verdict carries ``model_hash``, a SHA-256 over the model fields: a certificate can be self-consistent about a
more constrained problem than the reader meant, and only a fingerprint the reader recomputes catches that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from fractions import Fraction

# numpy is imported lazily: the exact tier and the fingerprint run on a bare CPython.
np = None


def _numpy():
    global np
    if np is None:
        import numpy as _np
        np = _np
    return np


U = 2.0 ** -53          # float64 unit roundoff

REQUIRED = ("Q", "mu", "lo", "hi", "K", "rho", "d", "w", "nu", "pi", "lam", "beta_z", "claimed_bound")
#: the fields that define the problem, in hashing order
MODEL_FIELDS = ("Q", "mu", "lo", "hi", "K", "rho")


def model_hash(cert: dict) -> str:
    """SHA-256 of the model fields over their IEEE-754 big-endian bytes (exact; independent of the JSON text).

    Arrays hash as ``(rows, cols)`` followed by the row-major values; numpy arrays and nested lists give the same
    digest."""
    h = hashlib.sha256()
    for name in MODEL_FIELDS:
        v = cert[name]
        h.update(name.encode())
        if isinstance(v, (int, float)):
            h.update(struct.pack(">d", float(v)))
            continue
        if type(v).__module__ == "numpy" and getattr(v, "ndim", 0) in (1, 2):
            rows = v if v.ndim == 2 else v.reshape(1, -1)
            h.update(struct.pack(">2q", rows.shape[0], rows.shape[1]))
            h.update(rows.astype(">f8", copy=False).tobytes(order="C"))
            continue
        rows = v if v and isinstance(v[0], list) else [v]
        h.update(struct.pack(">2q", len(rows), len(rows[0])))
        for row in rows:
            h.update(struct.pack(f">{len(row)}d", *(float(x) for x in row)))
    return h.hexdigest()


_SCALARS = ("K", "rho", "nu", "pi", "lam", "beta_z", "claimed_bound")
_VECTORS = ("mu", "lo", "hi", "d", "w")


def _is_real(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_cardinality(v) -> bool:
    """``K`` is a positive integer, carried as an int or an integral float within 2^53."""
    return _is_real(v) and v == math.floor(v) and 1 <= v <= 2 ** 53


def _malformed_plain(cert: dict) -> str | None:
    """Why ``cert`` is not well formed, or None (stdlib only, for the exact tier)."""
    missing = [k for k in REQUIRED if k not in cert]
    if missing:
        return f"missing fields {missing}"
    plain = {k: (cert[k].tolist() if hasattr(cert[k], "tolist") else cert[k]) for k in REQUIRED}
    Q = plain["Q"]
    if not isinstance(Q, list) or not Q or not all(isinstance(r, list) and len(r) == len(Q) for r in Q):
        return "Q is not a non-empty square matrix"
    n = len(Q)
    if not all(_is_real(v) for r in Q for v in r):
        return "Q has an entry that is not a finite real number"
    for k in _VECTORS:
        v = plain[k]
        if not isinstance(v, list) or len(v) != n or not all(_is_real(x) for x in v):
            return f"{k} is not a vector of {n} finite real numbers"
    for k in _SCALARS:
        if not _is_real(plain[k]):
            return f"{k} is not a finite real number"
    if not _is_cardinality(plain["K"]):
        return "K is not a positive integer"
    return None


def _malformed_numeric(cert: dict) -> str | None:
    """``_malformed_plain``, vectorised for the rigorous tier."""
    np = _numpy()
    missing = [k for k in REQUIRED if k not in cert]
    if missing:
        return f"missing fields {missing}"
    try:
        Q = np.asarray(cert["Q"], dtype=np.float64)
        vecs = {k: np.asarray(cert[k], dtype=np.float64) for k in _VECTORS}
        scal = {k: cert[k] for k in _SCALARS}
    except (TypeError, ValueError):
        return "a field is not numeric or not rectangular"
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] == 0 or not np.all(np.isfinite(Q)):
        return "Q is not a non-empty square matrix of finite reals"
    n = Q.shape[0]
    for k, v in vecs.items():
        if v.shape != (n,) or not np.all(np.isfinite(v)):
            return f"{k} is not a vector of {n} finite real numbers"
    for k, v in scal.items():
        if isinstance(v, np.generic):
            v = v.item()
        if not _is_real(v):
            return f"{k} is not a finite real number"
        if k == "K" and not _is_cardinality(v):
            return "K is not a positive integer"
    return None


def _refuse(cert: dict, tier: str, why: str) -> dict:
    """The verdict for malformed input: REFUTED with a reason (a checker answers every input, never raises)."""
    try:
        mh = model_hash(cert)
    except Exception:                                                  # noqa: BLE001 -- the input is malformed
        mh = "unavailable (malformed model)"
    out = {"tier": tier, "n": 0, "model_hash": mh, "certified_bound": None, "shortfall": float("inf"),
           "checks": [{"name": "the certificate is well formed", "ok": False, "status": "refuted", "detail": why,
                       "reason": f"malformed certificate: {why}"}]}
    out["status"], out["verdict"] = "REFUTED", f"REFUTED: malformed certificate: {why}"
    return out


def load(path: str) -> dict:
    with open(path) as fh:
        cert = json.load(fh)
    missing = [k for k in REQUIRED if k not in cert]
    if missing:
        raise ValueError(f"certificate is missing {missing}")
    return cert


# -- exact tier ---------------------------------------------------------------------------------------------------

def _fl(q) -> float:
    """``float(q)`` saturating to +-inf; only report text, never a verdict, is computed in floats here."""
    try:
        return float(q)
    except OverflowError:
        return math.inf if q > 0 else -math.inf


def _rational_ldl(A: list[list[Fraction]]):
    """Exact ``LDL'`` of a symmetric rational matrix, natural order. Returns ``(L, D, ok, zero_rows | reason)``.

    A negative pivot proves ``A`` is not PSD. A zero pivot is admissible only when the rest of its column is zero
    (the singular PSD case); for a PSD matrix natural-order elimination never meets a nonzero entry under a zero
    pivot, so no pivoting is needed."""
    n = len(A)
    L = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    D = [Fraction(0)] * n
    M = [row[:] for row in A]
    zero_rows = []
    for j in range(n):
        piv = M[j][j]
        if piv < 0:
            return L, D, False, f"negative pivot {_fl(piv):.3e} at index {j}"
        L[j][j] = Fraction(1)
        if piv == 0:
            for i in range(j + 1, n):
                if M[i][j] != 0:
                    return L, D, False, f"zero pivot at {j} with nonzero entry at ({i}, {j})"
            zero_rows.append(j)
            continue
        D[j] = piv
        for i in range(j + 1, n):
            L[i][j] = M[i][j] / piv
        for i in range(j + 1, n):
            lij = L[i][j]
            if lij == 0:
                continue
            for k in range(j + 1, i + 1):
                M[i][k] -= lij * M[j][k]
            for k in range(j + 1, i + 1):
                M[k][i] = M[i][k]
    return L, D, True, zero_rows


def _ldl_solve(L, D, zero_rows, b):
    """Solve ``A y = b`` in the range of ``A`` exactly. Returns ``(y, in_range)``; ``b`` with a component on
    ``null(A)`` makes ``min_z z'Az - b'z = -inf``."""
    n = len(b)
    zr = set(zero_rows)
    u = [Fraction(0)] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            if L[i][k]:
                s -= L[i][k] * u[k]
        u[i] = s
    v = [Fraction(0)] * n
    for i in range(n):
        if i in zr:
            if u[i] != 0:
                return None, False
            v[i] = Fraction(0)
        else:
            v[i] = u[i] / D[i]
    y = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = v[i]
        for k in range(i + 1, n):
            if L[k][i]:
                s -= L[k][i] * y[k]
        y[i] = s
    return y, True


def _boxed_min_exact(d: Fraction, c: Fraction, lo: Fraction, hi: Fraction) -> Fraction:
    """Exact ``min_{s in [lo, hi]} d s^2 + c s`` for any sign of ``d``: an endpoint, or the clipped vertex."""
    def q(s):
        return d * s * s + c * s

    best = min(q(lo), q(hi))
    if d > 0:
        vert = -c / (2 * d)
        if lo < vert < hi:
            best = min(best, q(vert))
    return best


def check_exact(cert: dict) -> dict:
    """The whole chain in exact rational arithmetic (stdlib only). Two-valued: PROVED or REFUTED."""
    why = _malformed_plain(cert)
    if why:
        return _refuse(cert, "exact", why)
    F = Fraction
    Q = [[F(v) for v in row] for row in cert["Q"]]
    n = len(Q)
    mu = [F(v) for v in cert["mu"]]
    lo = [F(v) for v in cert["lo"]]
    hi = [F(v) for v in cert["hi"]]
    d = [F(v) for v in cert["d"]]
    w = [F(v) for v in cert["w"]]
    nu, pi, lam = F(cert["nu"]), F(cert["pi"]), F(cert["lam"])
    rho, K = F(cert["rho"]), F(cert["K"])
    claimed = F(cert["claimed_bound"])

    out = {"tier": "exact", "n": n, "checks": [], "model_hash": model_hash(cert)}

    def add(name, ok, detail=""):
        out["checks"].append({"name": name, "ok": bool(ok), "status": "ok" if ok else "refuted",
                              "detail": detail, "reason": name})
        return ok

    ok_sym = all(Q[i][j] == Q[j][i] for i in range(n) for j in range(i))
    add("Q is exactly symmetric", ok_sym)
    if "model_hash" in cert:
        add("the carried model fingerprint matches the model carried with it",
            cert["model_hash"] == out["model_hash"],
            f"carried {str(cert['model_hash'])[:16]}, recomputed {out['model_hash'][:16]}")
    add("pi >= 0", pi >= 0, f"pi = {_fl(pi):.6e}")
    add("lam >= 0", lam >= 0, f"lam = {_fl(lam):.6e}")
    # d may be negative: the per-asset minimum is exact for either sign, and the PSD test below accounts for it.
    add("d is finite", all(v == v for v in d), f"min d = {_fl(min(d)):.6e}")
    add("lo <= hi", all(lo[i] <= hi[i] for i in range(n)))

    A = [[Q[i][j] - (d[i] if i == j else F(0)) for j in range(n)] for i in range(n)]
    L, D, psd, info = _rational_ldl(A)
    if not add("A = Q - diag(d) is positive semidefinite", psd, "" if psd else str(info)):
        out["certified_bound"] = None
        out["shortfall"] = float("inf")
        out["status"] = "REFUTED"
        out["verdict"] = "REFUTED: the split leaves a matrix that is not PSD, so the z-block "\
                         "minimum is -inf and the published bound is not a bound"
        return out
    zero_rows = info if isinstance(info, list) else []
    add("A is singular on some directions", True,
        f"{len(zero_rows)} zero pivots" if zero_rows else "A is positive definite")

    y, in_range = _ldl_solve(L, D, zero_rows, w)
    if not add("w lies in range(A)", in_range,
               "" if in_range else "w has a component on null(A): min_z z'Az - w'z = -inf"):
        out["certified_bound"] = None
        out["shortfall"] = float("inf")
        out["status"] = "REFUTED"
        out["verdict"] = "REFUTED: w is outside range(A)"
        return out

    beta_star = -F(1, 4) * sum((w[i] * y[i] for i in range(n)), F(0))     # min_z z'Az - w'z = -w'A^+w / 4
    beta_pub = F(cert["beta_z"])
    add("published beta_z <= the exact z-block minimum", beta_pub <= beta_star,
        f"beta_z = {_fl(beta_pub):.12g}, exact = {_fl(beta_star):.12g}, "
        f"slack = {_fl(beta_star - beta_pub):.3e}")

    per = F(0)
    for i in range(n):
        c = w[i] + nu - pi * mu[i]
        m = _boxed_min_exact(d[i], c, lo[i], hi[i])
        per += min(F(0), m + lam)

    # the certified value uses the exact z-block minimum; beta_z is judged separately above
    certified = beta_star - nu + pi * rho - lam * K + per
    certified_pub = beta_pub - nu + pi * rho - lam * K + per
    out["certified_bound"] = _fl(certified)
    out["certified_bound_at_published_beta"] = _fl(certified_pub)
    out["claimed_bound"] = _fl(claimed)
    out["shortfall"] = _fl(claimed - certified)
    add("claimed bound <= exactly re-derived bound", claimed <= certified,
        f"claimed = {_fl(claimed):.12g}, exact = {_fl(certified):.12g}, "
        f"shortfall = {_fl(claimed - certified):.3e}")
    out["verdict"] = ("PROVED" if all(c["ok"] for c in out["checks"])
                      else "REFUTED")
    out["status"] = out["verdict"]
    return out


# -- rigorous float tier ------------------------------------------------------------------------------------------

def _down(x):
    """One ulp below: a rigorous underestimate of a correctly rounded result (error <= ulp/2)."""
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


# Interval arithmetic with one ulp of outward widening per operation. numpy has no rounding-mode control, so each
# operation rounds to nearest and is widened on both sides; operations are nested one at a time, because a compound
# expression rounded once at the end is not covered by a single ulp.

def _iv(x):
    x = np.asarray(x, dtype=np.float64)
    return (x, x)


def _iadd(a, b):
    return (np.nextafter(a[0] + b[0], -np.inf), np.nextafter(a[1] + b[1], np.inf))


def _isub(a, b):
    return _iadd(a, (-b[1], -b[0]))


def _imul(a, b):
    prods = np.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return (np.nextafter(np.min(prods, axis=0), -np.inf),
            np.nextafter(np.max(prods, axis=0), np.inf))


def _idiv_pos(a, b):
    """``a / b`` for a degenerate interval ``b`` with strictly positive endpoints."""
    quots = np.stack([a[0] / b[0], a[1] / b[0]])
    return (np.nextafter(np.min(quots, axis=0), -np.inf),
            np.nextafter(np.max(quots, axis=0), np.inf))


def verified_psd(A) -> tuple[bool, float]:
    """Prove ``A >= 0`` in float64, or fail to. Returns ``(proved, lower bound on lambda_min)``.

    If the float Cholesky of a symmetric ``B`` completes, ``R'R = B + E`` with
    ``||E||_2 <= gamma_{n+1} || |R| ||_F^2`` (Higham, Accuracy and Stability, Thm 10.3-10.5), so
    ``lambda_min(B) >= -||E||_2``. Factor ``B = A - cI``; success gives ``lambda_min(A) >= c - delta_B``. Every
    quantity feeding the conclusion is rounded outward."""
    np = _numpy()
    n = A.shape[0]
    if not np.array_equal(A, A.T):
        return False, float("nan")

    def _delta(B):
        try:
            R = np.linalg.cholesky(B).T
        except np.linalg.LinAlgError:
            return None
        fro2 = _up(float(np.sum(np.square(np.abs(R)))))
        g = _up((n + 1) * U / (1.0 - (n + 1) * U))
        return _up(g * fro2)

    delta0 = _delta(A)
    if delta0 is None:
        return False, float("nan")
    scale = _up(float(np.max(np.abs(np.diag(A)))))
    for mult in (4.0, 64.0, 1024.0, 65536.0):
        c = _up(mult * max(delta0, U * scale))
        B = A.copy()
        np.fill_diagonal(B, np.nextafter(np.diag(A) - c, -np.inf))   # B <= A - cI entrywise
        delta = _delta(B)
        if delta is not None and delta < c:
            return True, _down(c - delta)
    return False, float("nan")


#: exit codes, shared with the C checker; 2 means "not the model the reader holds"
EXIT_CODES = {"PROVED": 0, "REFUTED": 1, "MODEL_MISMATCH": 2, "NOT_PROVED": 3}


def _gamma_up(k: int) -> float:
    """``gamma_k = k u / (1 - k u)`` rounded up (Higham, section 3.1)."""
    g = _up(k * U)
    return float(_up(g / _down(1.0 - g)))


def _quadform_upper(M, x, lin=None) -> float:
    """A rigorous upper bound on ``x'Mx - lin'x``.

    The sum is formed in ordinary float64 (any order, any fused operations) and bounded afterwards: each exact term
    reaches the result through at most two multiplications and ``m - 1`` additions at each of two levels plus one
    subtraction, so the computed sum is ``sum t (1 + delta)`` with ``|delta| <= gamma_{2m+2}`` whatever the order
    (Higham, Lemma 3.1)."""
    np = _numpy()
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    s = x @ (M @ x)
    T = ax @ (np.abs(M) @ ax)
    if lin is not None:
        lin = np.asarray(lin, dtype=np.float64)
        s = s - lin @ x
        T = T + np.abs(lin) @ ax
    s, T = float(s), float(T)
    if not (math.isfinite(s) and math.isfinite(T)):
        return math.inf
    g = _gamma_up(2 * M.shape[0] + 2)
    err = float(_up(_up(g * T) / _down(1.0 - g)))
    return float(_up(s + err))


def _negative_curvature_direction(M):
    """A candidate ``x`` with ``x'Mx <= 0`` from where Cholesky breaks down, or None.

    At the first non-positive pivot ``s_j``, the leading block is positive definite and ``s_j`` is its Schur
    complement, so ``x = [-M11^{-1} m12; 1; 0...]`` has ``x'Mx = s_j``. Only a candidate: a refutation rests on
    ``_quadform_upper`` evaluated on it. The C checker builds the identical direction."""
    np = _numpy()
    m = M.shape[0]
    S = np.array(M, dtype=np.float64, copy=True)
    R = np.zeros_like(S)
    for j in range(m):
        piv = S[j, j]
        if not piv > 0.0:
            x = np.zeros(m)
            x[j] = 1.0
            y = R[:j, j]
            t = np.zeros(j)
            for i in range(j - 1, -1, -1):
                t[i] = (y[i] - R[i, i + 1:j] @ t[i + 1:j]) / R[i, i]
            x[:j] = -t
            return x
        r = math.sqrt(piv)
        R[j, j] = r
        R[j, j + 1:] = S[j, j + 1:] / r
        S[j + 1:, j + 1:] -= np.outer(R[j, j + 1:], R[j, j + 1:])
    return None


def _certify_not_psd(M_up) -> tuple[bool, float]:
    """Prove the exact matrix is not PSD, or fail to. ``M_up`` bounds it from above on the diagonal and equals it
    elsewhere, so an upper bound below zero on ``x'M_up x`` certifies negative curvature."""
    x = _negative_curvature_direction(M_up)
    if x is None:
        return False, math.nan
    ub = _quadform_upper(M_up, x)
    return ub < 0.0, ub


def _beta_star_upper(A_up, w) -> float:
    """A rigorous upper bound on ``beta* = min_z z'Az - w'z``: any ``z`` bounds it above; ``z = A^{-1}w/2`` is near
    the minimiser and ``z = 0`` gives ``beta* <= 0`` whatever the solve does."""
    np = _numpy()
    try:
        z = 0.5 * np.linalg.solve(A_up, np.asarray(w, dtype=np.float64))
    except np.linalg.LinAlgError:
        return 0.0
    return min(_quadform_upper(A_up, z, lin=w), 0.0)


def _finish(out: dict) -> dict:
    """The verdict is a function of the checks alone: any certified refutation wins, then any undecided check."""
    refuted = [c for c in out["checks"] if c["status"] == "refuted"]
    unproved = [c for c in out["checks"] if c["status"] == "unproved"]
    if refuted:
        out["status"], out["verdict"] = "REFUTED", f"REFUTED: {refuted[0]['reason']}"
    elif unproved:
        out["status"], out["verdict"] = "NOT_PROVED", f"NOT PROVED: {unproved[0]['reason']}"
    else:
        out["status"], out["verdict"] = "PROVED", "PROVED"
    return out


def check_rigorous(cert: dict) -> dict:
    """The chain in float64 with outward rounding. Three-valued: PROVED, REFUTED, NOT PROVED.

    REFUTED is issued only on a certified negation (a definite violation, a verified negative quadratic form, or a
    claim above a rigorous upper bound on what the exact tier would certify), so a valid certificate is never
    refuted."""
    np = _numpy()
    why = _malformed_numeric(cert)
    if why:
        return _refuse(cert, "rigorous", why)
    Q = np.asarray(cert["Q"], dtype=np.float64)
    n = Q.shape[0]
    mu = np.asarray(cert["mu"], dtype=np.float64)
    lo = np.asarray(cert["lo"], dtype=np.float64)
    hi = np.asarray(cert["hi"], dtype=np.float64)
    d = np.asarray(cert["d"], dtype=np.float64)
    w = np.asarray(cert["w"], dtype=np.float64)
    nu, pi, lam = float(cert["nu"]), float(cert["pi"]), float(cert["lam"])
    rho, K = float(cert["rho"]), float(cert["K"])
    beta_z, claimed = float(cert["beta_z"]), float(cert["claimed_bound"])

    out = {"tier": "rigorous", "n": n, "checks": [], "model_hash": model_hash(cert),
           "certified_bound": None, "certified_bound_upper": None, "shortfall": float("inf")}

    def add(name, status, detail="", reason=None):
        out["checks"].append({"name": name, "ok": status == "ok", "status": status,
                              "detail": detail, "reason": reason or name})
        return status == "ok"

    def definite(name, cond, detail="", reason=None):
        """A check on published numbers compared exactly: failure is a refutation."""
        return add(name, "ok" if cond else "refuted", detail, reason)

    sym = definite("Q is exactly symmetric", bool(np.array_equal(Q, Q.T)))
    if "model_hash" in cert:
        definite("the carried model fingerprint matches the model carried with it",
                 cert["model_hash"] == out["model_hash"],
                 f"carried {str(cert['model_hash'])[:16]}, recomputed {out['model_hash'][:16]}")
    definite("pi >= 0", pi >= 0.0, f"pi = {pi:.6e}", "pi < 0, so weak duality does not hold")
    definite("lam >= 0", lam >= 0.0, f"lam = {lam:.6e}", "lam < 0, so weak duality does not hold")
    definite("d is finite", bool(np.all(np.isfinite(d))), f"min d = {float(np.min(d)):.6e}")
    definite("lo <= hi", bool(np.all(lo <= hi)))

    # A rounds the diagonal DOWN (proving it PSD proves the true matrix PSD); A_up rounds it UP (the side a
    # refutation must use).
    A = Q.copy()
    np.fill_diagonal(A, np.nextafter(np.diag(Q) - d, -np.inf))
    A = 0.5 * (A + A.T)
    A_up = Q.copy()
    np.fill_diagonal(A_up, np.nextafter(np.diag(Q) - d, np.inf))
    A_up = 0.5 * (A_up + A_up.T)
    psd, lmin = verified_psd(A)
    if not psd:
        lmin_est = float(np.linalg.eigvalsh(A)[0])         # diagnostic only, never part of the proof
        rel = lmin_est / max(float(np.max(np.abs(np.diag(A)))), 1e-300)
        out["psd_shortfall_rel"] = rel
        refuted, ub = _certify_not_psd(A_up) if sym else (False, math.nan)
        add("A = Q - diag(d) is positive semidefinite (verified Cholesky)",
            "refuted" if refuted else "unproved",
            (f"certified not PSD: x'Ax <= {ub:.3e} < 0 along a Cholesky breakdown direction"
             if refuted else f"not proved; unverified lambda_min ~ {lmin_est:.3e}, i.e. "
                             f"{rel:.2e} of max|diag(A)|"),
            ("the split leaves a matrix that is not PSD, so the z-block minimum is -inf"
             if refuted else "PSD of the split could not be established in float64"))
        return _finish(out)
    add("A = Q - diag(d) is positive semidefinite (verified Cholesky)", "ok",
        f"lambda_min >= {lmin:.6e}")

    # z-block: z'Az - w'z >= beta_z for all z  <=  the bordered matrix [[A, -w/2], [-w'/2, -beta_z]] is PSD.
    # -0.5 * w is exact (a power-of-two scaling); slack is added only in the corner, where rounding -beta_z down
    # strengthens what is proved.
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[:n, :n] = A
    M[:n, n] = -0.5 * w
    M[n, :n] = M[:n, n]
    M[n, n] = np.nextafter(-beta_z, -np.inf)
    psd_m, lmin_m = verified_psd(M)
    if not psd_m:
        # at beta_z = 0 the rounded corner is -5e-324 and M is genuinely not PSD, so a refutation is sought on the
        # exact matrix (A_up, corner -beta_z) instead
        M_up = M.copy()
        M_up[:n, :n] = A_up
        M_up[n, n] = -beta_z
        refuted, ub = _certify_not_psd(M_up) if sym else (False, math.nan)
        add("beta_z is a lower bound on min_z z'Az - w'z (bordered matrix is PSD)",
            "refuted" if refuted else "unproved",
            (f"certified not PSD: the form is <= {ub:.3e} < 0, so beta_z overstates the z-block "
             f"minimum" if refuted else "not proved: beta_z sits within rounding of the z-block "
                                        "minimum, where verified Cholesky cannot succeed"),
            ("published beta_z is not a valid lower bound on the z-block" if refuted
             else "the z-block bound has no slack the float64 checker can resolve"))
        return _finish(out)
    add("beta_z is a lower bound on min_z z'Az - w'z (bordered matrix is PSD)", "ok",
        f"lambda_min >= {lmin_m:.6e}")

    # per-asset minima in interval arithmetic: endpoints always; the vertex -c^2/(4d) where it can lie in the box
    iv_d, iv_lo, iv_hi, iv_mu = _iv(d), _iv(lo), _iv(hi), _iv(mu)
    c = _isub(_iadd(_iv(w), _iv(np.full(n, nu))), _imul(_iv(np.full(n, pi)), iv_mu))

    def q(s):
        return _iadd(_imul(iv_d, _imul(s, s)), _imul(c, s))

    q_at_lo, q_at_hi = q(iv_lo), q(iv_hi)
    m_lo = np.minimum(q_at_lo[0], q_at_hi[0])
    m_hi = np.minimum(q_at_lo[1], q_at_hi[1])            # upper side, for refutation: endpoints are feasible
    pos = d > 0.0
    if np.any(pos):
        safe_d = np.where(pos, d, 1.0)
        v = _idiv_pos((-c[1], -c[0]), _iv(2.0 * safe_d))
        inside = pos & (v[1] > lo) & (v[0] < hi)        # possibly inside: include for the lower bound
        vert = _idiv_pos(_imul((-c[1], -c[0]), c), _iv(4.0 * safe_d))
        m_lo = np.where(inside, np.minimum(m_lo, vert[0]), m_lo)
        surely_inside = pos & (v[0] >= lo) & (v[1] <= hi)    # certainly inside: include for the upper bound
        m_hi = np.where(surely_inside, np.minimum(m_hi, vert[1]), m_hi)

    terms = np.minimum(np.nextafter(m_lo + lam, -np.inf), 0.0)
    per = np.nextafter(math.fsum(terms.tolist()), -np.inf)        # fsum is correctly rounded: one ulp covers it
    terms_hi = np.minimum(np.nextafter(m_hi + lam, np.inf), 0.0)
    per_hi = np.nextafter(math.fsum(terms_hi.tolist()), np.inf)

    total = beta_z
    for term in (-nu, np.nextafter(pi * rho, -np.inf), np.nextafter(-lam * K, -np.inf), per):
        total = np.nextafter(total + term, -np.inf)
    certified = float(total)

    # the upper bound only matters once the claim has failed claimed <= certified; otherwise +inf is a true bound
    if claimed <= certified:
        certified_hi = math.inf
    else:
        total_hi = _beta_star_upper(A_up, w)
        for term in (-nu, np.nextafter(pi * rho, np.inf), np.nextafter(-lam * K, np.inf), per_hi):
            total_hi = np.nextafter(total_hi + term, np.inf)
        certified_hi = float(total_hi)

    out["certified_bound"] = certified
    out["certified_bound_upper"] = certified_hi
    out["claimed_bound"] = claimed
    out["shortfall"] = claimed - certified
    out["lambda_min_A"] = lmin
    detail = (f"claimed = {claimed:.12g}, certified in [{certified:.12g}, {certified_hi:.12g}], "
              f"shortfall = {claimed - certified:.3e}")
    if claimed <= certified:
        add("claimed bound <= rigorously re-derived bound", "ok", detail)
    elif claimed > certified_hi:
        add("claimed bound <= rigorously re-derived bound", "refuted", detail,
            "the claimed bound exceeds anything the published witness supports")
    else:
        add("claimed bound <= rigorously re-derived bound", "unproved", detail,
            "the claim sits within rounding of the re-derived bound")
    return _finish(out)


def report(res: dict) -> str:
    lines = [f"tier={res['tier']}  n={res['n']}  verdict={res['verdict']}",
             f"  model  = {res['model_hash'][:32]}  (the problem this verdict is about)"]
    tag = {"ok": "PASS", "refuted": "FAIL", "unproved": "UNPROVED"}
    for ch in res["checks"]:
        lines.append(f"  [{tag[ch['status']]}] {ch['name']}"
                     + (f" -- {ch['detail']}" if ch["detail"] else ""))
    if res.get("certified_bound") is not None:
        lines.append(f"  claimed   = {res['claimed_bound']:.12g}")
        lines.append(f"  certified = {res['certified_bound']:.12g}")
        lines.append(f"  shortfall = {res['shortfall']:.6e}"
                     + ("   <-- the claim exceeds what is provable" if res["shortfall"] > 0 else ""))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m heurics_cert.checkers.bound")
    ap.add_argument("certificate")
    ap.add_argument("--exact", action="store_true", help="exact rational tier (small n only)")
    ap.add_argument("--expect-model", default="",
                    help="sha256 (prefix) of the model you hold; any other model is a mismatch")
    args = ap.parse_args()
    cert = load(args.certificate)
    res = check_exact(cert) if args.exact else check_rigorous(cert)
    print(report(res))
    if args.expect_model and not res["model_hash"].startswith(args.expect_model):
        print(f"  MODEL MISMATCH: expected {args.expect_model}, certificate carries "
              f"{res['model_hash']}")
        return EXIT_CODES["MODEL_MISMATCH"]
    return EXIT_CODES[res["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
