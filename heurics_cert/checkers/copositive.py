"""Checker for CCPO-CERT/2: diagonal (tier A) and copositive (tier B) certificates, by reduction to CCPO-CERT/1.

A tier-B certificate carries ``sigma`` and a symmetric ``G_off >= 0`` with zero diagonal. For every feasible ``x``
(``x >= 0``, ``1'x = 1``)

    x'Qx = x'Ax + x'G_off x + sum_i g_i x_i^2 - sigma,       A = Q + sigma 11' - G_off - diag(g),

and ``x'G_off x >= 0``, so ``x'Qx >= [x'Qt x with Qt = Q + sigma 11' - G_off, split g] - sigma``. The bracket is a
CCPO-CERT/1 claim about ``Qt``, which is handed to the CCPO-CERT/1 checkers unchanged. Tier A (``sigma = 0``,
``G_off = 0``) passes through untouched.

``Qt`` is rounded down entrywise (and mirrored to stay exactly symmetric): with ``x >= 0`` the form ``x'Mx`` grows
with every entry of ``M``, so a bound proved for the rounded matrix holds for the true one. Both steps need
``x >= 0``, so tier B requires ``lo >= 0``.

Refutable here: a negative, asymmetric or non-zero-diagonal ``G_off``, and (tier B) a negative ``lo``. Everything
else is delegated, and a delegated NOT PROVED is never promoted to REFUTED. The verdict reports the fingerprint of the
original model ``(Q, mu, lo, hi, K, rho)``, which is what the reader holds.
"""
from __future__ import annotations

import math

from heurics_cert.checkers import bound

REQUIRED = bound.REQUIRED + ("sigma",)
#: ``G_off`` may be omitted (tier A: all zeros)
OPTIONAL = ("G_off",)


def _numpy():
    return bound._numpy()


def reduce(cert: dict) -> tuple[dict, list, str]:
    """``(ccpo1_document, checks, model_hash_of_the_original)``. The document is about ``Qt`` (rounded down, exactly
    symmetric) and claims ``claimed_bound + sigma``; a ``refuted`` entry in ``checks`` is definite."""
    np = _numpy()
    checks = []

    def add(name, status, detail=""):
        checks.append({"name": name, "ok": status == "ok", "status": status, "detail": detail})
        return status == "ok"

    Q = np.asarray(cert["Q"], dtype=np.float64)
    n = Q.shape[0]
    sigma = float(cert.get("sigma", 0.0))
    original_hash = bound.model_hash(cert)

    G = cert.get("G_off")
    if G is None:
        G = np.zeros((n, n), dtype=np.float64)
    else:
        G = np.asarray(G, dtype=np.float64)

    add("G_off has the model's shape", "ok" if G.shape == (n, n) else "refuted", f"{G.shape} vs ({n}, {n})")
    add("G_off is exactly symmetric", "ok" if bool(np.array_equal(G, G.T)) else "refuted")
    add("G_off has a zero diagonal", "ok" if bool(np.all(np.diag(G) == 0.0)) else "refuted",
        "the diagonal belongs to the split d, never to the dropped term")
    neg = float(np.min(G)) if G.size else 0.0
    add("G_off >= 0 entrywise", "ok" if neg >= 0.0 else "refuted",
        f"min entry {neg:.6e}" + ("" if neg >= 0 else "; x'G_off x >= 0 is what licenses dropping it"))
    add("sigma is finite", "ok" if math.isfinite(sigma) else "refuted", f"sigma = {sigma:.6e}")
    tier_a = sigma == 0.0 and not G.any()
    if not tier_a:
        lo_min = float(np.min(np.asarray(cert["lo"], dtype=np.float64)))
        add("lo >= 0 (tier B)", "ok" if lo_min >= 0.0 else "refuted",
            f"min lo {lo_min:.6e}" + ("" if lo_min >= 0 else "; dropping x'G_off x and rounding Qt down need x >= 0"))
    if any(c["status"] == "refuted" for c in checks):
        return {}, checks, original_hash

    if tier_a:
        Qt = Q
    else:
        Qt = np.nextafter(np.nextafter(Q + sigma, -np.inf) - G, -np.inf)
        Qt = np.triu(Qt) + np.triu(Qt, 1).T

    inner = dict(cert)
    inner.pop("G_off", None)
    inner.pop("sigma", None)
    inner.pop("model_hash", None)                  # the inner model is Qt, not the model the reader holds
    inner["Q"] = Qt
    inner["claimed_bound"] = float(cert["claimed_bound"]) + sigma
    return inner, checks, original_hash


def _malformed(cert: dict) -> str | None:
    """Why ``cert`` is not well formed, or None: the CCPO-CERT/1 fields, a finite ``sigma``, and ``G_off`` (when
    present) an n x n matrix of finite reals."""
    why = bound._malformed_numeric(cert)
    if why:
        return why
    np = _numpy()
    if "sigma" not in cert or not bound._is_real(cert["sigma"].item() if hasattr(cert["sigma"], "item")
                                                 else cert["sigma"]):
        return "sigma is not a finite real number"
    if cert.get("G_off") is not None:
        n = len(cert["mu"])
        try:
            G = np.asarray(cert["G_off"], dtype=np.float64)
        except (TypeError, ValueError):
            return "G_off is not a numeric matrix"
        if G.shape != (n, n) or not np.all(np.isfinite(G)):
            return f"G_off is not an {n} x {n} matrix of finite reals"
    return None


def check(cert: dict, *, method: str = "rigorous", qbin: str | None = None) -> dict:
    """Verify a CCPO-CERT/2 certificate; ``method`` is the tier for the delegated chain (``"rigorous"``,
    ``"exact"`` or ``"c"``)."""
    why = _malformed(cert)
    if why:
        refused = bound._refuse(cert, f"copos+{method}", why)
        refused["verdict"], refused["reason"] = "REFUTED", f"malformed certificate: {why}"
        return refused
    inner, checks, original_hash = reduce(cert)
    sigma = float(cert.get("sigma", 0.0))
    out = {"tier": f"copos+{method}", "checks": list(checks), "model_hash": original_hash,
           "sigma": sigma, "certified_bound": None}
    if any(c["status"] == "refuted" for c in checks):
        out["verdict"] = "REFUTED"
        out["reason"] = next(c["name"] for c in checks if c["status"] == "refuted")
        return out

    if "model_hash" in cert:
        ok = cert["model_hash"] == original_hash
        out["checks"].append({"name": "the carried fingerprint matches the ORIGINAL model", "ok": ok,
                              "status": "ok" if ok else "refuted",
                              "detail": f"carried {str(cert['model_hash'])[:16]}, recomputed {original_hash[:16]}"})
        if not ok:
            out["verdict"] = "REFUTED"
            out["reason"] = "the certificate is about a different model than the reader holds"
            return out

    np = _numpy()
    if method == "exact":
        res = bound.check_exact({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in inner.items()})
    elif method == "c":
        from heurics_cert.checkers import native
        res = native.run(inner, qbin=qbin)
    else:
        res = bound.check_rigorous(inner)
    out["checks"] += [dict(c, name=f"[{method}] {c['name']}") for c in res.get("checks", [])]
    raw = str(res.get("verdict", "NOT PROVED"))
    out["verdict"] = raw.split(":")[0].strip()
    if ":" in raw:
        out["reason"] = raw.split(":", 1)[1].strip()
    cb = res.get("certified_bound")
    if cb is not None:
        # the reader's bound is the delegated one minus sigma, rounded down when sigma moves it
        out["certified_bound"] = float(cb) if sigma == 0.0 else float(np.nextafter(float(cb) - sigma, -np.inf))
        if out["verdict"] == "PROVED" and float(cert["claimed_bound"]) > out["certified_bound"]:
            out["verdict"] = "NOT PROVED"
            out["reason"] = "the claim sits within rounding of the certified bound once sigma is subtracted"
    return out
