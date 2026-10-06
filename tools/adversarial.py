"""Does the checker actually check anything? A mutation suite, because *"a verifier validated by
its author verifies nothing."*

Independent third-party review is the real answer to that, and this does not substitute for it. What it
does do is close the cheapest way for a checker to be worthless, which is to accept everything. Every mutation
below turns a valid certificate into an invalid one in a way a buggy prover could plausibly produce, and the
checker has to reject each. A mutation that slips through is a hole in the checker, and the run fails.

`witness_from_the_wrong_iterate` reproduces a defect an earlier prover actually had.

    python tools/adversarial.py
"""
from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heurics_cert.checkers import bound as certcheck  # noqa: E402

FAILURES: list[str] = []
CHECKS = [0]


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS[0] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def valid_certificate(seed: int = 3, n: int = 18, K: int = 5, margin: float = 1e-9):
    """A certificate built from the tangent construction, with a margin, so it is genuinely valid."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 5))
    Q = F @ F.T / n + np.diag(rng.uniform(0.05, 0.5, n))
    Q = 0.5 * (Q + Q.T)
    mu = rng.uniform(0.0, 1.0, n)
    lo, hi = np.zeros(n), np.full(n, 2.0 / K)
    rho = float(0.3 * mu.max())

    theta = float(np.linalg.eigvalsh(Q)[0])
    scale = float(np.max(np.sum(np.abs(Q), axis=1)))
    d = np.full(n, theta) - margin * scale
    A = Q - np.diag(d)
    xbar = np.zeros(n)
    xbar[np.argsort(-mu)[:K]] = 1.0 / K
    w = 2.0 * A @ xbar

    sig, U = np.linalg.eigh(0.5 * (A + A.T))
    alpha = U.T @ w
    beta = -0.25 * float(np.sum(alpha ** 2 / sig)) * (1.0 + margin)
    c = w
    cand = np.stack([lo, hi, np.clip(-c / (2.0 * d), lo, hi)])
    m = np.min(d * cand ** 2 + c * cand, axis=0)
    raw = beta + float(np.sum(np.minimum(m, 0.0)))
    return {"Q": Q.tolist(), "mu": mu.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "K": K, "rho": rho, "d": d.tolist(), "w": w.tolist(),
            "nu": 0.0, "pi": 0.0, "lam": 0.0,
            "beta_z": beta, "claimed_bound": raw - margin * abs(raw)}


def mutations(base: dict):
    """Each entry turns the certificate invalid in one specific, plausible way."""
    n = len(base["mu"])
    rng = np.random.default_rng(99)
    out = []

    def m(name, **over):
        c = copy.deepcopy(base)
        c.update(over)
        out.append((name, c))

    # One ulp above what the checker can actually certify -- the sharpest test of its resolution.
    # (One ulp above the *claimed* value proves nothing: a certificate with a safety margin has
    # room to spare by construction, and accepting inside that room is correct behaviour.)
    certified = certcheck.check_exact(base)["certified_bound"]
    m("claimed_one_ulp_above_certified", claimed_bound=math.nextafter(certified, math.inf))
    m("claimed_bound_up_1e-9", claimed_bound=base["claimed_bound"] + 1e-9)
    m("claimed_bound_up_1_percent",
      claimed_bound=base["claimed_bound"] + 0.01 * abs(base["claimed_bound"]))
    m("negative_pi", pi=-1e-3)
    m("negative_lam", lam=-1e-3)
    # `pi` and `lam` priced *without* being paid for in the objective: the classic sign slip.
    m("pi_moved_without_the_rho_term", pi=1.0)
    m("lam_moved_without_the_K_term", lam=1.0)
    # the split pushed past lambda_min, i.e. exactly the shipped defect
    m("split_past_lambda_min",
      d=(np.asarray(base["d"]) + 1e-6 * float(np.max(np.abs(base["Q"])))).tolist())
    # beta_z overstated: the z-block "minimum" claimed higher than it is
    m("beta_z_overstated", beta_z=base["beta_z"] * (1.0 - 1e-6))
    # w perturbed after the fact, so the witness no longer matches the value
    m("w_perturbed", w=(np.asarray(base["w"]) * (1.0 + 1e-6)).tolist())
    # a real prover bug: publishing a *different* iterate's multipliers than the one that won
    m("witness_from_the_wrong_iterate",
      w=(np.asarray(base["w"]) + 0.05 * rng.standard_normal(n)).tolist())
    return out


#: Mutations the rigorous tier is *allowed* to leave undecided, each because no float64 interval
#: argument can resolve it -- not because the checker is weak. Anything else coming back NOT PROVED
#: is a detection regression, and fails the suite.
SUB_RESOLUTION = {
    # One ulp above the exact certified value. A float64 enclosure is at least one ulp wide, so the
    # claim sits inside the interval that contains the true value; only exact arithmetic separates it.
    "claimed_one_ulp_above_certified",
}


def boundary_certificate(n: int = 6) -> dict:
    """Valid, with no slack at all: `w = 0`, so the z-block minimum is exactly `0 = beta_z`."""
    return {"Q": np.diag(np.linspace(2.0, 4.0, n)).tolist(), "mu": [0.1] * n, "lo": [0.0] * n,
            "hi": [1.0] * n, "K": n, "rho": 0.0, "d": [1.0] * n, "w": [0.0] * n,
            "nu": 0.0, "pi": 0.0, "lam": 0.0, "beta_z": 0.0, "claimed_bound": -1.0}


def model_substitutions(base: dict):
    """Mutations that leave the certificate *valid* -- about a different problem.

    These are not checker holes and the suite does not treat them as such. They are the boundary of
    what any self-contained certificate can promise, and the reason `certcheck.model_hash` exists:
    the verdict is about the model the certificate carries, so a reader has to bind that model to
    the one they meant. Tightening a constraint raises the optimum, so a substituted model yields a
    *higher* bound that is perfectly provable and quietly answers the wrong question.
    """
    out = []
    for name, over in (
        ("box_narrowed", {"hi": (np.asarray(base["hi"]) * 0.5).tolist()}),
        ("return_floor_raised", {"rho": base["rho"] * 2.0}),
        ("cardinality_reduced", {"K": max(1, int(base["K"]) - 2)}),
    ):
        c = copy.deepcopy(base)
        c.update(over)
        out.append((name, c))
    return out


def stdlib_only_check(base: dict) -> bool:
    """The exact tier must run on a bare interpreter, with numpy deliberately unavailable.

    The whole claim of that tier is that nothing has to be trusted. Requiring a compiled
    linear-algebra stack to run it would undercut precisely that, so this makes `import numpy` fail
    and re-runs the proof: a reviewer with nothing but CPython has to be able to check us.
    """
    import builtins
    real_import = builtins.__import__

    def guarded(name, *a, **k):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy deliberately unavailable")
        return real_import(name, *a, **k)

    saved = {k: v for k, v in sys.modules.items() if k.startswith("numpy")}
    for k in saved:
        del sys.modules[k]
    builtins.__import__ = guarded
    try:
        cert = dict(base)
        cert["model_hash"] = certcheck.model_hash(cert)
        res = certcheck.check_exact(cert)
        touched = any(k.startswith("numpy") for k in sys.modules)
        return res["verdict"] == "PROVED" and not touched
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def main() -> int:
    base = valid_certificate()
    print("baseline (must be accepted by both tiers):")
    for tier, fn in (("exact", certcheck.check_exact), ("rigorous", certcheck.check_rigorous)):
        res = fn(base)
        check(f"valid certificate accepted -- {tier}", res["verdict"] == "PROVED",
              res["verdict"])

    # A valid certificate on a degeneracy boundary. Verified Cholesky cannot prove zero slack, so the
    # rigorous tier may decline -- but it must never *refute*: a REFUTED here is a checker indicting a
    # correct solver. This is the case both verifiers got wrong until the verdicts became three-valued.
    print("\nboundary certificates (valid; may be undecided, must never be refuted):")
    boundary = boundary_certificate()
    check("boundary certificate proved -- exact",
          certcheck.check_exact(boundary)["status"] == "PROVED",
          certcheck.check_exact(boundary)["verdict"])
    res = certcheck.check_rigorous(boundary)
    check("boundary certificate not refuted -- rigorous", res["status"] != "REFUTED", res["verdict"])

    # Detection and inconclusiveness are reported *separately*. The old criterion -- "anything but
    # PROVED counts as rejected" -- scored a checker that answered NOT PROVED to every forgery as a
    # perfect detector, and so could not see the weakness a reviewer would probe first.
    print("\nmutations (exact must refute; rigorous must never prove, and should refute):")
    detected, undecided = [], []
    muts = mutations(base)
    for name, cert in muts:
        ex, rg = certcheck.check_exact(cert), certcheck.check_rigorous(cert)
        check(f"{name}: refuted -- exact", ex["status"] == "REFUTED", ex["verdict"][:60])
        check(f"{name}: not proved -- rigorous", rg["status"] != "PROVED", rg["verdict"][:60])
        if rg["status"] == "REFUTED":
            detected.append(name)
        elif rg["status"] == "NOT_PROVED":
            undecided.append(name)
            check(f"{name}: undecided only because it is below float64 resolution",
                  name in SUB_RESOLUTION, rg["verdict"][:60])
    print(f"\n  rigorous detection rate   {len(detected)}/{len(muts)}  (REFUTED, with a certified reason)")
    print(f"  rigorous inconclusive     {len(undecided)}/{len(muts)}  (NOT PROVED: "
          f"{', '.join(undecided) or 'none'})")

    print("\nno-dependency check:")
    check("the exact tier proves a certificate with numpy unavailable", stdlib_only_check(base))

    print("\nmodel substitutions (each must stay provable, and must move the fingerprint):")
    base_hash = certcheck.model_hash(base)
    base_certified = certcheck.check_exact(base)["certified_bound"]
    for name, cert in model_substitutions(base):
        res = certcheck.check_exact(cert)
        check(f"{name} moves the model fingerprint",
              certcheck.model_hash(cert) != base_hash)
        check(f"{name} still certifies, and certifies a HIGHER bound",
              res["certified_bound"] is not None and res["certified_bound"] >= base_certified,
              f"{res['certified_bound']!r} vs {base_certified:.10g} on the true model")

    print(f"\n{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
    if FAILURES:
        print("FAILURES: " + ", ".join(FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
