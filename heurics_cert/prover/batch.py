"""Batched prover for singular covariances: every problem sharing one ``Q`` at once, on the GPU (``jax`` extra).

Eligible problems have a singular ``Q``, uniform ``hi`` and the cardinality row implied (``K * hi >= 1``). There the
split is ``d = 0`` and the relaxation is a capped-simplex QP with one return row, dualised by ``pi >= 0`` and solved by
FISTA in float32 on a Nystrom factor of ``Q``. The GPU only proposes tangent points; the witness arithmetic is float64
on the host and the checkers are unchanged, so float32 can cost tightness, never validity.

Per covariance: Nystrom factor -> coarse FISTA over a log grid of ``pi`` (+ refinements) -> final FISTA at ``pi*`` ->
exact refinement of each point (``active_set``, falling back to the raw point) -> float32 grid + float64 search over
``(nu, pi)`` -> margin ladder -> checkers. A problem whose bound falls short of the solver's own dual estimate is
re-solved with a longer budget.
"""
from __future__ import annotations

import functools
import time

import numpy as np

from heurics_cert.certificates.certificate import Certificate
from heurics_cert.certificates.witness import Witness
from heurics_cert.errors import UnsupportedProblemError
from heurics_cert.prover import active_set, tangent
from heurics_cert.prover.result import Result
from heurics_cert.checkers.api import verify

MARGIN_LADDER = tangent.MARGIN_LADDER


def eligible(problem, lam_min: float | None = None) -> tuple[bool, str]:
    """``(eligible, reason)``. ``lam_min`` of ``Q`` may be passed in (O(n^3), shared by every problem on one
    covariance)."""
    Q = problem.Q
    if not np.allclose(problem.hi, problem.hi[0]):
        return False, "hi is not uniform"
    if problem.K * float(problem.hi.min()) < 1.0:
        return False, "cardinality row not implied by the budget"
    lam_min = float(np.linalg.eigvalsh(Q)[0]) if lam_min is None else float(lam_min)
    if lam_min > 1e-12 * float(np.max(np.abs(np.diag(Q)))):
        return False, "Q is nonsingular: a positive split exists, use prove()"
    return True, ""


@functools.lru_cache(maxsize=1)
def kernels():
    import jax
    import jax.numpy as jnp
    P = jax.lax.Precision.HIGHEST

    @jax.jit
    def nystrom(Q, Om):
        """`Q ~ L L'`, exact to rounding when `rank(Q) <= Om.shape[1]`: one n x n matmul and a small eigh."""
        F = jnp.matmul(Q, Om, precision=P)
        C = jnp.matmul(Om.T, F, precision=P)
        C = 0.5 * (C + C.T)
        ev, U = jnp.linalg.eigh(C)
        keep = ev > 1e-13 * jnp.max(ev)
        return jnp.matmul(F, U * jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, ev, 1.0)), 0.0), precision=P)

    def project(v, hi, steps=30):
        """Projection onto `{0 <= x <= hi, 1'x = 1}`: bisection on the shift, unrolled into one kernel (60
        sequential launches per iteration otherwise -- 10x slower, identical arithmetic)."""
        a = jnp.min(v, axis=-1, keepdims=True) - 1.0
        b = jnp.max(v, axis=-1, keepdims=True)

        def body(_, t):
            a, b = t
            m = 0.5 * (a + b)
            s = jnp.sum(jnp.clip(v - m, 0.0, hi), axis=-1, keepdims=True)
            return (jnp.where(s > 1.0, m, a), jnp.where(s > 1.0, b, m))

        a, b = jax.lax.fori_loop(0, steps, body, (a, b), unroll=steps)
        return jnp.clip(v - 0.5 * (a + b), 0.0, hi)

    @functools.partial(jax.jit, static_argnums=(5,))
    def fista(L, mu, hi, rho, pis, iters):
        """min x'(LL')x - pi (mu'x - rho) over the capped simplex, one row per (cell, pi), with a per-row
        gradient-scheme restart. Returns `(x, inner value)`; the inner value is a valid dual for any iterate."""
        Lt = L.T
        v = jnp.ones((L.shape[0],), L.dtype) / jnp.sqrt(L.shape[0])
        v = jax.lax.fori_loop(0, 30, lambda _, v: (lambda u: u / jnp.linalg.norm(u))(
            jnp.matmul(L, jnp.matmul(Lt, v, precision=P), precision=P)), v)
        lam = jnp.vdot(v, jnp.matmul(L, jnp.matmul(Lt, v, precision=P), precision=P))
        step = 1.0 / (2.0 * lam * 1.02)
        g = pis[:, None] * mu
        x0 = project(jnp.zeros_like(g), hi)

        def body(_, st):
            x, z, t = st
            grad = 2.0 * jnp.matmul(jnp.matmul(z, L, precision=P), Lt, precision=P) - g
            xn = project(z - step * grad, hi)
            bad = jnp.sum((z - xn) * (xn - x), axis=-1, keepdims=True) > 0.0
            t = jnp.where(bad, 1.0, t)
            tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            zn = xn + jnp.where(bad, 0.0, (t - 1.0) / tn) * (xn - x)
            return (xn, zn, tn)

        x, _, _ = jax.lax.fori_loop(0, iters, body, (x0, x0, jnp.ones((g.shape[0], 1), g.dtype)))
        Lx = jnp.matmul(x, L, precision=P)
        return x, jnp.sum(Lx * Lx, axis=-1) - pis * jnp.sum(x * mu, axis=-1) + pis * rho

    @functools.partial(jax.jit, static_argnums=(5,))
    def scalar_grid(w, d, mu, lo, hi, K, rho, z_block, nus, pis):
        """The certificate's dual over a grid of `(nu, pi)`, float32 -- only to locate the maximiser."""
        c = w[None, :] + nus[:, None] - pis[:, None] * mu[None, :]
        safe = jnp.where(d > 0, d, 1.0)
        vert = jnp.where(d > 0, jnp.clip(-c / (2.0 * safe), lo, hi), jnp.broadcast_to(lo, c.shape))
        m = jnp.minimum(jnp.minimum(d * lo * lo + c * lo, d * hi * hi + c * hi), d * vert * vert + c * vert)
        kth = -jax.lax.top_k(-m, K + 1)[0][:, K]
        lam = jnp.maximum(0.0, -kth)
        return (z_block - nus + pis * rho - lam * K
                + jnp.sum(jnp.minimum(m + lam[:, None], 0.0), axis=-1)), lam

    return nystrom, fista, scalar_grid


def refine_scalars(w, beta_z, d, mu, lo, hi, K, rho, nu0, pi0, span_nu, span_pi, rounds=5, iters=40):
    f = lambda nu, pi: tangent.dual_value(w, nu, max(pi, 0.0), beta_z, d, mu, lo, hi, K, rho)[0]   # noqa: E731
    nu, pi = float(nu0), max(float(pi0), 0.0)
    best = (f(nu, pi), nu, pi)
    rn, rp = span_nu, span_pi
    for _ in range(rounds):
        nu = tangent.ternary_max(lambda t: f(t, pi), nu - rn, nu + rn, iters)
        pi = max(tangent.ternary_max(lambda t: f(nu, t), max(pi - rp, 0.0), pi + rp, iters), 0.0)
        v = f(nu, pi)
        if v > best[0]:
            best = (v, nu, pi)
        rn, rp = 0.4 * rn, 0.4 * rp
    return best


def _extract_at(problem, x_hat, margin, pi_seed, nu_kkt=None) -> Witness:
    """Tangent witness at ``x_hat`` with ``d = -margin * err`` (no factorisation here: the checker's verified
    Cholesky is the authority, and a NOT PROVED retries at the next margin)."""
    import jax.numpy as jnp
    _, _, scalar_grid = kernels()
    Q, mu, lo, hi, K, rho = problem.Q, problem.mu, problem.lo, problem.hi, problem.K, problem.rho
    n = problem.n
    err = (n + 2) * tangent.U * float(np.sum(np.abs(np.diag(Q))))
    d = np.full(n, -margin * err)
    Ax = Q @ x_hat - d * x_hat
    w = 2.0 * Ax
    z = -float(x_hat @ Ax)
    beta_z = z - (1e-9 * max(abs(z), 1e-300) + margin * err)
    span_nu = 2.0 * max(float(np.max(np.abs(w))), 1e-12)
    span_pi = span_nu / max(float(np.max(np.abs(mu))), 1e-12)
    args32 = [jnp.asarray(a, jnp.float32) for a in (w, d, mu, lo, hi)]
    nu_c, pi_c, rn, rp = 0.0, max(float(pi_seed), 0.0), span_nu, span_pi
    for _ in range(4):
        nus = jnp.linspace(nu_c - rn, nu_c + rn, 64, dtype=jnp.float32)
        pis = jnp.maximum(jnp.linspace(pi_c - rp, pi_c + rp, 64, dtype=jnp.float32), 0.0)
        NU, PI = [a.ravel() for a in jnp.meshgrid(nus, pis, indexing="ij")]
        vals, _ = scalar_grid(*args32, K, float(rho), np.float32(beta_z), NU, PI)
        j = int(jnp.argmax(vals))
        nu_c, pi_c = float(NU[j]), float(PI[j])
        rn, rp = 2.0 * rn / 63, 2.0 * rp / 63
    val, nu, pi = refine_scalars(w, beta_z, d, mu, lo, hi, K, rho, nu_c, pi_c, rn, rp)
    if nu_kkt is not None:
        # D(nu) has a cusp at the relaxation's own budget multiplier: seed it and refine in a hairline bracket
        for pi0 in {0.0, float(pi_seed)}:
            v2, nu2, pi2 = refine_scalars(w, beta_z, d, mu, lo, hi, K, rho, -nu_kkt, pi0,
                                          max(1e-6 * abs(nu_kkt), 1e-300),
                                          max(1e-6 * abs(pi0), 1e-300) if pi0 else 0.0)
            if v2 > val:
                val, nu, pi = v2, nu2, pi2
    _, lam = tangent.dual_value(w, nu, pi, beta_z, d, mu, lo, hi, K, rho)
    return Witness(d=d, w=w, nu=nu, pi=pi, lam=lam, beta_z=beta_z, claimed_bound=val - 1e-9 * abs(val))


def extract(problem, x_hat, margin, pi_seed) -> Witness:
    """The better of two witnesses: at the GPU iterate, and at its exact refinement (which can fail to converge)."""
    best = _extract_at(problem, x_hat, margin, pi_seed)
    xr, nu_r, pi_r, it = active_set.solve(problem.Q, problem.mu, problem.hi, problem.rho, x_hat)
    if it < active_set.MAX_ITER and np.isfinite(xr).all() and np.isfinite(nu_r):
        alt = _extract_at(problem, xr, margin, pi_r, nu_kkt=nu_r)
        if alt.claimed_bound > best.claimed_bound:
            best = alt
    return best


def solve_group(problems, *, grid=24, rounds=3, coarse=500, final=3000, rank=128, seed=0):
    """One Nystrom factor and one batched FISTA for every problem sharing this covariance. Returns
    ``(x_hats, pi_stars, dual_estimates, seconds)`` in model units."""
    import jax
    import jax.numpy as jnp
    nystrom, fista, _ = kernels()
    t0 = time.perf_counter()
    Q = problems[0].Q
    alpha = float(np.mean(np.diag(Q)))
    n = Q.shape[0]
    L = nystrom(jnp.asarray(Q / alpha, jnp.float32),
                jax.random.normal(jax.random.PRNGKey(seed), (n, rank), jnp.float32))
    C = len(problems)
    MU = jnp.asarray(np.array([p.mu for p in problems]), jnp.float32)
    HI = jnp.asarray(np.array([[float(p.hi[0])] for p in problems]), jnp.float32)
    RHO = np.array([p.rho for p in problems])
    mu_scale = np.array([max(float(np.max(np.abs(p.mu))), 1e-12) for p in problems])

    def run(pis_2d, iters):
        g = pis_2d.shape[1]
        rep = lambda A: jnp.repeat(A, g, axis=0)                              # noqa: E731
        x, val = fista(L, rep(MU), rep(HI), jnp.asarray(np.repeat(RHO, g), jnp.float32),
                       jnp.asarray(pis_2d.reshape(-1), jnp.float32), iters)
        return x, np.asarray(val, np.float64).reshape(C, g)

    pis = np.concatenate([np.zeros((C, 1)), np.logspace(-4.0, 4.0, grid - 1) / mu_scale[:, None]], axis=1)
    x, val = run(pis, coarse)
    for _ in range(rounds):
        j = np.argmax(val, axis=1)
        lo = np.array([max(pis[i, max(j[i] - 1, 0)], 0.0) for i in range(C)])
        hi_ = np.array([pis[i, min(j[i] + 1, pis.shape[1] - 1)] for i in range(C)])
        hi_ = np.where(hi_ <= lo, lo + 1.0 / mu_scale, hi_)
        pis = np.linspace(lo, hi_, grid, axis=1)
        x, val = run(pis, coarse)
    j = np.argmax(val, axis=1)
    pi_star = np.array([pis[i, j[i]] for i in range(C)])
    x, val_final = run(pi_star[:, None], final)
    xs = np.asarray(x, np.float64)
    return xs, pi_star * alpha, val_final[:, 0] * alpha, time.perf_counter() - t0


def nystrom_rank(eigenvalues, n: int, floor: int = 128) -> int:
    """Factor rank for ``Q``: at least its numerical rank (rounded up), so the factor is exact to rounding. A rank
    below it makes FISTA solve a different matrix: the bounds stay valid but become useless."""
    ev = np.asarray(eigenvalues)
    r = int(np.sum(ev > 1e-12 * max(float(ev.max()), 1e-300)))
    return int(min(n, max(floor, 32 * ((r + 8 + 31) // 32))))


def prove_batch(problems, *, check=("rigorous", "c"), qbin: str | None = None, coarse=500, final=3000,
                rank: int | None = None) -> list[Result]:
    """Certify every problem in ``problems``; all must share one ``Q`` and be ``eligible``. ``rank`` defaults to
    ``nystrom_rank``."""
    Q = problems[0].Q
    ev = np.linalg.eigvalsh(Q)
    lam_min = float(ev[0])
    rank = rank or nystrom_rank(ev, Q.shape[0])
    for p in problems:
        if p.Q is not Q and not np.array_equal(p.Q, Q):
            raise UnsupportedProblemError("prove_batch needs every problem on the same covariance")
        ok, why = eligible(p, lam_min)
        if not ok:
            raise UnsupportedProblemError(f"{p.name or 'problem'} is not eligible for the batched path: {why}")
    xs, pis, duals, t_solve = solve_group(problems, coarse=coarse, final=final, rank=rank)
    out = []
    for p, x_hat, pi_star, dual_ref in zip(problems, xs, pis, duals):
        t = {"solve_s": t_solve / len(problems), "extract_s": 0.0, "check_s": 0.0}
        for margin in MARGIN_LADDER:
            t0 = time.perf_counter()
            wit = extract(p, x_hat, margin, pi_star)
            t["extract_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            verdicts = {m: verify(Certificate(p, wit), m, qbin=qbin if m == "c" else None) for m in check}
            t["check_s"] += time.perf_counter() - t0
            if all(v.proved for v in verdicts.values()):
                break
        # escalate a cell whose bound falls short of the solver's own dual estimate
        if np.isfinite(dual_ref) and wit.claimed_bound < dual_ref - 1e-3 * abs(dual_ref):
            t0 = time.perf_counter()
            xs2, pis2, _, _ = solve_group([p], coarse=4 * coarse, final=12 * final, rank=rank)
            wit2 = extract(p, xs2[0], margin, pis2[0])
            t["extract_s"] += time.perf_counter() - t0
            if wit2.claimed_bound > wit.claimed_bound:
                t0 = time.perf_counter()
                v2 = {m: verify(Certificate(p, wit2), m, qbin=qbin if m == "c" else None) for m in check}
                t["check_s"] += time.perf_counter() - t0
                if all(v.proved for v in v2.values()) or not check:
                    wit, verdicts = wit2, v2
        out.append(Result(Certificate(p, wit), verdicts, "zero (batched)", float(dual_ref),
                          relaxation_point=np.asarray(x_hat, np.float64), margin=margin, timings=t))
    return out
