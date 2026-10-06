"""The default relaxation solver: FISTA on the capped simplex, for any diagonal split ``d >= 0`` (JAX; CPU or GPU).

    min_x  x'Ax + phi(x) - pi (mu'x - rho)    over   {1'x = 1, 0 <= x <= hi}          (A = Q - diag d)

    phi(x) = min_y { sum_i d_i x_i^2 / y_i :  x_i/hi <= y_i <= min(1, x_i/lo),  1'y <= K }

This is the perspective relaxation with ``y`` eliminated. It requires ``K * min(hi) >= 1``: then the relaxed
cardinality row ``sum x_i/hi_i <= K`` is implied by the budget and the feasible ``x`` are the capped simplex. The
return row is dualised by a scalar ``pi``, located by bracketing the sign change of ``mu'x(pi) - rho``.

``phi`` is a water-filling problem: stationarity gives ``y_i = clip(s sqrt(d_i) x_i, l_i, u_i)`` with one scalar ``s``
(``tau = 1/s^2`` is the multiplier of ``1'y <= K``), found exactly from one sort of the 2n breakpoints of the
piecewise-linear ``sum_i y_i(s) = K``. Its gradient, by regime:

    interior   l < y < u            2 sqrt(d_i) / s
    at lower   y = x/hi             d_i hi + tau/hi
    at upper   y = 1   (x >= lo)    2 d_i x_i
    at upper   y = x/lo (x < lo)    d_i lo + tau/lo
"""
from __future__ import annotations

import functools

import numpy as np

from heurics_cert.errors import UnsupportedProblemError
from heurics_cert.prover.relaxation import Relaxation


@functools.lru_cache(maxsize=1)
def kernels():
    """The compiled FISTA program (built on first use)."""
    import jax
    import jax.numpy as jnp
    P = jax.lax.Precision.HIGHEST

    def project(v, hi, steps=40):
        """Projection onto ``{0 <= x <= hi, 1'x = 1}``: bisection on the shift (partly unrolled for compile time)."""
        a = jnp.min(v, axis=-1, keepdims=True) - 1.0
        b = jnp.max(v, axis=-1, keepdims=True)

        def body(_, t):
            a, b = t
            m = 0.5 * (a + b)
            s = jnp.sum(jnp.clip(v - m, 0.0, hi), axis=-1, keepdims=True)
            return (jnp.where(s > 1.0, m, a), jnp.where(s > 1.0, b, m))

        a, b = jax.lax.fori_loop(0, steps, body, (a, b), unroll=4)
        return jnp.clip(v - 0.5 * (a + b), 0.0, hi)

    def phi(x, sd, d, lo, hi, K):
        """``(phi value, gradient, y)`` per row, ``sd = sqrt(d)``.

        ``S(s) = sum_i clip(s a_i, l_i, u_i)`` (``a = sd * x``) is piecewise linear and nondecreasing, with slope
        changes ``+a_i`` at ``l_i/a_i`` and ``-a_i`` at ``u_i/a_i``; sorting the breakpoints gives ``S`` at each one
        and the root of ``S(s) = K`` by linear interpolation."""
        at_or_above_lo = x >= lo                         # u = 1 there, u = x/lo below (lo = 0: always u = 1)
        lo_safe = jnp.where(lo > 0, lo, 1.0)
        hi_safe = jnp.where(hi > 0, hi, 1.0)             # hi = 0 excludes an asset (x = 0 there)
        u = jnp.where(at_or_above_lo, 1.0, x / lo_safe)
        l = x / hi_safe
        a = sd * x
        binding = jnp.sum(u, axis=-1, keepdims=True) > K
        act = a > 0
        big = jnp.asarray(1e30, x.dtype)
        b_lo = jnp.where(act, l / jnp.where(act, a, 1.0), big)          # slope +a starts
        b_up = jnp.where(act, u / jnp.where(act, a, 1.0), big)          # slope -a starts
        bp = jnp.concatenate([b_lo, b_up], axis=-1)
        ds = jnp.concatenate([jnp.where(act, a, 0.0), jnp.where(act, -a, 0.0)], axis=-1)
        order = jnp.argsort(bp, axis=-1)
        bp = jnp.take_along_axis(bp, order, axis=-1)
        ds = jnp.take_along_axis(ds, order, axis=-1)
        slope_after = jnp.cumsum(ds, axis=-1)                           # slope on [bp_k, bp_{k+1})
        seg = jnp.diff(bp, axis=-1, prepend=bp[..., :1])                # bp_k - bp_{k-1}
        slope_before = slope_after - ds
        S_at = jnp.sum(l, axis=-1, keepdims=True) + jnp.cumsum(slope_before * jnp.where(bp < big, seg, 0.0),
                                                                axis=-1)
        k = jnp.sum(S_at < K, axis=-1, keepdims=True) - 1                # last breakpoint with S < K
        k = jnp.clip(k, 0, bp.shape[-1] - 1)
        S_k = jnp.take_along_axis(S_at, k, axis=-1)
        b_k = jnp.take_along_axis(bp, k, axis=-1)
        sl = jnp.take_along_axis(slope_after, k, axis=-1)
        s = b_k + (K - S_k) / jnp.where(sl > 0, sl, 1.0)
        s = jnp.maximum(s, 1e-30)
        r = s * sd                                        # regimes are scale-free in x: compare r with 1/hi, 1/lo
        lower = r * hi <= 1.0
        upper = jnp.where(at_or_above_lo, r * x >= 1.0, r * lo_safe >= 1.0)
        y_b = jnp.where(lower, l, jnp.where(upper, u, s * a))
        y = jnp.where(binding, y_b, u)
        tau = jnp.where(binding, 1.0 / (s * s), 0.0)
        g_up = jnp.where(at_or_above_lo, 2.0 * d * x, d * lo + tau / lo_safe)
        g_b = jnp.where(lower, d * hi + tau / hi_safe, jnp.where(upper, g_up, 2.0 * sd / s))
        g = jnp.where(binding, g_b, g_up)
        val = jnp.sum(jnp.where(y > 0, d * x * x / jnp.where(y > 0, y, 1.0), 0.0), axis=-1)
        return val, g, y

    @functools.partial(jax.jit, static_argnums=(8,))
    def fista(A, d, lo, hi, K, mu, rho, pis, iters):
        """One row per value of ``pi``. Monotone FISTA: a step that raises the objective restarts that row's momentum
        and halves its step; a successful step grows it by 1.05."""
        sd = jnp.sqrt(jnp.maximum(d, 0.0))
        v = jnp.ones((A.shape[0],), A.dtype) / jnp.sqrt(A.shape[0])
        v = jax.lax.fori_loop(0, 50, lambda _, v: (lambda w: w / jnp.linalg.norm(w))(
            jnp.matmul(A, v, precision=P)), v)
        lam = jnp.vdot(v, jnp.matmul(A, v, precision=P))
        # smallest positive weight scale over the assets that can be held (hi = 0 marks an excluded asset)
        L0 = 2.0 * lam + 2.0 * jnp.max(d) * jnp.max(hi) / jnp.maximum(
            jnp.min(jnp.where(hi > 0, jnp.where(lo > 0, lo, hi), jnp.inf)), 1e-12)
        R = pis.shape[0]
        g_lin = pis[:, None] * mu

        def F(x):
            val, g, y = phi(x, sd, d, lo, hi, K)
            Ax = jnp.matmul(x, A, precision=P)
            return jnp.sum(x * Ax, axis=-1) + val - jnp.sum(x * g_lin, axis=-1), 2.0 * Ax + g - g_lin, y

        x0 = project(jnp.zeros((R, A.shape[0]), A.dtype) + 1.0 / A.shape[0], hi)
        f0, _, _ = F(x0)

        def body(_, st):
            x, z, t, step, fx = st
            _, gz, _ = F(z)
            xn = project(z - step * gz, hi)
            fn, _, _ = F(xn)
            worse = (fn > fx)[:, None]
            xn = jnp.where(worse, x, xn)
            fn = jnp.where(worse[:, 0], fx, fn)
            step = jnp.where(worse, 0.5 * step, jnp.minimum(1.05 * step, 1.0 / L0 * 64.0))
            restart = worse | (jnp.sum((z - xn) * (xn - x), axis=-1, keepdims=True) > 0.0)
            t = jnp.where(restart, 1.0, t)
            tn = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            zn = xn + jnp.where(restart, 0.0, (t - 1.0) / tn) * (xn - x)
            return (xn, zn, tn, step, fn)

        st = (x0, x0, jnp.ones((R, 1), A.dtype), jnp.full((R, 1), 1.0 / L0, A.dtype), f0)
        x, _, _, _, fx = jax.lax.fori_loop(0, iters, body, st)
        _, g, y = F(x)
        return x, fx + pis * rho, g, y

    return fista


def solve(problem, d, *, grid: int = 24, rounds: int = 3, coarse: int = 500, final: int = 4000,
          dtype=None, pi_hint: float | None = None) -> Relaxation | None:
    """The relaxation at split ``d``. ``value`` is the objective at the returned point; ``duals`` carries the budget
    multiplier read off the free coordinates and the located ``pi``. ``pi_hint`` warm-starts the bracket."""
    import jax.numpy as jnp
    hi_v = np.asarray(problem.hi, np.float64)
    if problem.K * float(hi_v.min()) < 1.0 or float(hi_v.sum()) < 1.0:
        raise UnsupportedProblemError("the FISTA solver needs K * min(hi) >= 1 (the relaxed cardinality row implied "
                                      "by the budget); use Params(solver='clarabel')")
    fista = kernels()
    Q, mu = problem.Q, problem.mu
    d = np.maximum(np.asarray(d, np.float64), 0.0)
    alpha = float(np.mean(np.diag(Q)))                    # objectives are ~1e-4..1e-6: scale them
    mscale = max(float(np.max(np.abs(mu))), 1e-12)
    f = dtype or jnp.float32
    A = jnp.asarray((Q - np.diag(d)) / alpha, f)
    args = (A, jnp.asarray(d / alpha, f), jnp.asarray(problem.lo, f), jnp.asarray(problem.hi, f),
            float(problem.K), jnp.asarray(mu / mscale, f), float(problem.rho / mscale))

    def run(pis, iters):
        x, v, g, y = fista(*args, jnp.asarray(pis, f), iters)
        return tuple(np.asarray(a, np.float64) for a in (x, v, g, y))

    log_grid = np.concatenate([[0.0], np.logspace(-4.0, 4.0, grid - 1)])
    if pi_hint is not None:                               # warm start: one bracket round around the hint
        h = max(float(pi_hint) * mscale / alpha, 0.0)
        pis = np.concatenate([[0.0], np.linspace(0.5 * h, 1.5 * h + 1e-12, grid - 1)]) if h > 0 else log_grid
        rounds = 1
    else:
        pis = log_grid
    mus, rhos = np.asarray(mu / mscale, np.float64), float(problem.rho / mscale)

    def bracket(pis, x):
        """The bracket containing the root of ``r(pi) = mu'x(pi) - rho``, which is nondecreasing in ``pi``. (The grid
        point with the largest dual value is not used: the dual is flat at its maximum and noise picks badly.)"""
        r = x @ mus - rhos
        if pis[0] == 0.0 and r[0] >= 0.0:
            return 0.0, 0.0                              # the return row is slack at pi = 0
        ok = np.flatnonzero(r >= 0.0)
        if ok.size == 0:
            return float(pis[-1]), float(pis[-1]) * 10.0 + 1.0   # root beyond the grid: widen
        k = int(ok[0])
        return float(pis[max(k - 1, 0)]), float(pis[k])

    x_all, val, _, _ = run(pis, coarse)
    lo_, hi_ = bracket(pis, x_all)
    for rnd in range(rounds + 1):
        if hi_ <= lo_:
            break
        pis = np.linspace(lo_, hi_, grid)
        x_all, val, _, _ = run(pis, coarse if rnd < rounds else final // 2)
        lo_, hi_ = bracket(pis, x_all)
    pi_star = float(hi_)                                  # the feasible side of the bracket
    x, val, g, y = run(np.array([pi_star]), final)
    x, g, y = x[0], g[0], y[0]
    free = (x > 1e-9) & (x < hi_v - 1e-9)
    nu = float(np.median(g[free])) if free.any() else 0.0
    obj = float(x @ (Q - np.diag(d)) @ x) + float(np.sum(np.where(y > 0, d * x * x / np.where(y > 0, y, 1.0), 0.0)))
    return Relaxation(x=x, y=y, value=obj, status="fista",
                      duals=dict(nu=nu * alpha, pi=pi_star * alpha / mscale, lam=0.0))
