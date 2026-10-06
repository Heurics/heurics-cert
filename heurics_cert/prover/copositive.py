"""Copositive certificates (CCPO-CERT/2): a split that escapes the diagonal ceiling on singular covariances.

For ``G >= 0`` entrywise and ``sigma`` with ``Q + sigma 11' - G`` PSD, every feasible ``x`` (``x >= 0``, ``1'x = 1``)
satisfies ``x'Qx >= x'(Q + sigma 11' - G)x + sum_i G_ii x_i^2 - sigma``: a diagonal-split bound on a modified matrix.
Off-diagonal mass in ``G`` buys a positive split even where every diagonal split of ``Q`` is zero.

``G`` comes from the doubly-nonnegative (DNN) relaxation, solved by consensus ADMM (GPU, float32; ``jax`` extra):

    min <Q, X>  s.t.  Y = [[1, x'], [x, X]] PSD,  X >= 0,  1'X1 = 1,  1'x = 1,  mu'x >= rho,  0 <= x <= hi,
                      diag X <= hi x,   x_i^2 <= X_ii y_i,  0 <= y <= 1,  1'y <= K

with blocks for the PSD cone, the polyhedral constraints, the perspective cone and the return row. The solver is
untrusted: the certificate is extracted from its dual matrix and checked like any other. It converges well up to
n ~ 250; ``prove_copositive`` warm-starts at the diagonal certificate and keeps whichever is better.
"""
from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from heurics_cert.certificates.certificate import Certificate
from heurics_cert.models.problem import Problem
from heurics_cert.params import DEFAULT, Params
from heurics_cert.prover import tangent
from heurics_cert.prover.result import Result

log = logging.getLogger(__name__)


def _bisect(f, lo, hi, it=40):
    """Root of a nonincreasing scalar ``f`` on ``[lo, hi]``."""
    def body(_, ab):
        a, b = ab
        m = 0.5 * (a + b)
        pos = f(m) > 0
        return jnp.where(pos, m, a), jnp.where(pos, b, m)
    a, b = jax.lax.fori_loop(0, it, body, (lo, hi))
    return 0.5 * (a + b)


def project_polyhedral(V, mu, rho, hi):
    """Project the symmetric (n+1) x (n+1) ``V`` onto ``{Y00=1, X>=0, 1'X1=1, 1'x=1, 0<=x<=hi, diag X <= hi*x}``
    (the return row is its own block, so ``rho`` is unused here).

    x appears twice (row and column): its Frobenius weight is 2. The diag/x coupling diag X <= hi x is per-i 2-D;
    sums are handled by scalar multipliers found by bisection (tau for 1'X1, a for 1'x, b>=0 for mu'x)."""
    n = V.shape[0] - 1
    xv = 0.5 * (V[0, 1:] + V[1:, 0])
    Xv = V[1:, 1:]
    dv = jnp.diag(Xv)
    off = 1.0 - jnp.eye(n)

    def pair(p, q):
        # project (x=p with weight 2, delta=q with weight 1) onto {0<=x<=hi, 0<=delta<=hi*x}
        x0 = jnp.clip(p, 0.0, hi)
        d0 = jnp.clip(q, 0.0, hi * x0)
        # if q > hi*p: project onto the line delta = hi x under metric diag(2,1): minimise 2(x-p)^2 + (hi x - q)^2
        xl = jnp.clip((2 * p + hi * q) / (2 + hi * hi), 0.0, hi)
        use = q > hi * jnp.clip(p, 0.0, hi)
        return jnp.where(use, xl, x0), jnp.where(use, hi * xl, d0)

    def x_delta(tau, a, b):
        # stationarity: x gets -(a + b mu)/2 shift per copy (weight 2 => shift (a+b mu)/2), delta gets -tau
        return pair(xv - 0.5 * (a + b * mu), dv - tau)

    def sums(tau, a, b):
        x, dl = x_delta(tau, a, b)
        Xo = jnp.maximum(Xv - tau, 0.0) * off
        return x, dl, Xo

    def solve_tau(a, b):
        f = lambda t: (lambda x, dl, Xo: jnp.sum(Xo) + jnp.sum(dl) - 1.0)(*sums(t, a, b))
        return _bisect(f, jnp.min(Xv) - 2.0, jnp.max(Xv) + 2.0)

    def solve_a(b):
        # 1'x = 1 given b (tau does not enter x)
        f = lambda a: jnp.sum(x_delta(0.0, a, b)[0]) - 1.0
        half = 4.0 * jnp.max(jnp.abs(xv)) + 4.0 + 2 * jnp.abs(b) * jnp.max(jnp.abs(mu))
        return _bisect(f, -half, half)

    b = 0.0
    a = solve_a(b)
    tau = solve_tau(a, b)
    x, dl, Xo = sums(tau, a, b)
    Y = jnp.zeros_like(V)
    Y = Y.at[0, 0].set(1.0).at[0, 1:].set(x).at[1:, 0].set(x)
    Y = Y.at[1:, 1:].set(Xo + jnp.diag(dl))
    return Y


def project_soc(t, u):
    """Project (t, u) onto {||u|| <= t}, u in R^2, vectorised over the leading axis."""
    nu = jnp.linalg.norm(u, axis=-1)
    inside = nu <= t
    polar = nu <= -t
    s = 0.5 * (t + nu)
    scale = jnp.where(nu > 0, s / jnp.maximum(nu, 1e-30), 0.0)
    tp = jnp.where(inside, t, jnp.where(polar, 0.0, s))
    up = jnp.where(inside[:, None], u, jnp.where(polar[:, None], 0.0, u * scale[:, None]))
    return tp, up


def project_cone(xa, da, ya, K, rounds=3):
    """Project (x, delta, y) (x weight 2) onto {x^2 <= delta y, 0 <= y <= 1, 1'y <= K} approximately:
    Dykstra between the rotated cone (in the weighted metric via variable z = sqrt2 x) and {y <= 1, 1'y <= K}."""
    r2 = jnp.sqrt(2.0)
    z, d, y = r2 * xa, da, ya
    pz, pd, py = jnp.zeros_like(z), jnp.zeros_like(d), jnp.zeros_like(y)
    qy = jnp.zeros_like(y)
    for _ in range(rounds):
        # rotated cone: (z/sqrt2)^2 <= d y  <=>  ||(z*sqrt2/ sqrt2 ... )||: x^2 <= d y <=> ||(2x, d-y)|| <= d+y
        zz, dd, yy = z + pz, d + pd, y + py
        xx = zz / r2
        t, u = project_soc((dd + yy) / r2, jnp.stack([r2 * xx, (dd - yy) / r2], -1))
        # map back: t = (d+y)/sqrt2, u0 = sqrt2 x, u1 = (d-y)/sqrt2
        nd = (t + u[:, 1]) / r2
        ny = (t - u[:, 1]) / r2
        nz = u[:, 0]
        pz, pd, py = zz - nz, dd - nd, yy - ny
        z, d, y = nz, nd, ny
        # {y <= 1, sum y <= K} (y >= 0 is implied by the cone)
        yy = y + qy
        th = jnp.where(jnp.sum(jnp.minimum(yy, 1.0)) <= K, 0.0,
                       _bisect(lambda s: jnp.sum(jnp.minimum(yy - s, 1.0)) - K, 0.0, jnp.max(yy) + 1.0))
        ny = jnp.minimum(yy - th, 1.0)
        qy = yy - ny
        y = ny
    return z / r2, d, y


def admm(Q, mu, lo, hi, K, rho, *, iters=400, rho_pen=1.0, log_every=0, x0=None, b3_rounds=3, balance=True,
         A0=None, w0=None, beta0=None):
    """Consensus ADMM for the doubly-nonnegative relaxation. Returns ``dict(A, x, y, obj, hist, seconds)``: ``A`` is
    the PSD dual matrix of the matrix block (model units), ``x`` the primal point.

    ``A0, w0, beta0`` warm-start the dual at a diagonal certificate (``A0 = Q - diag(d)``), so iterate 0 reproduces
    it."""
    n = Q.shape[0]
    s = float(np.trace(Q) / n)
    msc = 1.0 / float(np.abs(mu).max())
    Qs = jnp.asarray(Q / s, jnp.float32)
    mu_j = jnp.asarray(mu * msc, jnp.float32)
    rho_j = float(rho * msc)
    hi_j = jnp.asarray(hi, jnp.float32)
    C = jnp.zeros((n + 1, n + 1), jnp.float32).at[1:, 1:].set(Qs)
    G = jnp.zeros((n + 1, n + 1), jnp.float32).at[0, 0].set(1.0)
    xs = jnp.full(n, 1.0 / n, jnp.float32) if x0 is None else jnp.asarray(x0, jnp.float32)
    G = G.at[0, 1:].set(xs).at[1:, 0].set(xs).at[1:, 1:].set(jnp.outer(xs, xs))
    U1, U2 = jnp.zeros_like(G), jnp.zeros_like(G)
    if A0 is not None:
        # the dual PSD matrix of block 1 is S = C + rho U1 (X block A, border the tangent (w, beta));
        # U1 = (S_A - C)/rho makes iterate 0 reproduce the diagonal certificate
        S_A = jnp.zeros_like(G).at[1:, 1:].set(jnp.asarray(A0 / s, jnp.float32))
        if w0 is not None:
            wv = jnp.asarray(np.asarray(w0, np.float64) / s, jnp.float32) * 0.5
            S_A = S_A.at[0, 1:].set(-wv).at[1:, 0].set(-wv)
        if beta0 is not None:
            S_A = S_A.at[0, 0].set(jnp.asarray(-float(beta0) / s, jnp.float32))
        U1 = (S_A - C) / rho_pen
    ux, ud = jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.float32)
    y = jnp.minimum(jnp.full(n, K / n), 1.0)
    u4 = jnp.zeros(n, jnp.float32)
    mu_n2 = float(np.sum((mu * msc) ** 2))

    @jax.jit
    def step(G, U1, U2, ux, ud, u4, y, rp):
        # B1: PSD
        V1 = G - U1 - C / rp
        w, P = jnp.linalg.eigh(0.5 * (V1 + V1.T))
        Y1 = (P * jnp.maximum(w, 0.0)) @ P.T
        # B2: polyhedral
        Y2 = project_polyhedral(G - U2, mu_j, rho_j, hi_j)
        # B3: cone on (x, diag X, y)
        x3, d3, y3 = project_cone(G[0, 1:] - ux, jnp.diag(G)[1:] - ud, y, float(K), rounds=b3_rounds)
        # B4: the return row alone -- projection onto the halfspace {mu'x >= rho}
        a4 = G[0, 1:] - u4
        x4 = a4 + jnp.maximum(rho_j - mu_j @ a4, 0.0) * mu_j / mu_n2
        # global average: entries x (row+col) and diag X have 3 copies
        Gn = 0.5 * ((Y1 + U1) + (Y2 + U2))
        xg = (2 * (Gn[0, 1:]) + (x3 + ux) + (x4 + u4)) / 4.0   # two matrix copies, the cone block, the return row
        dg = (2 * jnp.diag(Gn)[1:] + (d3 + ud)) / 3.0
        Gn = Gn.at[0, 1:].set(xg).at[1:, 0].set(xg)
        Gn = Gn.at[jnp.arange(1, n + 1), jnp.arange(1, n + 1)].set(dg)
        U1n = U1 + Y1 - Gn
        U2n = U2 + Y2 - Gn
        uxn = ux + x3 - Gn[0, 1:]
        udn = ud + d3 - jnp.diag(Gn)[1:]
        u4n = u4 + x4 - Gn[0, 1:]
        r = jnp.linalg.norm(Y1 - Gn) + jnp.linalg.norm(Y2 - Gn)
        sres = rp * jnp.linalg.norm(Gn - G)
        obj = jnp.sum(Qs * Gn[1:, 1:])
        return Gn, U1n, U2n, uxn, udn, u4n, y3, r, sres, obj

    rp = rho_pen
    t0 = time.time()
    hist = []
    for it in range(1, iters + 1):
        G, U1, U2, ux, ud, u4, y, r, sr, obj = step(G, U1, U2, ux, ud, u4, y, rp)
        if balance and it % 10 == 0:                               # residual balancing
            r_, s_ = float(r), float(sr)
            if r_ > 10 * s_:
                rp *= 2
                U1, U2, ux, ud, u4 = U1 / 2, U2 / 2, ux / 2, ud / 2, u4 / 2
            elif s_ > 10 * r_:
                rp /= 2
                U1, U2, ux, ud, u4 = U1 * 2, U2 * 2, ux * 2, ud * 2, u4 * 2
        if log_every and it % log_every == 0:
            hist.append((it, float(obj) * s, float(r), float(sr), time.time() - t0))
            log.info("it %5d  primal obj %.5e  r %.2e  s %.2e  rho %.2e  %.1fs", it, float(obj) * s, float(r),
                     float(sr), rp, time.time() - t0)
    # dual PSD matrix of block 1: S = C + rp U1  ->  its X block is A (scaled units)
    S = C + rp * U1
    w, P = jnp.linalg.eigh(0.5 * (S + S.T))
    S = (P * jnp.maximum(w, 0.0)) @ P.T
    A = np.asarray(S[1:, 1:], np.float64) * s
    x = np.asarray(G[0, 1:], np.float64)
    return dict(A=A, x=x, y=np.asarray(y, np.float64), obj=float(obj) * s, hist=hist, seconds=time.time() - t0)


def certificate_from_dual(problem, A_dual, x_hat, floor: float = -np.inf) -> Certificate | None:
    """A copositive certificate from an ADMM dual matrix, or None when its bound does not exceed ``floor``.

    ``sigma`` and ``G_off`` are read off ``A_dual``; the witness is the best tangent of the modified problem at
    ``x_hat`` and at that problem's own relaxation point. Valid for any ``A_dual``: the checkers decide."""
    from heurics_cert.prover import fista
    Q = np.asarray(problem.Q, np.float64)
    n = problem.n
    off = ~np.eye(n, dtype=bool)
    A = 0.5 * (A_dual + A_dual.T)
    sigma = max(0.0, float(np.max((A - Q)[off])))
    G = np.where(off, Q + sigma - A, 0.0)
    G = np.maximum(0.5 * (G + G.T), 0.0)
    np.fill_diagonal(G, 0.0)
    Qt = 0.5 * ((Q + sigma - G) + (Q + sigma - G).T)
    d = np.diag(Q) + sigma - np.diag(A)
    Pt = Problem(Q=Qt, mu=problem.mu, lo=problem.lo, hi=problem.hi, K=problem.K, rho=problem.rho)
    pts = [np.asarray(x_hat, np.float64)]
    rel = fista.solve(Pt, np.maximum(d, 0.0))
    if rel is not None:
        pts.append(np.asarray(rel.x, np.float64))
    best = None
    for xh in pts:
        wit = tangent.extract(Pt, d, xh)
        if best is None or wit.claimed_bound > best.claimed_bound:
            best = wit
    if best is None or best.claimed_bound - sigma <= floor:
        return None
    return Certificate(problem, best, sigma=sigma, G_off=G)


def prove_copositive(problem, *, base: Result | None = None, params: Params = DEFAULT, iters: int = 800,
                     penalty: float = 2.0, cone_rounds: int = 4) -> Result:
    """Improve a diagonal certificate with a copositive one.

    ``base`` is the diagonal result (``prove(problem)`` when omitted). The ADMM is warm-started there, and the base
    is returned when the copositive certificate does not beat it. Requires ``lo >= 0``."""
    from heurics_cert.prover.pipeline import check, prove
    if base is None:
        base = prove(problem, params=params.replace(check=()))
    wit = base.certificate.witness
    t0 = time.perf_counter()
    r = admm(problem.Q, problem.mu, problem.lo, problem.hi, problem.K, problem.rho, iters=iters, rho_pen=penalty,
             b3_rounds=cone_rounds, A0=problem.Q - np.diag(wit.d), w0=wit.w, beta0=wit.beta_z)
    cert = certificate_from_dual(problem, r["A"], r["x"], floor=base.bound)
    timings = dict(base.timings, copositive_s=time.perf_counter() - t0)
    if cert is None:
        verdicts = base.verdicts or check(base.certificate, params.check, timeout=params.check_timeout)
        return Result(base.certificate, verdicts, base.split, base.relaxation_value, base.relaxation_point,
                      base.ascent_accepted, base.margin, timings)
    verdicts = check(cert, params.check, timeout=params.check_timeout)
    return Result(cert, verdicts, "copositive", float(r["obj"]), np.asarray(r["x"], np.float64),
                  base.ascent_accepted, base.margin, timings)
