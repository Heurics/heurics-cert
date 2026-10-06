"""Optimality certificates for cardinality-constrained mean-CVaR.

    min  CVaR_beta(-Rw) - lam mu'w     s.t.  1'w = 1,  0 <= w_i <= hi_i,  ||w||_0 <= K

``CVaR_beta`` of the scenario losses ``-r_s'w`` is the mean of the worst ``m = ceil((1 - beta) S)``. That tail mean is
``max_{q in Q} q'L`` over ``Q = {0 <= q <= 1/m, 1'q = 1}``, so the objective is ``max_q c(q)'w`` with
``c(q) = -R'q - lam mu``: for any fixed ``q`` it is linear in ``w``, and ``min_w c(q)'w`` over any feasible set is a
valid lower bound (weak duality; an evaluation, not a solve).

Every bound convex in ``w`` is capped at the cardinality-free optimum: for a linear objective the minimum over the
K-sparse set equals the minimum over its convex hull, which is the whole capped simplex once ``ceil(1/max hi) <= K``.
``cvar_convex_ceiling`` computes that ceiling.

The certificate is therefore per support: for a fixed support ``s`` and any ``q``, ``val(s) >= min_{i in s} c_i(q)``.
With a pool of duals ``q_1..q_M`` (``C[i, j] = c_i(q_j)``, ``cvar_pool_costs``), every support's bound is
``max_j min_{i in s} C[i, j]`` (``cvar_support_bounds``). ``certify_cvar`` enumerates supports, discards those whose
bound exceeds the incumbent, refines the pool on the survivors, and returns a proof of optimality or an anytime-valid
bound. Assets with ``C[i, j] >= UB`` for every ``j`` are dead and never enumerated: a support survives exactly when its
live part does, with the same bound.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from ._capped_simplex import project_capped_simplex


# ------------------------------------------------------------------------------------------------
# The scenario-weight simplex `Q`.

def project_tail_simplex(
    v: Float[Array, " S"],
    cap: float,
) -> Float[Array, " S"]:
    """Euclidean projection of `v` onto `Q = {q : sum(q) = 1, 0 <= q <= cap}`, in `O(S log S)`.

    Same closed form as [`heurics_cert.extras._capped_simplex.project_capped_simplex`][] -- `g(tau) = sum(clip(v -
    tau, 0, cap))` is piecewise linear and non-increasing, so evaluating it at its `2S` breakpoints
    brackets the root `g(tau) = 1` inside one linear segment and a single interpolation lands on it
    -- but evaluated **without** the `O(S^2)` breakpoint matrix that version materializes. At the
    scenario counts this module runs at (`S = 10,000` and up) that matrix is 200M entries per
    projection and the projection sits inside an ascent loop, so the difference is between a
    certificate and an out-of-memory error.

    The `O(S log S)` route: sort `v` once, and read `g(tau)` off two `searchsorted`s and a prefix
    sum, since `g(tau) = cap * #{v_i > tau + cap} + sum_{tau < v_i <= tau+cap} (v_i - tau)`.

    Requires `cap * S >= 1` for `Q` to be non-empty, which for `cap = 1/m` with `m <= S` is
    automatic.
    """
    dtype = v.dtype
    S = v.shape[-1]
    capd = jnp.asarray(cap, dtype)

    vs = jnp.sort(v)
    cum = jnp.concatenate([jnp.zeros((1,), dtype), jnp.cumsum(vs)])
    breaks = jnp.sort(jnp.concatenate([v, v - capd]))

    def g(tau):
        hi_i = jnp.searchsorted(vs, tau + capd, side="right")
        lo_i = jnp.searchsorted(vs, tau, side="right")
        n_full = (S - hi_i).astype(dtype)
        n_part = (hi_i - lo_i).astype(dtype)
        sum_part = cum[hi_i] - cum[lo_i]
        return capd * n_full + sum_part - tau * n_part

    gb = g(breaks)
    j = jnp.clip(jnp.sum(gb >= 1.0) - 1, 0, 2 * S - 2)
    b0, b1 = breaks[j], breaks[j + 1]
    g0, g1 = gb[j], gb[j + 1]
    den = g0 - g1
    tau = jnp.where(den > 0.0,
                    b0 + (g0 - 1.0) * (b1 - b0) / jnp.where(den > 0.0, den, 1.0),
                    b0)
    return jnp.clip(v - tau, jnp.zeros((), dtype), capd)


def tail_dual(
    losses: Float[Array, " S"],
    m: int,
) -> Float[Array, " S"]:
    """The `q in Q` attaining `max_q q'losses`: uniform `1/m` on the worst `m` scenarios.

    This is the risk envelope's own maximizer, so it is the natural dual to read off any portfolio
    -- the incumbent's, the equal-weight book's, a single asset's. It is also exactly the "tail
    scenario weights" Gap 2 recovers as the epigraph duals, which is why seeding a pool with them
    costs nothing beyond a `top_k` that has already been computed once.
    """
    _, idx = jax.lax.top_k(losses, m)
    one = jnp.asarray(1.0 / m, losses.dtype)
    return jnp.zeros(losses.shape, losses.dtype).at[idx].set(one)


def cvar_pool_costs(
    R: Float[Array, "S n"],
    mu: Float[Array, " n"],
    lam: float,
    Q: Float[Array, "M S"],
) -> Float[Array, "n M"]:
    """`C[i, j] = c_i(q_j) = -(R' q_j)_i - lam * mu_i` -- the per-asset dual cost for a pool of duals.

    `M` matvecs, computed once. Everything downstream is a reduction over this matrix, which is what
    makes a per-support bound cost `O(K*M)` instead of a solve.
    """
    return -jnp.einsum("sn,ms->nm", R, Q, precision=jax.lax.Precision.HIGHEST) - lam * mu[:, None]


def cvar_support_bounds(
    C: Float[Array, "n M"],
    supports: Int[Array, "B k"],
) -> Float[Array, " B"]:
    """`max_j min_{i in s} C[i, j]` for a batch of supports -- a valid lower bound on each `val(s)`.

    Valid at **any** pool, converged or not: for fixed `q` the objective is linear in `w`, so its
    minimum over the simplex on `s` is attained at a vertex, i.e. at `min_{i in s} c_i(q)`. A
    maximum of valid lower bounds is a valid lower bound, so pooling can only tighten.

    Streamed over the pool with a `scan` rather than written as `max(min(C[supports]))`. The direct
    form materializes a `(B, k, M)` gather, and `B` is a chunk of the support enumeration: at one
    million supports, `K = 10` and a pool of a hundred duals that intermediate is 4 GB and the
    scan's is 40 MB. The arithmetic is identical; only the peak allocation differs, and the direct
    form OOMs at exactly the scale this is for.
    """
    neg_inf = jnp.asarray(-jnp.inf, C.dtype)

    def step(best, col):                      # col: (n,) -- one dual's per-asset costs
        return jnp.maximum(best, jnp.min(col[supports], axis=-1)), None

    best, _ = jax.lax.scan(step, jnp.full((supports.shape[0],), neg_inf, C.dtype), C.T)
    return best


# ------------------------------------------------------------------------------------------------
# The convex ceiling -- the obstruction, measured rather than asserted.

def cvar_convex_ceiling(
    R: Float[Array, "S n"],
    mu: Float[Array, " n"],
    lam: float,
    m: int,
    upper_bound: float,
    *,
    q0: Float[Array, " S"] | None = None,
    iters: int = 2000,
    gamma: float = 1.0,
    level_decay: float | None = None,
) -> tuple[Float[Array, ""], Float[Array, " S"], dict]:
    """`max_{q in Q} min_i c_i(q)` -- the best bound ANY `w`-convex relaxation of this model admits.

    Every iterate is a valid lower bound on the cardinality-free continuous optimum and therefore
    on the `K`-sparse one, so the running maximum returned is valid whether or not the ascent
    converged. At the supremum it **equals** the cardinality-free optimum (minimax over the
    compact convex `Q` and the capped simplex, bilinear objective), and with `hi = 1` that is also
    the value of the LP relaxation, of any cutting-plane bound projected into `w`-space, and of any
    Lagrangian decomposition that keeps `w` convex. So this one number is the ceiling for a whole
    family of proposals at once -- which is the point of computing it before building any of them.

    Projected supergradient ascent: at the argmin asset `i*`, `d/dq of c_{i*}(q) = -R[:, i*]`, one
    `(n, S) @ (S,)` matvec per iteration plus one projection. Steps are Polyak's, with the level
    decayed over the **whole** budget rather than per-step -- a constant per-step decay freezes the
    iteration and reads exactly like convergence.
    """
    dtype = R.dtype
    S, n = R.shape
    cap = 1.0 / m
    upper = jnp.asarray(upper_bound, dtype)
    if level_decay is None:
        level_decay = float(1e-3) ** (1.0 / max(iters, 1))

    if q0 is None:
        q0 = jnp.full((S,), cap, dtype) if m == S else project_tail_simplex(
            jnp.full((S,), 1.0 / S, dtype), cap)
    q0 = project_tail_simplex(jnp.asarray(q0, dtype), cap)

    def evaluate(q):
        c = -(R.T @ q) - lam * mu
        i = jnp.argmin(c)
        return c[i], i

    val0, _ = evaluate(q0)
    level0 = jnp.maximum(upper - val0, jnp.abs(val0) * 0.1 + 1e-12) * 0.5

    def body(_, state):
        q, best, best_q, level = state
        val, i = evaluate(q)
        take = val > best
        best = jnp.where(take, val, best)
        best_q = jnp.where(take, q, best_q)
        sg = -R[:, i]
        nrm = jnp.sum(sg * sg)
        target = jnp.minimum(best + level, upper)
        step = gamma * jnp.maximum(target - val, 0.0) / jnp.maximum(nrm, 1e-30)
        return project_tail_simplex(q + step * sg, cap), best, best_q, level * level_decay

    q, best, best_q, _ = jax.lax.fori_loop(0, iters, body, (q0, val0, q0, level0))
    return best, best_q, {"iters": iters, "init_bound": val0, "last_q": q}


# ------------------------------------------------------------------------------------------------
# Per-support refinement: the dual ascent that tightens a survivor, and the primal that may beat it.

@functools.partial(jax.jit, static_argnames=("m", "iters"))
def cvar_support_ascent(
    R: Float[Array, "S n"],
    mu: Float[Array, " n"],
    lam: float,
    supports: Int[Array, "B k"],
    m: int,
    upper: Float[Array, " B"],
    q0: Float[Array, "B S"],
    *,
    iters: int = 200,
    gamma: float = 1.0,
) -> tuple[Float[Array, " B"], Float[Array, "B S"]]:
    """`max_{q in Q} min_{i in s} c_i(q)`, batched over supports. Every iterate is valid for `s`.

    Unlike [`cvar_convex_ceiling`][] this is **not** capped by the cardinality-free optimum: the
    support is fixed, so nothing about the cardinality constraint is being relaxed and the supremum
    is `val(s)` itself, by strong duality on the (compact, convex, bilinear) restricted minimax. So
    the same ascent that is provably vacuous globally is exactly tight per support -- which is the
    whole reason the certificate is organized by support.
    """
    dtype = R.dtype
    cap = 1.0 / m
    Rs = jnp.take(R, supports, axis=1).transpose(1, 0, 2)         # (B, S, k)
    mus = mu[supports]                                            # (B, k)
    level_decay = float(1e-3) ** (1.0 / max(iters, 1))
    proj = jax.vmap(project_tail_simplex, in_axes=(0, None))

    def evaluate(q):
        c = -jnp.einsum("bsk,bs->bk", Rs, q,
                        precision=jax.lax.Precision.HIGHEST) - lam * mus
        i = jnp.argmin(c, axis=-1)
        return jnp.take_along_axis(c, i[:, None], axis=-1)[:, 0], i

    q0 = proj(jnp.asarray(q0, dtype), cap)
    val0, _ = evaluate(q0)
    level0 = jnp.maximum(upper - val0, jnp.abs(val0) * 0.1 + 1e-12) * 0.5

    def body(_, state):
        q, best, best_q, level = state
        val, i = evaluate(q)
        take = val > best
        best = jnp.where(take, val, best)
        best_q = jnp.where(take[:, None], q, best_q)
        sg = -jnp.take_along_axis(Rs, i[:, None, None], axis=2)[:, :, 0]      # (B, S)
        nrm = jnp.sum(sg * sg, axis=-1)
        target = jnp.minimum(best + level, upper)
        step = gamma * jnp.maximum(target - val, 0.0) / jnp.maximum(nrm, 1e-30)
        return proj(q + step[:, None] * sg, cap), best, best_q, level * level_decay

    _, best, best_q, _ = jax.lax.fori_loop(0, iters, body, (q0, val0, q0, level0))
    return best, best_q


@functools.partial(jax.jit, static_argnames=("m",))
def cvar_saddle_dual(
    R: Float[Array, "S n"],
    supports: Int[Array, "B k"],
    w: Float[Array, "B k"],
    m: int,
) -> Float[Array, "B S"]:
    """The dual `q` implied by a restricted **primal** solution: the tail weights of its own losses.

    This is the cheapest good dual in the module, and it is the one that makes the certifier close.
    `(w, q)` is a saddle point of the restricted bilinear problem when `w` solves it and `q`
    attains the inner maximum at `w` -- and `q` attaining the inner maximum at `w` is exactly
    [`tail_dual`][] of `w`'s own loss vector. At that saddle point `min_{i in s} c_i(q) = val(s)`,
    so one `top_k` yields the dual the per-support ascent spends thousands of iterations chasing.

    **Why not just ascend.** [`cvar_support_ascent`][] maximizes over `q`, which lives in `S`
    dimensions (10,000 here) and is nonsmooth, so a subgradient method zigzags: measured, 2,000
    iterations left the bound 2.4% short of `val(s)` on a 10-asset instance, which was enough to
    keep the *optimal* support alive and block a proof that should have been trivial. The primal
    lives in `k` dimensions -- five, or ten -- and converges far better for the same effort. So the
    honest division of labour is: solve the small side, and read the large side off it.

    Validity does not depend on any of that. The returned `q` is a point of `Q` however it was
    obtained, so `min_{i in s} c_i(q)` is a valid lower bound on `val(s)` whether or not `w` is
    optimal; a bad `w` costs tightness only.
    """
    Rs = jnp.take(R, supports, axis=1).transpose(1, 0, 2)
    losses = -jnp.einsum("bsk,bk->bs", Rs, w, precision=jax.lax.Precision.HIGHEST)
    return jax.vmap(tail_dual, in_axes=(0, None))(losses, m)


@functools.partial(jax.jit, static_argnames=("m", "iters"))
def cvar_support_primal(
    R: Float[Array, "S n"],
    mu: Float[Array, " n"],
    lam: float,
    hi: Float[Array, " n"],
    supports: Int[Array, "B k"],
    m: int,
    *,
    iters: int = 300,
    step0: float = 1.0,
    dual_burn: float = 0.5,
) -> tuple[Float[Array, "B k"], Float[Array, " B"], Float[Array, "B S"]]:
    """The restricted primal on a fixed support, batched -- and the dual that falls out of it.

    Returns `(w, value, q_avg)`: an **upper** bound on `val(s)` from the feasible `w`, and a point
    of `Q` whose `min_{i in s} c_i(q)` is a **lower** bound on the same `val(s)`. One run of one
    subgradient method produces both ends of the certificate for this support.

    Convex but non-smooth (the tail set changes identity at the kinks), so this is projected
    subgradient with best-iterate tracking rather than an accelerated smooth method.

    **Why the dual has to be averaged, and cannot just be read off the best `w`.** The obvious dual
    is [`cvar_saddle_dual`][]: the tail weights of the optimal `w`. It is not tight, and the reason
    is structural rather than numerical -- at the optimum of a CVaR minimization the tail boundary
    is typically **degenerate**, i.e. the `m`-th and `(m+1)`-th losses are equal. Measured on a
    10-asset instance: the exact LP optimum of the *optimal* support has tied boundary losses on
    every support tried, and its uniform-`1/m` tail dual scores `0.0597` against a true value of
    `0.0654` -- 8.7% short, which is enough to keep the optimal support alive and block a proof
    that should be trivial. With a tie the inner maximum has many maximizers and the uniform one is
    not the saddle point; the dual optimum splits mass *across* the tied scenarios. Averaging the
    iterates does that splitting automatically, because the tail set flips between the tied
    scenarios from step to step and the average inherits both.

    No classical solver: the subgradient is one `top_k` and a masked mean, the dual is a
    scatter-add, and the projection is [`heurics_cert.extras._capped_simplex.project_capped_simplex`][] at support
    size `k`, where its `O(k^2)` breakpoint evaluation is free.
    """
    dtype = R.dtype
    B, k = supports.shape
    S = R.shape[0]
    Rs = jnp.take(R, supports, axis=1).transpose(1, 0, 2)         # (B, S, k)
    mus, his = mu[supports], hi[supports]
    lo = jnp.zeros(mus.shape, dtype)
    rows = jnp.arange(B)

    def value(w):
        losses = -jnp.einsum("bsk,bk->bs", Rs, w, precision=jax.lax.Precision.HIGHEST)
        top, idx = jax.lax.top_k(losses, m)
        return jnp.mean(top, axis=-1) - lam * jnp.sum(mus * w, axis=-1), idx

    def subgrad(idx):
        rw = jnp.take_along_axis(Rs, idx[:, :, None], axis=1)     # (B, m, k)
        return -jnp.mean(rw, axis=1) - lam * mus

    w = project_capped_simplex(jnp.full(mus.shape, 1.0 / k, dtype), lo, his)
    f, idx = value(w)
    scale = step0 / jnp.maximum(jnp.linalg.norm(subgrad(idx), axis=-1), 1e-30)
    burn = int(dual_burn * iters)

    def body(t, state):
        w, best_w, best_f, qbar, wsum = state
        _, idx = value(w)
        g = subgrad(idx)
        step = scale / jnp.sqrt(jnp.asarray(t, dtype) + 1.0)
        # Ergodic dual averaging, and this is the part that makes the certificate close. Each
        # iterate's tail set IS a point of `Q` (mass `1/m` on its own worst `m` scenarios), and the
        # step-weighted average of those points converges to a dual-optimal `q` -- the classical
        # primal-dual property of a subgradient method on a saddle problem. Accumulating it costs
        # one scatter-add per iteration and nothing else.
        # ...over the TAIL of the run only. Averaging from iteration zero drags the early, far-from-
        # optimal tail sets into the answer, and those are precisely the iterates whose tail set is
        # wrong. Measured on a 10-asset instance: averaging everything left the OPTIMAL support's
        # dual 3.7% short while every other support sat under 1% -- the one support that has to be
        # closed for a proof to exist was the one the average served worst. Burning the first half
        # costs nothing and is the standard fix.
        live_w = jnp.where(t >= burn, step, jnp.zeros((), dtype))
        qbar = qbar.at[rows[:, None], idx].add((live_w / m)[:, None])
        wsum = wsum + live_w
        nxt = project_capped_simplex(w - step[:, None] * g, lo, his)
        fn, _ = value(nxt)
        better = fn < best_f
        return (nxt, jnp.where(better[:, None], nxt, best_w), jnp.where(better, fn, best_f),
                qbar, wsum)

    _, w, f, qbar, wsum = jax.lax.fori_loop(
        0, iters, body, (w, w, f, jnp.zeros((B, S), dtype), jnp.zeros((B,), dtype)))
    # A convex combination of points of `Q` is a point of `Q`, so the average is feasible by
    # construction and the bound it yields is valid however far the primal is from converged.
    return w, f, qbar / jnp.maximum(wsum, 1e-30)[:, None]


# ------------------------------------------------------------------------------------------------
# Support enumeration.

def cvar_cardinality_free_bound(
    R: Float[Array, "S n"],
    mu: Float[Array, " n"],
    lam: float,
    hi: Float[Array, " n"],
    m: int,
    *,
    iters: int = 100000,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, " S"]]:
    """A valid lower bound on the `K`-sparse mean-CVaR optimum **for every `K` at once**, plus the
    bracket that says how converged it is.

    Returns `(lower, upper, q)`. `lower = min_i c_i(q)` is valid for any `K` because dropping the
    cardinality row only enlarges the feasible set; `upper` is the value of a feasible
    cardinality-free portfolio, so the true ceiling lies in `[lower, upper]` and the bracket width
    is a *measured* convergence statement rather than an iteration count.

    **This is the tightest bound the `w`-convex family admits**, and by the obstruction it is also
    the tightest any LP relaxation, cutting plane or Benders decomposition of this model can reach
    -- so it is the right thing to compare an external solver's dual bound against.

    **Why it is computed from the primal rather than by ascending the dual.** Both routes target
    the same saddle value, but the dual lives in `S` dimensions (10,000 here) and is nonsmooth,
    while the primal lives in `n` (500) and is far better conditioned. Measured on the Suite 2 cell
    `n = 500, S = 10,000`: projected supergradient on `q` reached `-0.00075466` in 17 s and
    plateaued; entropic smoothing of the same ascent reached `-0.00060660`; and reading the
    averaged dual off this primal reached `-0.00060583` with a bracket of `8.1e-08`. The dual is
    read off the primal for the same reason [`cvar_saddle_dual`][] exists -- solve the small side,
    and take the large side from it -- with the averaging that entry describes, which is what
    handles the degenerate tail boundary.

    Validity never depends on convergence: `q` is a point of `Q` however it was produced.
    """
    n = R.shape[1]
    supports = jnp.arange(n)[None, :]
    _w, value, q_avg = cvar_support_primal(R, mu, lam, hi, supports, m, iters=iters)
    q = q_avg[0]
    c = -(R.T @ q) - lam * mu
    return jnp.min(c), value[0], q


def _colex_prefix(n: int, K: int) -> np.ndarray:
    """`prefix[j, x] = sum_{c <= x} C(n-1-c, K-j-1)`; the cumulative counts colex unranking needs."""
    prefix = np.zeros((K, n), dtype=np.float64)
    for j in range(K):
        run = 0
        for x in range(n):
            if n - 1 - x >= 0:
                run += math.comb(n - 1 - x, K - j - 1)
            prefix[j, x] = run
    return prefix


def exact_rank_limit() -> int:
    """The largest subset count this process can unrank **exactly**.

    Colex unranking carries the rank as a float, and a float only represents integers exactly up to
    `2**mantissa`. Without `jax_enable_x64` that ceiling is `2**24 = 16.7M`, and `C(50, 10)` is
    `1.03e10` -- so on the default configuration the ranks past 16.7M would collide silently, the
    enumeration would revisit some subsets and skip others, and the certifier would "prove"
    optimality by never looking at the counterexample. That is the exact failure `serve/certify.py`
    guards with an exhaustiveness test, and it does not announce itself. Callers enumerating more
    than this must set `JAX_ENABLE_X64=1` before importing JAX.
    """
    return (1 << 53) if jax.config.jax_enable_x64 else (1 << 24)


@functools.partial(jax.jit, static_argnames=("n", "K"))
def subset_unrank(ranks, prefix, n: int, K: int):
    """Map colex ranks to the `K`-subsets of `range(n)`, fully vectorized -- `K` `searchsorted`s
    over an `n`-vector for a whole batch, no per-subset control flow (tested against
    `itertools.combinations`).
    """
    B = ranks.shape[0]
    out = jnp.zeros((B, K), dtype=jnp.int32)
    rem = ranks
    start = jnp.zeros(B, dtype=jnp.int32)
    for j in range(K):
        p = prefix[j]
        base = jnp.where(start > 0, p[jnp.maximum(start - 1, 0)], 0.0)
        x = jnp.searchsorted(p, base + rem, side="right").astype(jnp.int32)
        x = jnp.clip(x, start, n - 1)
        prev = jnp.where(x > 0, p[jnp.maximum(x - 1, 0)], 0.0)
        rem = rem - (prev - base)
        out = out.at[:, j].set(x)
        start = x + 1
    return out


def _scan_live_subsets(C, live, t, K, n_dead, ub_cut, chunk, keep_cap):
    """Every size-`t` subset of the live universe whose pooled bound is below `ub_cut`.

    Returns `(kept_subsets, kept_bounds, n_survivor_supports, n_enumerated, min_bound)` where the
    third entry counts **full size-`K` supports**: each surviving `T` stands for `C(n_dead, K - t)`
    of them, which is the number the kill line is written against.

    `min_bound` is accumulated over **every** survivor, not only the ones `keep_cap` retains for
    refinement. That distinction is soundness, not bookkeeping: the certified bound is the minimum
    over survivors, so computing it from a truncated list would report a bound that is too *high*
    -- i.e. a certificate claiming more than was proved. `keep_cap` may cost tightening effort; it
    must never touch the number that is quoted.
    """
    nl = int(live.shape[0])
    total = math.comb(nl, t)
    prefix = jnp.asarray(_colex_prefix(nl, t))
    fill = math.comb(n_dead, K - t)
    kept_idx, kept_bnd, n_surv = [], [], 0
    min_bound = float("inf")
    for lo_r in range(0, total, chunk):
        hi_r = min(lo_r + chunk, total)
        rank_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        ranks = jnp.arange(lo_r, hi_r, dtype=rank_dtype)
        sub = subset_unrank(ranks, prefix, nl, t)
        supports = live[sub]
        bnd = cvar_support_bounds(C, supports)
        surv = jnp.nonzero(bnd < ub_cut)[0]
        if surv.size:
            n_surv += int(surv.shape[0]) * fill
            min_bound = min(min_bound, float(jnp.min(bnd[surv])))
            if sum(int(a.shape[0]) for a in kept_idx) < keep_cap:
                kept_idx.append(np.asarray(supports[surv]))
                kept_bnd.append(np.asarray(bnd[surv]))
    idx = (np.concatenate(kept_idx) if kept_idx else np.zeros((0, t), np.int64))
    bnd = (np.concatenate(kept_bnd) if kept_bnd else np.zeros((0,), np.float64))
    return idx, bnd, n_surv, total, min_bound


def certify_cvar(
    R,
    mu,
    lam: float,
    hi,
    m: int,
    K: int,
    upper_bound: float,
    incumbent_w=None,
    *,
    pool_rounds: int = 2,
    refine_top: int = 32,
    ascent_iters: int = 200,
    primal_iters: int = 2000,
    tol: float = 1e-12,
    chunk: int = 1 << 22,
    keep_cap: int = 1 << 16,
    enumeration_cap: float = 4e9,
) -> tuple[bool, dict]:
    """Certify (or bound) the cardinality-constrained mean-CVaR optimum with no classical solver.

    Returns `(proven, info)`. `proven=True` means every support of size `<= K` was either eliminated
    -- its pooled dual bound is at or above the incumbent, so it cannot beat it -- or exactly
    resolved, which together with a feasible incumbent is a **proof** of global optimality. On
    `False`, `info["certified_bound"]` is still a valid lower bound on the global optimum: it is the
    minimum pooled bound over the supports that survived, and survivors are unresolved rather than
    counterexamples.

    The loop is: seed a pool of duals from portfolios already in hand, eliminate, refine the pool
    against the loosest survivors (and try to improve `UB` off their primals), eliminate again.
    Elimination only ever *adds* information, so stopping early costs tightness and never validity.

    **Only size-`K` supports are enumerated, and that covers size `< K` too.** Enlarging a support
    enlarges its feasible set, so `val(s') >= val(s)` whenever `s' ⊆ s`; every smaller support
    extends to some size-`K` one. So if every size-`K` support has `bound >= UB - tol` then every
    smaller support has `val >= UB - tol` as well, and the optimum -- attained at size `K` for the
    same reason -- is bounded by the size-`K` minimum. This is the only place the enumeration is
    allowed to skip anything, and it skips nothing that can change the answer.

    `proven=True` says "no support beats `UB`". That is only half a certificate on its own: it is
    *trivially* true for a `UB` below the optimum. The other half is that `UB` is attained by a
    feasible point, which the caller supplies -- pass the incumbent's objective, and the pair is a
    proof.

    `enumeration_cap` guards the one thing that can invalidate the answer -- an enumeration that did
    not finish. If a level exceeds it the level is not scanned and `info["exhaustive"]` is `False`,
    at which point `proven` is forced to `False` and the reported bound is `-inf`: "we ran out of
    budget" must never be reported as "nothing beats the incumbent".
    """
    dtype = R.dtype
    S, n = R.shape
    ub = float(upper_bound)

    # ---- seed pool. Every entry is the tail dual of a portfolio already in hand, so the pool costs
    # a top_k each and no solve at all.
    def tail_of(w):
        return tail_dual(-(R @ jnp.asarray(w, dtype)), m)

    seeds = [tail_of(jnp.full((n,), 1.0 / n, dtype))]
    if incumbent_w is not None:
        seeds.insert(0, tail_of(incumbent_w))
        top = np.argsort(-np.asarray(incumbent_w))[:K]
    else:
        top = np.arange(min(K, n))
    for i in top:                       # each held asset's own worst-m scenarios
        e = jnp.zeros((n,), dtype).at[int(i)].set(1.0)
        seeds.append(tail_of(e))
    pool = jnp.stack(seeds)
    if incumbent_w is not None and len(top) == K:
        # The most valuable seed in the pool: the ergodic dual of the incumbent's OWN support. The
        # incumbent's plain tail dual is already in `seeds`, but the tail boundary is degenerate at
        # a CVaR optimum, so the plain one is systematically short (see `cvar_support_primal`) --
        # and it is short on exactly the support that has to be eliminated for a proof to exist.
        inc_sup = jnp.asarray(np.sort(top)[None, :], jnp.int32)
        _w0, _f0, q0_avg = cvar_support_primal(R, mu, lam, hi, inc_sup, m, iters=primal_iters)
        pool = jnp.concatenate([pool, q0_avg])
    C = cvar_pool_costs(R, mu, lam, pool)

    info: dict = {"pool_size": [], "survivor_supports": [], "live": [], "enumerated": [],
                  "upper_bound": ub, "rounds": 0}
    exhaustive = True
    best_bound = float("inf")

    for rnd in range(max(pool_rounds, 1)):
        ub_cut = ub - tol
        Cnp = np.asarray(C)
        live_mask = np.any(Cnp < ub_cut, axis=1)
        live = np.nonzero(live_mask)[0]
        n_dead = int(n - live.size)
        live_j = jnp.asarray(live, jnp.int32)

        keep_idx, keep_bnd, n_surv, n_enum = [], [], 0, 0
        min_bound = float("inf")
        t_lo = max(1, K - n_dead)
        t_hi = min(K, int(live.size))
        limit = exact_rank_limit()
        for t in range(t_lo, t_hi + 1):
            total = math.comb(int(live.size), t)
            if total > enumeration_cap or total > limit:
                # Either too big to scan, or too big to *rank exactly* -- and the second is the
                # dangerous one, because it would not fail, it would silently skip subsets.
                exhaustive = False
                info.setdefault("skipped_levels", []).append(
                    {"round": rnd, "t": t, "count": total,
                     "reason": "rank_precision" if total > limit else "enumeration_cap"})
                continue
            idx, bnd, ns, ne, mb = _scan_live_subsets(C, live_j, t, K, n_dead, ub_cut, chunk,
                                                      keep_cap)
            n_enum += ne
            n_surv += ns
            min_bound = min(min_bound, mb)
            if idx.shape[0]:
                keep_idx.append(idx)
                keep_bnd.append(bnd)
        info["pool_size"].append(int(pool.shape[0]))
        info["live"].append(int(live.size))
        info["survivor_supports"].append(int(n_surv))
        info["enumerated"].append(int(n_enum))
        info["rounds"] = rnd + 1

        best_bound = min_bound
        if n_surv == 0 or rnd == max(pool_rounds, 1) - 1:
            break

        # ---- refine: the loosest survivors get their own dual ascent, and their primal is tried
        # as a better incumbent. Both feed the next elimination pass.
        full = _pad_to_support(keep_idx, K, live, n, Cnp)
        if full.shape[0] == 0:
            break
        order = np.argsort(cvar_support_bounds(C, jnp.asarray(full, jnp.int32)))[:refine_top]
        batch = jnp.asarray(full[np.asarray(order)], jnp.int32)
        q0 = jnp.broadcast_to(pool[0], (batch.shape[0], S))
        _best, qstar = cvar_support_ascent(R, mu, lam, batch, m,
                                           jnp.full((batch.shape[0],), ub, dtype), q0,
                                           iters=ascent_iters)
        w_s, f_s, q_avg = cvar_support_primal(R, mu, lam, hi, batch, m, iters=primal_iters)
        # Two duals per refined support, both free and both valid, so the pool takes both:
        #   * `q_avg`  -- the primal's ergodic dual average, which is what actually closes supports
        #                 because it splits mass across the tied tail boundary (see the primal's
        #                 docstring); and
        #   * `q_saddle` -- the tail dual of the best `w`, exact when the boundary is NOT tied.
        # The ascent's `qstar` is kept too. A maximum over more valid duals is never looser.
        q_saddle = cvar_saddle_dual(R, batch, w_s, m)
        # The primal is a feasible point's value, so it is a genuine upper bound -- but it is
        # computed in the same float32 as everything else here, so it is only accepted when it
        # improves by more than `tol`. Chasing a sub-tolerance "improvement" would tighten the
        # elimination cut on arithmetic noise, and eliminating on noise is how a proof becomes
        # a false claim.
        best_primal = float(jnp.min(f_s))
        if best_primal < ub - tol:
            ub = best_primal
            info["upper_bound"] = ub
        pool = jnp.concatenate([pool, qstar, q_avg, q_saddle])
        C = cvar_pool_costs(R, mu, lam, pool)

    # The global bound is the minimum pooled bound over every support. Eliminated supports all sit
    # at or above `ub - tol` by construction, so when survivors exist the minimum is theirs, and
    # when none do it is `ub - tol` -- which, paired with a feasible incumbent at `ub`, pins the
    # optimum inside `[ub - tol, ub]`.
    bound = min(ub - tol, best_bound) if np.isfinite(best_bound) else ub - tol
    proven = bool(exhaustive and info["survivor_supports"][-1] == 0)
    info.update({
        "exhaustive": exhaustive,
        "certified_bound": (bound if exhaustive else float("-inf")),
        "n_surviving_supports": info["survivor_supports"][-1],
        "final_pool": int(pool.shape[0]),
        "exact_rank_limit": exact_rank_limit(),
    })
    return proven, info


def _pad_to_support(keep_idx, K, live, n, Cnp):
    """Complete surviving live-subsets `T` into full size-`K` supports for refinement.

    The fillers come from the dead assets with the lowest pooled cost -- `T` alone is a legitimate
    smaller support, but the supports it *stands for* are the padded ones, and those are what the
    per-support primal should try to beat the incumbent with.
    """
    if not keep_idx:
        return np.zeros((0, K), np.int64)
    dead = np.setdiff1d(np.arange(n), live)
    order = dead[np.argsort(Cnp[dead].min(axis=1))] if dead.size else dead
    out = []
    for arr in keep_idx:
        t = arr.shape[1]
        need = K - t
        if need == 0:
            out.append(arr)
        elif order.size >= need:
            pad = np.broadcast_to(order[:need], (arr.shape[0], need))
            out.append(np.concatenate([arr, pad], axis=1))
    return np.concatenate(out) if out else np.zeros((0, K), np.int64)
