"""A global-optimality certifier for cardinality-constrained index tracking on a lot grid, by enumeration.

Relaxations of this problem sit far below the optimum (a larger cardinality tracks an index much better at the same
weight), so the certificate accounts for every support instead. For a fixed support ``S``, every lattice point
``w = m/L`` with ``lot_lo <= m_i <= lot_hi`` lies in ``{1'w = 1, lot_lo/L <= w <= lot_hi/L}``, so the continuous
minimum over ``S`` bounds the best lattice point on ``S``. If that bound is ``>= U`` for every ``S``, ``U`` is globally
optimal.

Two stages, cheapest first: a closed-form equality-only bound (one small batched solve per support), then the
capped-simplex QP on what it could not prune. Survivors are reported (``proven=False`` with the supports), never
assumed away.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._capped_simplex import capped_simplex_qp


def _pascal(n: int, k: int) -> np.ndarray:
    """`C[i, j] = comb(i, j)` as exact Python ints widened to float64 for the rank arithmetic."""
    c = np.zeros((n + 1, k + 2), dtype=object)
    for i in range(n + 1):
        c[i, 0] = 1
        for j in range(1, min(i, k + 1) + 1):
            c[i, j] = c[i - 1, j - 1] + (c[i - 1, j] if j <= i - 1 else 0)
    return c


@partial(jax.jit, static_argnames=("n", "K"))
def _unrank(ranks, prefix, n: int, K: int):
    """Map colex ranks to the K-subsets of `range(n)`, fully vectorized.

    `prefix[j, x] = sum_{c <= x} C(n-1-c, K-j-1)` counts the subsets at level `j` whose next chosen
    element is at most `x`. The next element is then the first `x >= s` with
    `prefix[j, x] - prefix[j, s-1] > rem`, i.e. one `searchsorted` per level -- so the whole
    enumeration is `K` searches over an `n`-vector rather than any per-subset control flow.
    """
    B = ranks.shape[0]
    out = jnp.zeros((B, K), dtype=jnp.int32)
    rem = ranks
    start = jnp.zeros(B, dtype=jnp.int32)
    for j in range(K):
        p = prefix[j]                                   # (n,)
        base = jnp.where(start > 0, p[jnp.maximum(start - 1, 0)], 0.0)
        x = jnp.searchsorted(p, base + rem, side="right").astype(jnp.int32)
        x = jnp.clip(x, start, n - 1)
        prev = jnp.where(x > 0, p[jnp.maximum(x - 1, 0)], 0.0)
        rem = rem - (prev - base)
        out = out.at[:, j].set(x)
        start = x + 1
    return out


@partial(jax.jit, static_argnames=("K",))
def _equality_bound(S, mu, idx, K: int):
    """Stage-1 lower bound per support: the `sum w = 1` minimum, box dropped. Closed form."""
    A = S[idx[:, :, None], idx[:, None, :]]             # (B, K, K)
    b = mu[idx]                                         # (B, K)
    one = jnp.ones((idx.shape[0], K), dtype=A.dtype)
    rhs = jnp.stack([b, one], axis=-1)                  # (B, K, 2)
    sol = jnp.linalg.solve(A, rhs)
    u, v = sol[..., 0], sol[..., 1]
    su, sv = u.sum(-1), v.sum(-1)
    t = (su - 1.0) / jnp.where(jnp.abs(sv) < 1e-300, 1e-300, sv)
    return t * t * sv - (u * b).sum(-1)


@partial(jax.jit, static_argnames=("K", "iters"))
def _capped_bound(S, mu, idx, lo, hi, K: int, iters: int):
    """Stage-2 lower bound: the true continuous relaxation of the lattice problem on this support."""
    A = S[idx[:, :, None], idx[:, None, :]]
    b = mu[idx]
    w = capped_simplex_qp(2.0 * A, 2.0 * b, lo, hi, max_iters=iters)
    return jnp.einsum("bi,bij,bj->b", w, A, w) - 2.0 * (b * w).sum(-1)


def _bounded_compositions(K: int, L: int, lo: int, hi: int, cap: int) -> np.ndarray | None:
    """Every integer `m` with `sum(m) = L` and `lo <= m_i <= hi`. `None` if there are more than `cap`.

    Built level by level, keeping only prefixes that can still be completed -- the surviving set at
    the last level is exactly the feasible lattice, so this enumerates without ever materialising an
    infeasible product space.
    """
    part = np.zeros((1, 0), dtype=np.int32)
    tot = np.zeros(1, dtype=np.int64)
    for j in range(K):
        rest = K - j - 1
        vals = np.arange(lo, hi + 1, dtype=np.int32)
        newt = tot[:, None] + vals[None, :]
        # a prefix survives only if the remaining coordinates can still reach exactly L
        ok = (newt + rest * lo <= L) & (newt + rest * hi >= L)
        rows, cols = np.nonzero(ok)
        if rows.size > cap:
            return None
        part = np.concatenate([part[rows], vals[cols][:, None]], axis=1)
        tot = newt[rows, cols]
    return part[tot == L]


def _lattice_min(A: np.ndarray, b: np.ndarray, comps: np.ndarray, L: int) -> float:
    """Exact minimum of `w'Aw - 2b'w` over the lattice points `comps/L`. No relaxation involved."""
    w = comps.astype(np.float64) / L
    return float((np.einsum("bi,ij,bj->b", w, A, w) - 2.0 * (w @ b)).min())


def certify(
    cov: np.ndarray,
    mean: np.ndarray,
    *,
    K: int,
    L: int,
    eps: float,
    delta: float,
    const: float,
    incumbent: float,
    tol: float = 1e-12,
    chunk: int = 1 << 20,
    qp_iters: int = 400,
    max_report: int = 64,
    lattice_cap: int = 1 << 24,
):
    """Prove (or fail to prove) that `incumbent` is the global optimum on an `L`-lot grid.

    Takes the problem in plain arrays -- `cov` (`n x n`), `mean` (`n`), the per-name lot fraction
    bounds `eps <= w_i <= delta` (as fractions of the book, so lot counts are
    `lot_lo = ceil(eps * L)`, `lot_hi = floor(delta * L)`), and `const`, the additive constant such
    that `objective(w) = w'cov*w - 2*mean'w + const` matches whatever convention the caller's own
    objective uses. There is no dependency here on any particular solver's spec object: any prover
    that can hand over these six numbers and a claimed value can be checked by this function.

    `incumbent` is the objective value INCLUDING `const`, i.e. exactly what the caller's own
    objective function returns.

    Returns `(proven, info)`. `proven=True` is a genuine certificate of global optimality. On
    `False`, `info["survivors"]` (actually `info["refuted"]`/`info["unresolved"]`) holds supports
    that could not be excluded -- they are not counterexamples, only unresolved.
    """
    n = cov.shape[0]
    lot_lo = max(1, int(math.ceil(eps * L)))
    lot_hi = min(L, int(math.floor(delta * L)))
    if K * lot_lo > L or K * lot_hi < L:
        raise ValueError(f"no feasible lattice point: K={K}, lot_lo={lot_lo}, lot_hi={lot_hi}, L={L}")

    total = math.comb(n, K)
    # Ranks are enumerated as floats. Without 64-bit JAX `jnp.float64` silently becomes float32, which holds integers
    # exactly only up to 2^24: above that, consecutive ranks collide and supports are SKIPPED (the top 4096 ranks of
    # C(50, 10) unrank to 5 distinct supports), so an "exhaustive" proof would not be. Refuse rather than mislead.
    rank_dtype = jnp.arange(0, 1, dtype=jnp.float64).dtype
    exact = 2 ** (jnp.finfo(rank_dtype).nmant + 1)
    if total > exact:
        raise ValueError(f"C({n}, {K}) = {total:,} supports exceeds the {exact:,} ranks {rank_dtype} represents "
                         "exactly; enable 64-bit JAX (JAX_ENABLE_X64=1, or jax.config.update('jax_enable_x64', True) "
                         "before any JAX work), otherwise ranks collide and supports are silently skipped")
    c = _pascal(n, K)
    prefix = np.zeros((K, n), dtype=np.float64)
    for j in range(K):
        run = 0
        for x in range(n):
            run += int(c[n - 1 - x, K - j - 1]) if (n - 1 - x) >= 0 else 0
            prefix[j, x] = run
    prefix = jnp.asarray(prefix)

    # The bound is compared against the SHIFTED objective (`const` dropped), because that is what
    # the quadratic form returns; shift the incumbent to match rather than the other way round.
    target = float(incumbent) - const
    S = jnp.asarray(cov)
    mu = jnp.asarray(mean)
    lo, hi = lot_lo / L, lot_hi / L

    survivors, n_stage2 = [], 0
    for lo_r in range(0, total, chunk):
        hi_r = min(lo_r + chunk, total)
        ranks = jnp.arange(lo_r, hi_r, dtype=jnp.float64)
        idx = _unrank(ranks, prefix, n, K)
        eb = _equality_bound(S, mu, idx, K)
        keep = jnp.nonzero(eb < target - tol)[0]
        if keep.size == 0:
            continue
        sub = idx[keep]
        n_stage2 += int(sub.shape[0])
        cb = _capped_bound(S, mu, sub, lo, hi, K, qp_iters)
        still = jnp.nonzero(cb < target - tol)[0]
        if still.size:
            survivors.append(np.asarray(sub[still]))

    surv = np.concatenate(survivors) if survivors else np.zeros((0, K), dtype=np.int64)

    # Stage 3: a support's CONTINUOUS relaxation can legitimately sit below its best LATTICE point,
    # so surviving stage 2 is not evidence of anything -- it just means the relaxation was too weak
    # to settle that support. Resolve those exactly, on the integer lattice, with no relaxation.
    # Two failure modes that must NEVER be conflated:
    #   REFUTED    -- a strictly better lattice solution was exhibited. The incumbent is not optimal.
    #   UNRESOLVED -- the support could not be settled within `lattice_cap`. Says nothing either way.
    # Reporting the second as the first would turn "we ran out of budget" into "your answer is wrong".
    n_stage3, refuted, unresolved, best_other = int(surv.shape[0]), [], [], float("inf")
    if n_stage3:
        comps = _bounded_compositions(K, L, lot_lo, lot_hi, cap=lattice_cap)
        Snp, munp = np.asarray(cov), np.asarray(mean)
        for s in surv:
            if comps is None:
                unresolved.append(s)
                continue
            v = _lattice_min(Snp[np.ix_(s, s)], munp[s], comps, L)
            best_other = min(best_other, v)
            if v < target - tol:
                refuted.append(s)
    ref = np.asarray(refuted) if refuted else np.zeros((0, K), dtype=np.int64)
    un = np.asarray(unresolved) if unresolved else np.zeros((0, K), dtype=np.int64)
    return (ref.shape[0] == 0 and un.shape[0] == 0), {
        "supports_enumerated": total,
        "reached_stage2": n_stage2,
        "reached_stage3": n_stage3,
        "n_refuted": int(ref.shape[0]),
        "n_unresolved": int(un.shape[0]),
        "refuted": ref[:max_report],
        "unresolved": un[:max_report],
        "lattice_points": (0 if n_stage3 == 0 else
                           (-1 if comps is None else int(comps.shape[0]))),
        "best_lattice_off_incumbent": (best_other + const
                                       if np.isfinite(best_other) else float("nan")),
    }
