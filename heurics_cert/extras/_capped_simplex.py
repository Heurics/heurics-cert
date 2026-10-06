"""Capped-simplex projection and QP (JAX), the primitives the extras' bounds are built on."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def project_capped_simplex(
    v: Float[Array, "... k"],
    eps: Float[Array, "..."] | float,
    delta: Float[Array, "..."] | float,
) -> Float[Array, "... k"]:
    """Euclidean projection of `v` onto `{x : sum(x) = 1, eps <= x <= delta}`.

    Solved in closed form: `g(tau) = sum(clip(v - tau, eps, delta))` is piecewise linear and
    non-increasing in `tau`, with breakpoints at `v_i - delta_i` and `v_i - eps_i`, so evaluating it
    at the `2k` sorted breakpoints brackets the root `g(tau) = 1` inside one linear segment and a
    single interpolation lands on it.

    Batches over any leading shape by ordinary broadcasting. `eps`/`delta` are scalars or arrays
    broadcastable against `v`; `eps_i == delta_i` pins coordinate `i`. Requires
    `sum(eps) <= 1 <= sum(delta)` for the feasible set to be non-empty.
    """
    lo = jnp.broadcast_to(jnp.asarray(eps, dtype=v.dtype), v.shape)
    hi = jnp.broadcast_to(jnp.asarray(delta, dtype=v.dtype), v.shape)

    breaks = jnp.sort(jnp.concatenate([v - hi, v - lo], axis=-1), axis=-1)  # (..., 2k) ascending
    g = jnp.sum(
        jnp.clip(v[..., None, :] - breaks[..., :, None], lo[..., None, :], hi[..., None, :]),
        axis=-1,
    )  # (..., 2k), non-increasing along the last axis

    n_break = breaks.shape[-1]
    j = jnp.clip(jnp.sum(g >= 1.0, axis=-1) - 1, 0, n_break - 2)[..., None]
    b_lo = jnp.take_along_axis(breaks, j, axis=-1)
    b_hi = jnp.take_along_axis(breaks, j + 1, axis=-1)
    g_lo = jnp.take_along_axis(g, j, axis=-1)
    g_hi = jnp.take_along_axis(g, j + 1, axis=-1)

    denom = g_lo - g_hi
    tau = jnp.where(denom > 0, b_lo + (g_lo - 1.0) * (b_hi - b_lo) / jnp.where(denom > 0, denom, 1.0),
                    b_lo)
    return jnp.clip(v - tau, lo, hi)


def capped_simplex_qp(
    G: Float[Array, "... k k"],
    a: Float[Array, "... k"],
    eps: Float[Array, "..."] | float,
    delta: Float[Array, "..."] | float,
    *,
    max_iters: int = 200,
) -> Float[Array, "... k"]:
    """Minimize `0.5 x'Gx - a'x` subject to `sum(x) = 1`, `eps <= x <= delta`.

    `G` must be symmetric PSD. Any leading batch shape broadcasts. Solved by FISTA
    (Nesterov-accelerated projected gradient); `max_iters` is a fixed budget, not a convergence
    check -- verify against a reference solver before relying on exactness.
    """
    k = G.shape[-1]
    L = jnp.max(jnp.sum(jnp.abs(G), axis=-1), axis=-1, keepdims=True)
    L = jnp.maximum(L, 1e-12)
    step = 1.0 / L

    x0 = jnp.broadcast_to(jnp.asarray(1.0 / k, dtype=a.dtype), a.shape)
    x0 = project_capped_simplex(x0, eps, delta)
    t0 = jnp.ones(L.shape[:-1], dtype=a.dtype)

    def body(_, state):
        x, y, t = state
        grad = jnp.einsum("...ij,...j->...i", G, y) - a
        x_new = project_capped_simplex(y - step * grad, eps, delta)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        momentum = ((t - 1.0) / t_new)[..., None]
        y_new = x_new + momentum * (x_new - x)
        return x_new, y_new, t_new

    x, _, _ = jax.lax.fori_loop(0, max_iters, body, (x0, x0, t0))
    return x
