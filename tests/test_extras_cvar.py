"""Tests for [`heurics_cert.extras.cvar`][] -- the GPU-native mean-CVaR certificate.

As in `test_certificate.py`, the property under test is **validity**, not tightness: a loose bound
is a weak certificate, a bound above the true optimum is a false claim of optimality. So the
load-bearing tests enumerate every support by brute force, solve each restricted problem to
optimality with an independent arbiter, and assert the bound never crosses the answer -- including
at *deliberately random* duals, since weak duality is supposed to hold at every `q in Q` and that
is the entire premise of certifying without converging.

`scipy.optimize.linprog` appears only as a **test-time** arbiter, and it is the right one here: with
the support fixed, mean-CVaR's Rockafellar-Uryasev form is an honest LP, so `linprog` returns the
exact value rather than an approximation of it. It is not, and must not become, part of the bound
path -- which `test_no_classical_solver_in_bound_path` asserts directly.

**The arbiter uses `1/m`, not `1/((1-beta)S)`, and that is not a detail.** The engine optimizes the
*discrete* tail mean: the average of the worst `m = ceil((1-beta)S)` losses. The textbook RU
coefficient `1/((1-beta)S)` equals `1/m` only when `(1-beta)S` is exactly integral, and in binary
floating point `1 - 0.95` is `0.05000000000000004`, which is precisely the off-by-one that
`cvar/instance.py` records having shipped once. Were the arbiter to use the textbook coefficient it
would be scoring a slightly different objective, and every "bound exceeds the optimum" failure here
would be that mismatch rather than a real defect -- or, worse, would hide one.

Everything runs in float32, the dtype production callers use, and this module does **not** enable
`jax_enable_x64`. Returns are scaled so objectives sit near 1e-2 rather than 1e-4: at 1e-4 a
float32 relative tolerance and a genuine sign error are the same size, and the test would pass on a
broken bound.
"""
import itertools
import unittest

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from heurics_cert.extras.cvar import (
    cvar_cardinality_free_bound,
    certify_cvar,
    cvar_convex_ceiling,
    cvar_pool_costs,
    cvar_saddle_dual,
    cvar_support_ascent,
    cvar_support_bounds,
    cvar_support_primal,
    project_tail_simplex,
    tail_dual,
)

_REL_TOL = 1e-5


def _slack(v):
    return _REL_TOL * max(abs(v), 1.0)


def _instance(rng, n, S, *, hi=1.0):
    """A small scenario panel with a factor structure and fat-ish tails, so the tail set of a
    portfolio is not simply the tail set of one dominant asset."""
    load = rng.normal(0.0, 1.0, size=(n, 3))
    f = rng.normal(0.0, 0.05, size=(S, 3))
    R = f @ load.T + rng.standard_t(6.0, size=(S, n)) * 0.03 + rng.normal(0.0, 0.01, size=n)
    return R, R.mean(axis=0), np.full(n, hi)


def _support_value(R, mu, lam, hi, m, support):
    """Exact `min CVaR_beta(-R_s w) - lam mu_s'w` over a fixed support, by LP.

    Columns are `(w, a, u)` and the rows are Rockafellar-Uryasev's epigraph `u_s >= -r_s'w - a`,
    with the tail coefficient `1/m` so the LP scores the same discrete tail mean the certificate
    bounds. Returns `None` where the support cannot carry the budget inside its box.
    """
    s = np.asarray(support)
    k, S = len(s), R.shape[0]
    if hi[s].sum() < 1.0 - 1e-12:
        return None
    Rs = R[:, s]
    c = np.concatenate([-lam * mu[s], [1.0], np.full(S, 1.0 / m)])
    # -u_s - r_s'w - a <= 0
    A_ub = np.hstack([-Rs, -np.ones((S, 1)), -np.eye(S)])
    b_ub = np.zeros(S)
    A_eq = np.zeros((1, k + 1 + S))
    A_eq[0, :k] = 1.0
    bounds = [(0.0, float(hi[i])) for i in s] + [(None, None)] + [(0.0, None)] * S
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    return float(r.fun) if r.success else None


def _brute_force(R, mu, lam, hi, m, K):
    """`(optimum, {support: value})` over every support of size 1..K."""
    n = R.shape[1]
    vals = {}
    best = float("inf")
    for size in range(1, K + 1):
        for s in itertools.combinations(range(n), size):
            v = _support_value(R, mu, lam, hi, m, s)
            if v is not None:
                vals[s] = v
                best = min(best, v)
    return best, vals


def _f32(*arrays):
    return tuple(jnp.asarray(a, jnp.float32) for a in arrays)


class TestTailSimplexProjection(unittest.TestCase):
    """`Q = {0 <= q <= 1/m, sum q = 1}` is the risk envelope. Everything else is built on it."""

    def test_is_feasible_and_optimal(self):
        rng = np.random.default_rng(0)
        for S, m in ((20, 5), (200, 10), (200, 200), (16, 1)):
            v = rng.normal(size=S) * 2.0
            q = np.asarray(project_tail_simplex(jnp.asarray(v, jnp.float32), 1.0 / m), np.float64)
            self.assertAlmostEqual(q.sum(), 1.0, places=4, msg=f"S={S} m={m}")
            self.assertGreaterEqual(q.min(), -1e-6)
            self.assertLessEqual(q.max(), 1.0 / m + 1e-6)
            # Optimality by the projection's own variational inequality: for the Euclidean
            # projection, `(v - q)` must be constant on the strictly interior coordinates -- the
            # single multiplier `tau` of the budget row.
            interior = (q > 1e-6) & (q < 1.0 / m - 1e-6)
            if interior.sum() > 1:
                spread = (v - q)[interior]
                self.assertLess(spread.max() - spread.min(), 1e-4, f"S={S} m={m}")

    def test_matches_a_dense_reference_solver(self):
        rng = np.random.default_rng(1)
        S, m = 60, 7
        v = rng.normal(size=S)
        q = np.asarray(project_tail_simplex(jnp.asarray(v, jnp.float32), 1.0 / m), np.float64)
        # The projection is a QP; as an LP-checkable proxy, its value must beat a grid of feasible
        # points drawn at random.
        mine = float(np.sum((q - v) ** 2))
        for _ in range(200):
            z = rng.random(S)
            z = np.clip(z, 0, None)
            z = z / z.sum()
            z = np.minimum(z, 1.0 / m)
            z = z / z.sum()
            if z.max() <= 1.0 / m + 1e-12:
                self.assertLessEqual(mine, float(np.sum((z - v) ** 2)) + 1e-5)

    def test_pinned_uniform_case(self):
        """`m == S` pins every coordinate at `1/S`: the feasible set is a single point, and the
        interpolation denominator is zero there. It must return that point, not a NaN."""
        q = np.asarray(project_tail_simplex(jnp.arange(9, dtype=jnp.float32), 1.0 / 9))
        np.testing.assert_allclose(q, np.full(9, 1.0 / 9), atol=1e-6)


class TestTailDual(unittest.TestCase):
    def test_reproduces_the_discrete_tail_mean(self):
        """`max_q q'L` over `Q` IS the mean of the worst `m` entries of `L` -- the definition the
        engine optimizes. If this drifts, every bound in the module bounds a different problem."""
        rng = np.random.default_rng(2)
        for S, m in ((200, 10), (50, 3), (32, 32)):
            L = rng.normal(size=S).astype(np.float32)
            q = np.asarray(tail_dual(jnp.asarray(L), m), np.float64)
            self.assertAlmostEqual(q.sum(), 1.0, places=5)
            self.assertAlmostEqual(float(q @ L), float(np.sort(L)[-m:].mean()), places=5)


class TestSupportBoundValidity(unittest.TestCase):
    """The per-support dual bound must never exceed the support's true value, at any dual."""

    def test_never_exceeds_the_lp_value_at_random_duals(self):
        rng = np.random.default_rng(3)
        n, S, m, K, lam = 10, 200, 10, 3, 1.0
        R, mu, hi = _instance(rng, n, S)
        _opt, vals = _brute_force(R, mu, lam, hi, m, K)

        # A deliberately arbitrary pool: random points of `Q`, plus the tail duals of random books.
        pool = []
        for _ in range(12):
            pool.append(np.asarray(project_tail_simplex(
                jnp.asarray(rng.normal(size=S), jnp.float32), 1.0 / m)))
        for _ in range(6):
            w = rng.random(n)
            w /= w.sum()
            pool.append(np.asarray(tail_dual(jnp.asarray(-(R @ w), jnp.float32), m)))
        Q = jnp.asarray(np.stack(pool), jnp.float32)
        C = cvar_pool_costs(*_f32(R, mu), lam, Q)

        checked = 0
        for support, true_val in vals.items():
            if len(support) != K:
                continue
            b = float(cvar_support_bounds(C, jnp.asarray([support], jnp.int32))[0])
            self.assertLessEqual(b, true_val + _slack(true_val),
                                 f"pooled bound {b:.8g} exceeds val({support}) = {true_val:.8g}")
            checked += 1
        self.assertGreater(checked, 50, "test did not actually exercise many supports")

    def test_ascent_converges_to_the_support_value_and_never_passes_it(self):
        """Strong duality on a fixed support: the sup over `q` IS `val(s)`, so the ascent should
        approach it from below and stop there. This is the property that makes the certificate
        possible at all -- the same ascent is provably vacuous when the support is free."""
        rng = np.random.default_rng(4)
        n, S, m, K, lam = 10, 200, 10, 3, 1.0
        R, mu, hi = _instance(rng, n, S)
        _opt, vals = _brute_force(R, mu, lam, hi, m, K)
        picks = [s for s in vals if len(s) == K][:8]

        Rj, muj = _f32(R, mu)
        sup = jnp.asarray(picks, jnp.int32)
        ub = jnp.asarray([vals[s] for s in picks], jnp.float32)
        q0 = jnp.broadcast_to(jnp.full((S,), 1.0 / S, jnp.float32), (len(picks), S))
        best, _q = cvar_support_ascent(Rj, muj, lam, sup, m, ub, q0, iters=2000)
        best = np.asarray(best, np.float64)
        for s, b in zip(picks, best):
            self.assertLessEqual(float(b), vals[s] + _slack(vals[s]),
                                 f"ascent passed val({s})")
        rel = np.array([(vals[s] - b) / max(abs(vals[s]), 1e-9) for s, b in zip(picks, best)])
        self.assertLess(float(np.median(rel)), 0.05,
                        f"ascent is not approaching val(s); median relative shortfall {rel}")

    def test_support_primal_is_an_upper_bound_on_the_lp_value(self):
        rng = np.random.default_rng(5)
        n, S, m, K, lam = 10, 200, 10, 3, 1.0
        R, mu, hi = _instance(rng, n, S)
        _opt, vals = _brute_force(R, mu, lam, hi, m, K)
        picks = [s for s in vals if len(s) == K][:8]
        _w, f, q_avg = cvar_support_primal(*_f32(R, mu), lam, jnp.asarray(hi, jnp.float32),
                                           jnp.asarray(picks, jnp.int32), m, iters=1500)
        for s, v in zip(picks, np.asarray(f, np.float64)):
            self.assertGreaterEqual(float(v), vals[s] - _slack(vals[s]),
                                    f"primal claims below the LP optimum on {s}")

        # The ergodic dual that falls out of the same run must be a feasible point of `Q` and must
        # bound the same support from BELOW -- one run, both ends of the certificate.
        q = np.asarray(q_avg, np.float64)
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-4)
        self.assertGreaterEqual(q.min(), -1e-6)
        self.assertLessEqual(q.max(), 1.0 / m + 1e-6)
        C = cvar_pool_costs(*_f32(R, mu), lam, q_avg)
        low = np.asarray(cvar_support_bounds(C, jnp.asarray(picks, jnp.int32)), np.float64)
        for s, b in zip(picks, low):
            self.assertLessEqual(float(b), vals[s] + _slack(vals[s]),
                                 f"ergodic dual over-claims on {s}")

    def test_ergodic_dual_beats_the_naive_tail_dual(self):
        """The measured reason the certifier closes at all. At a CVaR optimum the `m`-th and
        `(m+1)`-th losses are tied, so the uniform-`1/m` tail dual is not the saddle point and
        lands percent-level short; the averaged dual splits mass across the tie and does not."""
        rng = np.random.default_rng(20)
        n, S, m, K, lam = 10, 200, 10, 3, 1.0
        R, mu, hi = _instance(rng, n, S)
        _opt, vals = _brute_force(R, mu, lam, hi, m, K)
        picks = sorted(((v, s) for s, v in vals.items() if len(s) == K))[:6]
        sup = jnp.asarray([s for _v, s in picks], jnp.int32)
        w, _f, q_avg = cvar_support_primal(*_f32(R, mu), lam, jnp.asarray(hi, jnp.float32),
                                           sup, m, iters=2000)
        q_naive = cvar_saddle_dual(jnp.asarray(R, jnp.float32), sup, w, m)
        b_avg = np.asarray(cvar_support_bounds(cvar_pool_costs(*_f32(R, mu), lam, q_avg), sup))
        b_nai = np.asarray(cvar_support_bounds(cvar_pool_costs(*_f32(R, mu), lam, q_naive), sup))
        true = np.array([v for v, _s in picks])
        self.assertTrue(np.all(b_avg <= true + 1e-5), "ergodic dual over-claims")
        self.assertTrue(np.all(b_nai <= true + 1e-5), "naive tail dual over-claims")
        rel_avg = float(np.median((true - b_avg) / np.abs(true)))
        rel_nai = float(np.median((true - b_nai) / np.abs(true)))
        self.assertLess(rel_avg, rel_nai,
                        f"averaging did not help: {rel_avg:.4f} vs naive {rel_nai:.4f}")


class TestConvexCeiling(unittest.TestCase):
    """The obstruction: with `hi = 1` no `w`-convex bound can beat the cardinality-free optimum."""

    def test_never_exceeds_the_true_k_sparse_optimum(self):
        rng = np.random.default_rng(6)
        n, S, m, K, lam = 10, 200, 10, 3, 1.0
        R, mu, hi = _instance(rng, n, S)
        opt, _vals = _brute_force(R, mu, lam, hi, m, K)
        ceiling, _q, _i = cvar_convex_ceiling(*_f32(R, mu), lam, m, opt, iters=1500)
        self.assertLessEqual(float(ceiling), opt + _slack(opt))

    def test_is_capped_by_the_cardinality_free_optimum(self):
        """The theorem, as a test. `conv(K-sparse capped simplex) = capped simplex` when `hi = 1`,
        so the ceiling equals the FULL-universe optimum -- and is therefore blind to `K`. It must
        not exceed that number for ANY `K`, which is what makes it a ceiling rather than a bound."""
        rng = np.random.default_rng(7)
        n, S, m, lam = 10, 200, 10, 1.0
        R, mu, hi = _instance(rng, n, S)
        free = _support_value(R, mu, lam, hi, m, tuple(range(n)))
        self.assertIsNotNone(free)
        for K in (2, 3):
            opt, _ = _brute_force(R, mu, lam, hi, m, K)
            ceiling, _q, _i = cvar_convex_ceiling(*_f32(R, mu), lam, m, opt, iters=3000)
            self.assertLessEqual(float(ceiling), free + _slack(free),
                                 f"ceiling passed the cardinality-free optimum at K={K}")
        # And the obstruction itself: the cardinality-free optimum is genuinely below the K-sparse
        # one, so a bound capped there cannot certify the K-sparse answer tightly.
        opt3, _ = _brute_force(R, mu, lam, hi, m, 3)
        self.assertLess(free, opt3 - 1e-6,
                        "instance does not exhibit the obstruction; pick a harder one")


class TestCertifier(unittest.TestCase):
    """End to end: the enumeration, the elimination, and the claim of proof."""

    def _case(self, seed, n, K, S=200, m=10, lam=1.0):
        rng = np.random.default_rng(seed)
        R, mu, hi = _instance(rng, n, S)
        opt, vals = _brute_force(R, mu, lam, hi, m, K)
        return R, mu, hi, opt, vals

    def test_certified_bound_never_exceeds_the_true_optimum(self):
        for seed, n, K in ((10, 10, 2), (11, 12, 3)):
            R, mu, hi, opt, _vals = self._case(seed, n, K)
            # A deliberately BAD incumbent: 40% worse than optimal. The certifier must still
            # return a valid lower bound, and must not "prove" anything.
            proven, info = certify_cvar(*_f32(R, mu), 1.0, jnp.asarray(hi, jnp.float32), 10, K,
                                        opt + 0.4 * abs(opt) + 1e-3, pool_rounds=2,
                                        tol=1e-6)
            self.assertLessEqual(info["certified_bound"], opt + _slack(opt),
                                 f"certified bound {info['certified_bound']:.8g} exceeds the true "
                                 f"optimum {opt:.8g} (n={n}, K={K})")
            if proven:
                self.assertLessEqual(info["upper_bound"], opt + _slack(opt))

    def test_a_claimed_proof_matches_brute_force(self):
        for seed, n, K in ((12, 10, 2), (13, 10, 3)):
            R, mu, hi, opt, _vals = self._case(seed, n, K)
            tol = 1e-4 * max(abs(opt), 1.0)
            proven, info = certify_cvar(*_f32(R, mu), 1.0, jnp.asarray(hi, jnp.float32), 10, K,
                                        opt, pool_rounds=3, ascent_iters=1500, tol=tol)
            self.assertLessEqual(info["certified_bound"], opt + _slack(opt))
            if proven:
                self.assertAlmostEqual(info["upper_bound"], opt, delta=10.0 * tol,
                                       msg="claimed a proof at a value brute force disagrees with")

    def test_does_not_rubber_stamp_a_suboptimal_incumbent(self):
        """The falsification test with teeth. Handed an incumbent 40% WORSE than optimal, the
        certifier must refuse to prove it: the optimal support's bound is below that incumbent, so
        it cannot be eliminated and must be reported as surviving. A certifier that proves anything
        it is handed proves nothing -- and note the mirror-image trap, which is why this is the test
        that has teeth: "nothing beats `UB`" is *trivially* true for a `UB` **below** the optimum,
        so passing a deliberately low incumbent would not exercise the machinery at all."""
        for seed, n, K in ((14, 10, 3), (18, 12, 2)):
            R, mu, hi, opt, _vals = self._case(seed, n, K)
            bad = opt + 0.4 * abs(opt)
            proven, info = certify_cvar(*_f32(R, mu), 1.0, jnp.asarray(hi, jnp.float32), 10, K,
                                        bad, pool_rounds=2, tol=1e-6)
            # The certifier may legitimately IMPROVE the incumbent off a survivor's primal; it has
            # then proved something about a different, better `UB`, which is a success, not a
            # rubber stamp. What it must never do is prove the bad one.
            if info["upper_bound"] >= bad - _slack(bad):
                self.assertFalse(proven, f"proved a 40%-suboptimal incumbent (n={n}, K={K})")
                self.assertGreaterEqual(info["n_surviving_supports"], 1)
            self.assertLessEqual(info["certified_bound"], opt + _slack(opt))

    def test_proves_a_tiny_instance(self):
        """The proof path has to be exercised somewhere, or every other test here passes on a
        certifier that never proves anything. `C(10, 2) = 45` supports, a correct incumbent, and a
        tolerance loose enough for float32 to reach."""
        R, mu, hi, opt, _vals = self._case(19, 10, 2)
        tol = 1e-3 * max(abs(opt), 1e-6)
        proven, info = certify_cvar(*_f32(R, mu), 1.0, jnp.asarray(hi, jnp.float32), 10, 2,
                                    opt, pool_rounds=4, ascent_iters=2000, tol=tol)
        self.assertLessEqual(info["certified_bound"], opt + _slack(opt))
        self.assertTrue(proven, f"could not prove a 45-support instance; "
                                f"{info['n_surviving_supports']} survivors, "
                                f"bound {info['certified_bound']:.8g} vs optimum {opt:.8g}")

    def test_more_pool_rounds_never_loosen_the_bound(self):
        R, mu, hi, opt, _vals = self._case(15, 10, 3)
        args = (*_f32(R, mu), 1.0, jnp.asarray(hi, jnp.float32), 10, 3, opt)
        _p1, i1 = certify_cvar(*args, pool_rounds=1, ascent_iters=400, tol=1e-6)
        _p2, i2 = certify_cvar(*args, pool_rounds=3, ascent_iters=400, tol=1e-6)
        self.assertGreaterEqual(i2["certified_bound"], i1["certified_bound"] - _slack(opt))
        self.assertLessEqual(i2["n_surviving_supports"], i1["n_surviving_supports"])

    def test_bound_is_valid_with_hi_below_one(self):
        """`hi < 1` is the regime where the convex collapse does NOT hold (`ceil(1/hi) > K`), so
        the feasible set is genuinely different. The per-support bound has to stay valid there
        too -- it never used `hi = 1` for anything, but a wrong box in the pool costs would show up
        here and nowhere else."""
        rng = np.random.default_rng(16)
        n, S, m, K, lam = 10, 200, 10, 3, 1.0
        R, mu, hi = _instance(rng, n, S, hi=0.5)
        opt, _vals = _brute_force(R, mu, lam, hi, m, K)
        _proven, info = certify_cvar(*_f32(R, mu), lam, jnp.asarray(hi, jnp.float32), m, K,
                                     opt, pool_rounds=2, tol=1e-6)
        self.assertLessEqual(info["certified_bound"], opt + _slack(opt))


class TestNoClassicalSolver(unittest.TestCase):
    """The constraint that makes this a shippable certificate rather than a wrapper around someone's solver:
    the bound path loads nothing beyond the standard library, the array stack and this package."""

    SNIPPET = """
import numpy as np, jax.numpy as jnp
from heurics_cert.extras.cvar import certify_cvar, cvar_convex_ceiling
rng = np.random.default_rng(17); n, S = 10, 200
load = rng.normal(0.0, 1.0, size=(n, 3)); fac = rng.normal(0.0, 0.05, size=(S, 3))
R = fac @ load.T + rng.standard_t(6.0, size=(S, n)) * 0.03 + rng.normal(0.0, 0.01, size=n)
Rj, muj = jnp.asarray(R, jnp.float32), jnp.asarray(R.mean(axis=0), jnp.float32)
_, info = certify_cvar(Rj, muj, 1.0, jnp.ones(n, jnp.float32), 10, 3, 0.05, pool_rounds=2, tol=1e-6)
assert np.isfinite(info["certified_bound"]) or not info["exhaustive"]
ceiling, _q, _i = cvar_convex_ceiling(Rj, muj, 1.0, 10, 0.05, iters=100)
assert np.isfinite(float(ceiling))
"""

    def test_no_classical_solver_in_bound_path(self):
        from _solver_free import foreign_modules
        self.assertEqual(foreign_modules(self.SNIPPET), [],
                         "modules outside the allowlist were loaded while computing a CVaR certificate")


class CardinalityFreeBoundTests(unittest.TestCase):
    """The bound this campaign compares against an external solver's dual bound (IG18). It must be
    valid for every K, and its bracket must actually bracket."""

    def test_is_a_valid_lower_bound_for_every_k(self):
        """Dropping the cardinality row enlarges the feasible set, so one cardinality-free bound
        underestimates the K-sparse optimum at EVERY K."""
        rng = np.random.default_rng(11)
        R, mu, hi = _instance(rng, 8, 120)
        m, lam = 6, 1.0
        Rj, muj, hij = _f32(R, mu, hi)
        lower, upper, _q = cvar_cardinality_free_bound(Rj, muj, lam, hij, m, iters=4000)
        lower, upper = float(lower), float(upper)
        self.assertLessEqual(lower, upper + 1e-9, "the bracket must not be inverted")
        for K in (1, 2, 4):
            opt, _vals = _brute_force(R, mu, lam, hi, m, K)
            self.assertLessEqual(lower, opt + _slack(opt),
                                 f"bound {lower} exceeds the true K={K} optimum {opt}")

    def test_q_is_feasible_for_the_risk_envelope(self):
        """Validity rests entirely on `q` being a point of Q -- if it drifts outside, the number is
        not a bound at all, however tight it looks."""
        rng = np.random.default_rng(12)
        R, mu, hi = _instance(rng, 6, 90)
        m, lam = 5, 1.0
        Rj, muj, hij = _f32(R, mu, hi)
        _lower, _upper, q = cvar_cardinality_free_bound(Rj, muj, lam, hij, m, iters=2000)
        q = np.asarray(q, dtype=np.float64)
        self.assertAlmostEqual(float(q.sum()), 1.0, places=4)
        self.assertGreaterEqual(float(q.min()), -1e-6)
        self.assertLessEqual(float(q.max()), 1.0 / m + 1e-6)

    def test_more_iterations_tighten_the_bracket(self):
        """The bracket is the convergence statement, so it has to shrink with effort -- otherwise
        quoting it as evidence of convergence would be unfounded."""
        rng = np.random.default_rng(13)
        R, mu, hi = _instance(rng, 8, 120)
        m, lam = 6, 1.0
        Rj, muj, hij = _f32(R, mu, hi)
        short = cvar_cardinality_free_bound(Rj, muj, lam, hij, m, iters=500)
        long = cvar_cardinality_free_bound(Rj, muj, lam, hij, m, iters=8000)
        self.assertLess(float(long[1]) - float(long[0]),
                        float(short[1]) - float(short[0]) + 1e-9)


if __name__ == "__main__":
    unittest.main()
