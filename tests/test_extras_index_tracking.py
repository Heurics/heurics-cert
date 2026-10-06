"""Tests for [`heurics_cert.extras.index_tracking`][] -- the enumeration certifier for lot-constrained
index tracking.

The property under test is validity and completeness of the proof, not
speed: brute force every support and every lattice point at a size small enough to enumerate
directly, and check that `certify` (a) proves a genuine global optimum and (b) refuses to prove a
value that a brute-force search can beat.
"""
import itertools
import os
import unittest

import numpy as np

from heurics_cert.extras.index_tracking import certify


def _instance(rng, n):
    F = rng.normal(size=(n, 3))
    cov = F @ F.T + np.eye(n) * 0.5
    mean = rng.normal(scale=0.1, size=n)
    return cov, mean


def _lattice_points(K, L, lot_lo, lot_hi):
    for comp in itertools.product(range(lot_lo, lot_hi + 1), repeat=K):
        if sum(comp) == L:
            yield comp


def _brute_force_optimum(cov, mean, K, L, lot_lo, lot_hi):
    n = cov.shape[0]
    best = float("inf")
    for support in itertools.combinations(range(n), K):
        s = np.asarray(support)
        A = cov[np.ix_(s, s)]
        b = mean[s]
        for comp in _lattice_points(K, L, lot_lo, lot_hi):
            w = np.asarray(comp, dtype=np.float64) / L
            val = float(w @ A @ w - 2.0 * b @ w)
            best = min(best, val)
    return best


class TestCertify(unittest.TestCase):
    def test_proves_the_true_global_optimum(self):
        rng = np.random.default_rng(0)
        n, K, L = 6, 2, 6
        cov, mean = _instance(rng, n)
        opt = _brute_force_optimum(cov, mean, K, L, 1, L)

        proven, info = certify(cov, mean, K=K, L=L, eps=0.0, delta=1.0, const=0.0,
                                incumbent=opt, tol=1e-9)
        self.assertTrue(proven, f"failed to prove the true optimum; info={info}")
        self.assertEqual(info["n_refuted"], 0)
        self.assertEqual(info["n_unresolved"], 0)

    def test_refuses_to_prove_a_beatable_incumbent(self):
        """The falsification test with teeth: handed a value strictly worse than the true optimum,
        the certifier must find a counterexample support rather than rubber-stamp it."""
        rng = np.random.default_rng(1)
        n, K, L = 6, 2, 6
        cov, mean = _instance(rng, n)
        opt = _brute_force_optimum(cov, mean, K, L, 1, L)
        bad = opt + 0.5 * abs(opt) + 1e-3

        proven, info = certify(cov, mean, K=K, L=L, eps=0.0, delta=1.0, const=0.0,
                                incumbent=bad, tol=1e-9)
        self.assertFalse(proven, "proved a beatable incumbent")
        self.assertGreater(info["n_refuted"], 0)

    def test_const_shifts_the_reported_objective_consistently(self):
        """`const` must be a pure additive shift: proving `incumbent` at `const=c` must agree with
        proving `incumbent + c` at `const=0`, since both describe the same underlying problem."""
        rng = np.random.default_rng(2)
        n, K, L = 6, 2, 6
        cov, mean = _instance(rng, n)
        opt = _brute_force_optimum(cov, mean, K, L, 1, L)
        shift = 3.7

        p0, i0 = certify(cov, mean, K=K, L=L, eps=0.0, delta=1.0, const=0.0,
                          incumbent=opt, tol=1e-9)
        p1, i1 = certify(cov, mean, K=K, L=L, eps=0.0, delta=1.0, const=shift,
                          incumbent=opt + shift, tol=1e-9)
        self.assertEqual(p0, p1)
        self.assertEqual(i0["n_refuted"], i1["n_refuted"])
        self.assertEqual(i0["n_unresolved"], i1["n_unresolved"])


class TestRankPrecision(unittest.TestCase):
    """Supports are enumerated by float rank. In 32-bit JAX that is float32, exact only up to 2^24 supports."""

    def test_refuses_more_supports_than_ranks_it_can_represent(self):
        import jax
        if jax.config.jax_enable_x64:
            self.skipTest("64-bit JAX: every rank in range is exact")
        rng = np.random.default_rng(3)
        cov, mean = _instance(rng, 40)                 # C(40, 8) = 76.9M > 2^24
        with self.assertRaisesRegex(ValueError, "enable 64-bit JAX"):
            certify(cov, mean, K=8, L=40, eps=0.0, delta=1.0, const=0.0, incumbent=0.0, tol=1e-9)

    def test_64bit_ranks_reach_every_support_near_the_top_of_a_large_space(self):
        """The top 4096 ranks of C(50, 10) are 4096 distinct supports in 64-bit JAX (5 in float32)."""
        import subprocess
        import sys
        code = (
            "import math, numpy as np, jax.numpy as jnp\n"
            "from heurics_cert.extras import index_tracking as itb\n"
            "n, K = 50, 10\n"
            "c = itb._pascal(n, K)\n"
            "prefix = np.zeros((K, n))\n"
            "for j in range(K):\n"
            "    run = 0\n"
            "    for x in range(n):\n"
            "        run += int(c[n - 1 - x, K - j - 1])\n"
            "        prefix[j, x] = run\n"
            "total = math.comb(n, K)\n"
            "ranks = jnp.arange(total - 4096, total, dtype=jnp.float64)\n"
            "idx = np.asarray(itb._unrank(ranks, jnp.asarray(prefix), n, K))\n"
            "print(len({tuple(r) for r in idx}))\n")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu",
                   PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "4096")


if __name__ == "__main__":
    unittest.main()
