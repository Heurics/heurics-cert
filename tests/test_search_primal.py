"""ccqp.primal: the restricted QP respects buy-in bounds, is exact, and is never worse than the old drop-and-repair
route; polish returns feasible portfolios."""
import unittest

import numpy as np
from scipy.optimize import minimize

from heurics_cert.models import Problem
from heurics_cert.search import primal


def _instance(seed, n=40, K=30, lo=0.03, hi=0.1, factors=3):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, factors)) * 0.1
    Q = F @ F.T + np.diag(rng.uniform(0.002, 0.02, n))
    mu = rng.uniform(0.0, 0.1, n)
    lo_v, hi_v = np.full(n, lo), np.full(n, hi)
    rho = float(np.quantile(mu, 0.5))
    return Problem(Q=0.5 * (Q + Q.T), mu=mu, lo=lo_v, hi=hi_v, K=K, rho=rho)


def _reference(P, S):
    """Independent reference for the support QP: SLSQP from several starts."""
    S = np.asarray(sorted(S))
    QS = P.Q[np.ix_(S, S)]
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0},
            {"type": "ineq", "fun": lambda x: P.mu[S] @ x - P.rho}]
    best = np.inf
    rng = np.random.default_rng(0)
    for _ in range(5):
        x0 = rng.uniform(P.lo[S], P.hi[S])
        x0 /= x0.sum()
        r = minimize(lambda x: x @ QS @ x, x0, jac=lambda x: 2 * QS @ x, method="SLSQP",
                     bounds=list(zip(P.lo[S], P.hi[S])), constraints=cons, options={"ftol": 1e-14, "maxiter": 500})
        if r.success and abs(r.x.sum() - 1) < 1e-8 and P.mu[S] @ r.x >= P.rho - 1e-9:
            best = min(best, float(r.x @ QS @ r.x))
    return best


class RestrictedQP(unittest.TestCase):
    def test_exact_on_small_supports(self):
        for seed in range(6):
            P = _instance(seed, n=14, K=10, lo=0.05, hi=0.3)
            S = set(np.argsort(-P.mu)[:10].tolist())
            x, f = primal._restricted_lo(P, S)
            ref = _reference(P, S)
            if not np.isfinite(ref):
                self.assertIsNone(x)
                continue
            self.assertIsNotNone(x)
            self.assertTrue(P.feasibility(x)["feasible"])
            self.assertLessEqual(f, ref * (1 + 1e-7), f"seed {seed}: {f} vs reference {ref}")
            self.assertTrue(np.all(x[sorted(S)] >= P.lo[sorted(S)] * (1 - 1e-12)), "buy-in violated")

    def test_never_worse_than_drop_route(self):
        for seed in range(8):
            P = _instance(seed)
            S = set(np.argsort(np.diag(P.Q))[:30].tolist())
            x, f = primal.restricted_qp(P, S)
            xd, fd = primal._restricted_drop(P, S)
            if xd is not None:
                self.assertIsNotNone(x)
                self.assertLessEqual(f, fd * (1 + 1e-12))
            if x is not None:
                self.assertTrue(P.feasibility(x)["feasible"])

    def test_buy_in_dense_portfolio_is_not_dismantled(self):
        """The regression: with most holdings pinned at lo, relaxing lo and dropping dust discards the optimum's
        assets. The buy-in-respecting QP must keep the whole support and beat that route."""
        wins = 0
        for seed in range(8):
            P = _instance(seed)
            S = set(np.argsort(np.diag(P.Q))[:30].tolist())
            x, f = primal._restricted_lo(P, S)
            xd, fd = primal._restricted_drop(P, S)
            if x is None:
                continue
            self.assertEqual(int((x > 0).sum()), 30, "a buy-in asset was dropped")
            wins += (xd is None) or f < fd * (1 - 1e-9)
        self.assertGreater(wins, 0, "no instance where holding the full support beats dropping dust")


class Polish(unittest.TestCase):
    def test_feasible_and_improving(self):
        for seed in range(4):
            P = _instance(seed, n=60, K=20, lo=0.02, hi=0.15)
            x0 = np.full(P.n, 1.0 / P.n)
            x, f, info = primal.polish(P, x0, time_limit=5.0)
            self.assertIsNotNone(x)
            self.assertTrue(P.feasibility(x)["feasible"])
            self.assertLessEqual(int((x > 0).sum()), P.K)
            self.assertAlmostEqual(f, P.objective(x), places=15)


if __name__ == "__main__":
    unittest.main()
