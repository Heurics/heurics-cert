"""Certify a small cardinality-constrained portfolio problem end to end.

    python examples/quickstart.py

Builds a 40-stock problem from simulated returns, proves a lower bound, saves the certificate, checks it again from
the file alone, and reports the certified gap of a simple candidate portfolio.
"""
import os
import tempfile

import numpy as np

import heurics_cert as hc
from heurics_cert.models import factor_model, return_target
from heurics_cert.search import polish

rng = np.random.default_rng(0)
returns = 0.02 * (rng.standard_normal((104, 1)) + rng.standard_normal((104, 40)))     # 2 years of weekly returns
mu, Q, _ = factor_model(returns, kfac=3)
K, lo, hi = 8, np.full(40, 0.02), np.full(40, 0.3)
problem = hc.Problem(Q, mu, lo, hi, K, return_target(mu, lo, hi, K, 0.5))

params = hc.Params(check=("rigorous", "exact") + (("c",) if hc.checkers.c_checker() else ()))
result = hc.prove(problem, params=params)
print(f"status {result.status}, bound {result.bound:.6e}, spread {result.spread:.3f}")
for method, verdict in result.verdicts.items():
    print(f"  {method:<9} {verdict.status}")

path = os.path.join(tempfile.mkdtemp(), "cert.json")
result.certificate.save(path)
print(f"saved {path}; re-checked from the file: {hc.verify(path).status}")

x, objective, _ = polish(problem, result.relaxation_point)
gap = result.gap(x)
print(f"candidate portfolio: {np.count_nonzero(x)} names, objective {objective:.6e}, certified gap {100 * gap['gap']:.3f} %")
