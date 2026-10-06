"""Problems, and the risk models that produce their covariance.

    from heurics_cert.models import Problem, factor_model, return_target
    mu, Q, spec = factor_model(returns, kfac=10, floor=0.05)      # spec: the split the certificate uses
    problem = Problem(Q, mu, lo, hi, K, return_target(mu, lo, hi, K, 0.5))

Every builder is bit-reproducible: the same returns give the same model bytes, and the same fingerprint, on any
machine (no BLAS in the arithmetic).
"""
from heurics_cert.models.problem import Problem
from heurics_cert.models.risk import factor_form, factor_model, ledoit_wolf, sample_covariance, statistics
from heurics_cert.models.targets import extreme_return, return_target

__all__ = ["Problem", "extreme_return", "factor_form", "factor_model", "ledoit_wolf", "return_target",
           "sample_covariance", "statistics"]
