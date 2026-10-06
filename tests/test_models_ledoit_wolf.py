"""``heurics_cert.models.ledoit_wolf`` against the published definitions, written as loops."""
import numpy as np

from heurics_cert.models import ledoit_wolf


def _returns(seed=1, T=40, n=25):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) @ rng.standard_normal((n, n)) * 0.02


def test_identity_target_matches_definition():
    R = _returns()
    Q, d, delta = ledoit_wolf(R, "identity")
    T, n = R.shape
    X = R - R.mean(0)
    S = X.T @ X / T
    m = np.trace(S) / n
    nrm = lambda A: np.sum(A * A) / n                                   # noqa: E731  (the paper's normalised norm)
    d2 = nrm(S - m * np.eye(n))
    b2 = min(sum(nrm(np.outer(x, x) - S) for x in X) / T ** 2, d2)
    assert abs(delta - b2 / d2) < 1e-12
    assert np.allclose(Q, b2 / d2 * m * np.eye(n) + (1 - b2 / d2) * S, atol=1e-16, rtol=0)
    assert np.allclose(d, delta * m)


def test_constant_correlation_target_matches_definition():
    R = _returns()
    Q, d, delta = ledoit_wolf(R, "cc")
    T, n = R.shape
    X = R - R.mean(0)
    S = X.T @ X / T
    s = np.sqrt(np.diag(S))
    rbar = sum(S[i, j] / (s[i] * s[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    F = np.array([[S[i, i] if i == j else rbar * s[i] * s[j] for j in range(n)] for i in range(n)])
    pi = sum(np.mean((X[:, i] * X[:, j] - S[i, j]) ** 2) for i in range(n) for j in range(n))
    th = lambda i, j: np.mean((X[:, i] ** 2 - S[i, i]) * (X[:, i] * X[:, j] - S[i, j]))   # noqa: E731
    rho = sum(np.mean((X[:, i] ** 2 - S[i, i]) ** 2) for i in range(n)) + sum(
        rbar / 2 * (np.sqrt(S[j, j] / S[i, i]) * th(i, j) + np.sqrt(S[i, i] / S[j, j]) * th(j, i))
        for i in range(n) for j in range(n) if i != j)
    ref = max(0.0, min(1.0, (pi - rho) / np.sum((F - S) ** 2) / T))
    assert abs(delta - ref) < 1e-12
    assert np.allclose(Q, (1 - ref) * S + ref * F, atol=1e-16, rtol=0)


def test_split_leaves_psd_remainder():
    rng = np.random.default_rng(2)
    T, n = 30, 40                                                        # T < n: singular sample covariance
    R = 0.02 * (rng.standard_normal((T, 1)) + rng.standard_normal((T, n)))   # a market factor: mean correlation > 0
    for target in ("identity", "cc"):
        Q, d, _ = ledoit_wolf(R, target)
        assert (d > 0).all()
        assert np.linalg.eigvalsh(Q - np.diag(d))[0] > -1e-12 * np.trace(Q)


def test_constant_correlation_refuses_negative_mean_correlation():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((50, 1))
    R = 0.02 * np.hstack([z, -z]) + 0.001 * rng.standard_normal((50, 2))   # two names, correlation ~ -1
    try:
        ledoit_wolf(R, "cc")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a negative mean correlation")
