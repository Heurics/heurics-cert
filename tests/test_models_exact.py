"""The bit-reproducible model builders behind exact fingerprints (``heurics_cert.models``)."""
import hashlib

import numpy as np

from heurics_cert.models import _exact, risk, targets


def _returns():
    rng = np.random.default_rng(20261005)                  # PCG64: the same stream on every platform
    return 0.02 * (rng.standard_normal((40, 1)) + rng.standard_normal((40, 120)))


def test_pinned_bytes():
    """The exact construction's bytes, computed on Windows (numpy 2.2) and Linux (numpy 1.26 and 2.5, three OpenBLAS
    builds, 1-32 threads) alike. A different digest here means the construction is no longer machine-independent."""
    mu, Q, spec = risk.factor_model(_returns(), kfac=5, floor=0.05)
    h = hashlib.sha256()
    for a in (mu, Q, spec):
        h.update(np.ascontiguousarray(a, ">f8").tobytes())
    assert h.hexdigest() == "22f1f6d5e6b94c562c23ffacd2873aa3c9dfb18a6844876acd55c487ad5a5b31"
    lo, hi = np.full(120, 0.01), np.ones(120)
    assert targets.return_target(mu, lo, hi, 10, 0.5) == -0.0019699565089964528


def _digest(*arrays):
    return hashlib.sha256(b"".join(np.ascontiguousarray(a, ">f8").tobytes() for a in arrays)).hexdigest()


def test_pinned_bytes_shrinkage_and_factor_form():
    """Same for the Ledoit-Wolf and fundamental-model builders; inputs from the seeded generator only
    (an input built with BLAS, e.g. np.cov, would itself differ across machines)."""
    rng = np.random.default_rng(20261005)
    R = 0.02 * (rng.standard_normal((40, 1)) + rng.standard_normal((40, 120)))
    Q, d, delta = risk.ledoit_wolf(R, "identity")
    assert _digest(Q, d) == "dad4470cc2eb415b9849375fe3f296c6656607984bf3d6780ee7da80af678ac0"
    assert delta == 0.13420614734988026
    Q, d, _ = risk.ledoit_wolf(R, "cc")
    assert _digest(Q, d) == "bab7fa48884ecbe8dc9866eca5c2d266514adf9b235736a66fde27ee688a14c0"
    X = rng.standard_normal((120, 6))
    A = rng.standard_normal((6, 6))
    F = 0.01 * (A + A.T) + 0.1 * np.eye(6)
    sp = rng.uniform(0.01, 0.05, 120)
    Q = risk.factor_form(X, F, sp)
    assert _digest(Q) == "b66f85781b66b6529f96c0c924129ddf62e1688db8a24371efd742d4e9fb58a5"
    assert np.array_equal(Q, Q.T)
    assert np.abs(Q - (X @ F @ X.T + np.diag(sp))).max() <= 1e-13 * np.abs(Q).max()


def test_agrees_with_the_lapack_construction():
    R = _returns()
    mu, Q, spec = risk.factor_model(R, kfac=5, floor=0.05)
    S = np.cov(R, rowvar=False, ddof=1)
    w, U = np.linalg.eigh(S)
    B = U[:, -5:] * np.sqrt(w[-5:])
    sp = np.maximum(np.maximum(np.diag(S) - np.sum(B * B, axis=1), 0.0), 0.05 * np.diag(S))
    Q_ref = B @ B.T + np.diag(sp)
    assert np.allclose(mu, R.mean(axis=0), rtol=0, atol=1e-17)
    assert np.abs(Q - Q_ref).max() <= 1e-12 * np.abs(Q_ref).max()
    assert np.array_equal(Q, Q.T)


def test_jacobi_eigenpairs():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((30, 30))
    G = M @ M.T
    w, V = _exact.jacobi_eigh(G)
    assert np.all(np.diff(w) <= 0)
    assert np.allclose(V.T @ V, np.eye(30), atol=1e-12)
    assert np.allclose(G @ V, V * w, atol=1e-10 * np.abs(w).max())
    assert np.allclose(w, np.sort(np.linalg.eigvalsh(G))[::-1], rtol=1e-12, atol=1e-12 * w.max())


def test_exact_dot_is_correctly_rounded():
    a = np.array([1e16, 1.0, -1e16])
    b = np.array([1.0, 1.0, 1.0])
    assert _exact.dot(a, b) == 1.0                  # a naive left-to-right sum gives 0.0
