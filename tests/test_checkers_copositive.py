"""CCPO-CERT/2: the one verifier must accept both families and refuse the forgeries the extra fields allow."""
from __future__ import annotations

import numpy as np
import pytest

from heurics_cert.checkers import bound as certcheck, copositive as coposcheck
from heurics_cert.models import Problem
from heurics_cert.prover import fista, splits, tangent


def _problem(n=14, K=3, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 3))
    Q = F @ F.T + np.diag(rng.uniform(0.4, 1.0, n))
    Q = 0.5 * (Q + Q.T)
    mu = rng.uniform(0.01, 0.05, n)
    lo = np.full(n, 0.01)
    hi = np.full(n, 1.0)
    return Problem(Q=Q, mu=mu, lo=lo, hi=hi, K=K, rho=float(np.median(mu)))


def _tier_a_cert(P):
    d = splits.candidate_splits(P.Q)["uniform"]
    rel = fista.solve(P, d)
    wit = tangent.extract(P, d, rel.x, rel.duals)
    doc = dict(P.fields(), **wit.fields())
    doc["sigma"] = 0.0
    doc["model_hash"] = certcheck.model_hash(doc)
    return doc, wit


def _copos_cert(P, sigma=0.3, scale=0.2):
    """A genuine tier-B shaped certificate: G_off from a nonnegative outer product, split from what is left."""
    n = P.n
    Q = P.Q
    v = np.abs(np.linalg.eigh(Q)[1][:, -1])
    G = scale * np.outer(v, v)
    np.fill_diagonal(G, 0.0)
    Qt = Q + sigma - G
    Qt = 0.5 * (Qt + Qt.T)
    lmin = float(np.linalg.eigvalsh(Qt)[0])
    d = np.full(n, max(0.0, lmin) * (1 - 1e-9))
    Pt = Problem(Q=Qt, mu=P.mu, lo=P.lo, hi=P.hi, K=P.K, rho=P.rho)
    rel = fista.solve(Pt, d)
    wit = tangent.extract(Pt, d, rel.x, rel.duals)
    doc = dict(P.fields(), **wit.fields())               # the ORIGINAL model travels with it
    doc["sigma"] = sigma
    doc["G_off"] = G
    doc["claimed_bound"] = wit.claimed_bound - sigma
    doc["model_hash"] = certcheck.model_hash(doc)
    return doc


class TestBothFamilies:
    def test_tier_a_passes_and_matches_the_ccpo1_checker(self):
        P = _problem()
        doc, wit = _tier_a_cert(P)
        res = coposcheck.check(doc, method="rigorous")
        assert res["verdict"] == "PROVED", res
        # sigma = 0 and no G_off: the delegated chain must see the model unchanged, so the bound agrees
        direct = certcheck.check_rigorous(dict(P.fields(), **wit.fields()))
        assert direct["verdict"] == "PROVED"
        assert res["certified_bound"] == pytest.approx(direct["certified_bound"], rel=0, abs=1e-18)

    def test_copositive_certificate_passes(self):
        P = _problem()
        doc = _copos_cert(P)
        res = coposcheck.check(doc, method="rigorous")
        assert res["verdict"] == "PROVED", res
        assert res["certified_bound"] >= doc["claimed_bound"] - 1e-12

    def test_the_verdict_binds_to_the_original_model_not_the_modified_one(self):
        P = _problem()
        doc = _copos_cert(P)
        res = coposcheck.check(doc, method="rigorous")
        assert res["model_hash"] == certcheck.model_hash(dict(P.fields()))

    def test_bound_is_below_the_true_optimum(self):
        """Brute force over supports: the certified bound may never exceed the optimum."""
        P = _problem(n=10, K=2)
        doc = _copos_cert(P, sigma=0.2, scale=0.1)
        res = coposcheck.check(doc, method="rigorous")
        assert res["verdict"] == "PROVED"
        import itertools
        best = np.inf
        for S in itertools.combinations(range(P.n), P.K):
            S = list(S)
            for _ in range(1):
                # the restricted optimum over the simplex on S, by projected gradient (enough for a test)
                x = np.full(len(S), 1.0 / len(S))
                Qs = P.Q[np.ix_(S, S)]
                for _ in range(4000):
                    g = 2 * Qs @ x
                    x = x - 0.02 * (g - g.mean())
                    x = np.maximum(x, 0.0)
                    x = x / max(x.sum(), 1e-300)
                if float(P.mu[S] @ x) >= P.rho - 1e-12 and np.all(x[x > 0] >= P.lo[S][x > 0] - 1e-9):
                    best = min(best, float(x @ Qs @ x))
        assert res["certified_bound"] <= best + 1e-9


class TestForgeries:
    def test_negative_entry_in_G_is_refuted(self):
        P = _problem()
        doc = _copos_cert(P)
        G = np.array(doc["G_off"], dtype=float)
        i, j = 0, 1
        G[i, j] = G[j, i] = -abs(G[i, j]) - 1e-6
        doc = dict(doc, G_off=G)
        doc["model_hash"] = certcheck.model_hash(doc)
        res = coposcheck.check(doc, method="rigorous")
        assert res["verdict"] == "REFUTED", res
        assert "G_off >= 0" in res["reason"]

    def test_asymmetric_G_is_refuted(self):
        P = _problem()
        doc = _copos_cert(P)
        G = np.array(doc["G_off"], dtype=float)
        G[0, 1] += 1e-3
        res = coposcheck.check(dict(doc, G_off=G), method="rigorous")
        assert res["verdict"] == "REFUTED"

    def test_nonzero_diagonal_in_G_is_refuted(self):
        P = _problem()
        doc = _copos_cert(P)
        G = np.array(doc["G_off"], dtype=float)
        G[3, 3] = 1e-4
        res = coposcheck.check(dict(doc, G_off=G), method="rigorous")
        assert res["verdict"] == "REFUTED"

    def test_inflating_sigma_to_inflate_the_claim_is_caught(self):
        """sigma is subtracted from the inner bound, so claiming a larger bound with the same witness fails."""
        P = _problem()
        doc = _copos_cert(P)
        bad = dict(doc, claimed_bound=doc["claimed_bound"] + 0.05 * abs(doc["claimed_bound"]) + 1e-9)
        bad["model_hash"] = certcheck.model_hash(bad)
        res = coposcheck.check(bad, method="rigorous")
        assert res["verdict"] in ("REFUTED", "NOT PROVED"), res

    def test_substituted_model_is_refused(self):
        P = _problem()
        doc = _copos_cert(P)
        narrowed = dict(doc)
        narrowed["hi"] = np.asarray(doc["hi"], float) * 0.5          # a strictly tighter problem
        res = coposcheck.check(narrowed, method="rigorous")
        assert res["verdict"] == "REFUTED"
        assert "different model" in res.get("reason", "")


class TestVerdictsStayThreeValued:
    def test_a_claim_within_rounding_is_not_proved_not_refuted(self):
        """One ulp above the rigorous bound is undecided: certcheck says NOT PROVED, and so must this verifier."""
        P = _problem()
        doc, _ = _tier_a_cert(P)
        cb = certcheck.check_rigorous(doc)["certified_bound"]
        doc = dict(doc, claimed_bound=float(np.nextafter(cb, np.inf)))
        assert certcheck.check_rigorous(doc)["status"] == "NOT_PROVED"
        assert coposcheck.check(doc, method="rigorous")["verdict"] == "NOT PROVED"

    def test_tier_b_needs_nonnegative_lo(self):
        """Dropping x'G_off x >= 0 needs x >= 0; a negative buy-in bound breaks the reduction."""
        doc = _copos_cert(_problem())
        assert coposcheck.check(doc)["verdict"] == "PROVED"
        bad = dict(doc, lo=[-0.5] * len(doc["mu"]))
        bad.pop("model_hash")
        res = coposcheck.check(bad)
        assert res["verdict"] == "REFUTED" and "lo" in res["reason"], res


class TestMalformedInput:
    def test_malformed_tier_b_fields_are_refuted_not_raised(self):
        """A fuzzer crashed this verifier on about one malformed input in thirty (ValueError, TypeError)."""
        doc = _copos_cert(_problem())
        doc.pop("model_hash")
        n = len(doc["mu"])
        bad = [dict(doc, sigma="x"), dict(doc, sigma=float("nan")), dict(doc, G_off=[[0.0] * n] * (n - 1)),
               dict(doc, G_off=[[0.0, "x"]] * n), dict(doc, K=None), dict(doc, rho="0.1"),
               {k: v for k, v in doc.items() if k != "sigma"}]
        for d in bad:
            res = coposcheck.check(d)
            assert res["verdict"] == "REFUTED" and "malformed" in res["reason"], res
