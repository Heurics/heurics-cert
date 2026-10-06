"""The checker's verdict taxonomy: PROVED, REFUTED and NOT PROVED are three different claims.

A checker over rigorous float arithmetic can prove a certificate, refute it, or fail to decide. The
third outcome is not a weaker form of the second, and the two must never be conflated:

* **REFUTED** is a claim that the certificate is wrong. It may be issued only when the negation is
  itself *certified* -- a direction with a verified negative quadratic form, or a claim above a
  rigorous upper bound on what the witness supports. A REFUTED on a valid certificate is the worst
  thing a checker can do: it is a reproducible artifact indicting a correct solver.
* **NOT PROVED** means this checker, at this precision, cannot decide. Verified Cholesky fails at
  degeneracy by construction, so a valid certificate sitting exactly on a boundary lands here.

The exact rational tier decides every case, so it is the oracle: the rigorous tier may say NOT
PROVED where exact says PROVED, but it may never contradict exact in either direction.

The cases below pin the taxonomy in both the Python checker (`heurics_cert.checkers.bound`) and the C checker:
a valid `beta_z = 0` certificate, a claim exactly at the re-derived value, a split with a negative diagonal, and
`pi < 0` behind a PSD check that cannot complete.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from heurics_cert.checkers import bound as certcheck
from heurics_cert.checkers.native import binary

# a C checker built for this platform (make, or `heurics-cert build-checker`), if any
_C_BIN = binary()
_C_OK = _C_BIN is not None
_EXIT = {"PROVED": 0, "REFUTED": 1, "NOT PROVED": 3}


def _status(res: dict) -> str:
    return res["verdict"].split(":")[0]


def _cert(n: int = 6, **over) -> dict:
    """A small certificate whose z-block minimum is exactly 0 (`w = 0`) and whose re-derived bound is
    exactly `beta_z`: every per-asset minimum over `[0, 1]` of `d s^2` is 0, attained at `s = 0`."""
    c = {"Q": np.diag(np.linspace(2.0, 4.0, n)).tolist(), "mu": [0.1] * n, "lo": [0.0] * n,
         "hi": [1.0] * n, "K": n, "rho": 0.0, "d": [1.0] * n, "w": [0.0] * n,
         "nu": 0.0, "pi": 0.0, "lam": 0.0, "beta_z": 0.0, "claimed_bound": -1.0}
    c.update(over)
    return c


# (name, certificate, exact verdict, rigorous verdict)
CASES = [
    # The shipped false refutation. The checker rounds the corner `-beta_z` down to strengthen what
    # it proves, which at `beta_z = 0` makes it `-5e-324`: the bordered matrix *as constructed* is
    # not PSD, though the claim it stands for is valid with zero slack. Unprovable, never refutable.
    ("valid, beta_z = 0 on the boundary", _cert(), "PROVED", "NOT PROVED"),
    ("valid, beta_z with real slack", _cert(beta_z=-1e-3), "PROVED", "PROVED"),
    # The same error at a second site: the claim equals the exact re-derived value, and the rigorous
    # chain rounds that value down past it.
    ("valid, claim exactly at the re-derived value", _cert(beta_z=-1e-3, claimed_bound=-1e-3),
     "PROVED", "NOT PROVED"),
    ("beta_z overstated by 1", _cert(beta_z=1.0, claimed_bound=-10.0), "REFUTED", "REFUTED"),
    ("claim inflated to +10", _cert(beta_z=-1e-3, claimed_bound=10.0), "REFUTED", "REFUTED"),
    # A genuine corruption, previously reported as merely undecided.
    ("split leaves a negative diagonal", _cert(d=[3.0, 3.4, 3.8, 4.2, 4.6, 5.0]),
     "REFUTED", "REFUTED"),
    # `pi < 0` breaks weak duality outright; the unprovable `A = 0` after it must not mask that.
    ("pi < 0 behind an unprovable split", _cert(Q=np.eye(6).tolist(), pi=-1.0),
     "REFUTED", "REFUTED"),
]


class TestVerdictTaxonomy(unittest.TestCase):
    def test_exact_tier_is_the_oracle(self):
        for name, cert, want_exact, _ in CASES:
            with self.subTest(name):
                self.assertEqual(_status(certcheck.check_exact(cert)), want_exact)

    def test_rigorous_tier_verdicts(self):
        for name, cert, _, want in CASES:
            with self.subTest(name):
                res = certcheck.check_rigorous(cert)
                self.assertEqual(_status(res), want, res["verdict"])
                self.assertEqual(res["status"], want.replace(" ", "_"))

    def test_rigorous_never_contradicts_exact(self):
        for name, cert, _, _ in CASES:
            with self.subTest(name):
                ex, rg = _status(certcheck.check_exact(cert)), _status(certcheck.check_rigorous(cert))
                if rg == "REFUTED":
                    self.assertEqual(ex, "REFUTED", "rigorous refuted a certificate exact proves")
                if rg == "PROVED":
                    self.assertEqual(ex, "PROVED", "rigorous proved a certificate exact refutes")

    def test_refutation_is_carried_by_a_refuted_check(self):
        """A REFUTED verdict must name the check that certified it -- never fall out of a default."""
        for name, cert, _, want in CASES:
            if want != "REFUTED":
                continue
            with self.subTest(name):
                res = certcheck.check_rigorous(cert)
                self.assertTrue(any(ch["status"] == "refuted" for ch in res["checks"]), res)

    def test_python_cli_exit_codes(self):
        script = certcheck.__file__
        for name, cert, _, want in CASES:
            with self.subTest(name):
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump(cert, f)
                try:
                    p = subprocess.run([sys.executable, script, f.name],
                                       capture_output=True, text=True, timeout=120)
                finally:
                    os.unlink(f.name)
                self.assertEqual(p.returncode, _EXIT[want], p.stdout + p.stderr)


@unittest.skipUnless(_C_OK, "ccpocheck binary not built for this platform")
class TestMalformedInput(unittest.TestCase):
    """Every input gets a verdict. Malformed certificates crashed both tiers (ValueError, TypeError, LinAlgError) on
    about one fuzzed input in eight; a malformed certificate proves nothing, so it is REFUTED, from either tier."""

    MALFORMED = [
        ("a field missing", {k: v for k, v in _cert().items() if k != "w"}),
        ("a string scalar", _cert(rho="x")),
        ("a list where a scalar belongs", _cert(nu=[0.0])),
        ("None as a scalar", _cert(K=None)),
        ("NaN in Q", _cert(Q=[[float("nan") if i == j == 0 else (2.0 if i == j else 0.0) for j in range(6)]
                              for i in range(6)])),
        ("a vector one entry short", _cert(mu=[0.1] * 5)),
        ("a non-square Q", _cert(Q=[[1.0] * 6] * 5)),
        ("an infinite multiplier", _cert(lam=float("inf"))),
    ]

    def test_both_tiers_refute_instead_of_raising(self):
        for name, cert in self.MALFORMED:
            for tier in (certcheck.check_rigorous, certcheck.check_exact):
                with self.subTest(case=name, tier=tier.__name__):
                    res = tier(cert)
                    self.assertEqual(res["status"], "REFUTED", res["verdict"])
                    self.assertIn("malformed", res["verdict"])

    def test_exact_tier_answers_finite_but_astronomic_entries(self):
        """Finite entries near 1e300 are well formed; their exact products overflow a float, which crashed the exact
        tier's report formatting (OverflowError) before it could return the verdict exact arithmetic had decided."""
        for cert in (_cert(w=[1e300] + [0.0] * 5), _cert(beta_z=-1e300), _cert(nu=1e300, claimed_bound=-1e300)):
            res = certcheck.check_exact(cert)
            self.assertIn(res["status"], ("PROVED", "REFUTED"))

    def test_well_formed_certificates_are_unaffected(self):
        for name, cert, exact, rigorous in CASES:
            with self.subTest(case=name):
                self.assertEqual(_status(certcheck.check_exact(cert)), exact)
                self.assertEqual(_status(certcheck.check_rigorous(cert)), rigorous)


@unittest.skipUnless(_C_OK, "ccpocheck binary not built for this platform")
class TestCParity(unittest.TestCase):
    """The independent C verifier must reach the same verdict, and say so with the same exit code."""

    def test_c_matches_python_rigorous(self):
        for name, cert, _, want in CASES:
            with self.subTest(name):
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump(cert, f)
                try:
                    p = subprocess.run([_C_BIN, f.name, "--json"],
                                       capture_output=True, text=True, timeout=120)
                finally:
                    os.unlink(f.name)
                self.assertEqual(json.loads(p.stdout)["verdict"].split(":")[0], want, p.stdout)
                self.assertEqual(p.returncode, _EXIT[want])


class TestCardinalityField(unittest.TestCase):
    """`K` is a positive integer by the format. The C checker cast it to `long` (undefined behaviour out of range):
    K = 1e300 became LONG_MIN and a certificate claiming ~1e15 on a model whose optimum is ~2e-3 was PROVED; a
    non-integer K was hashed as its truncation, so the two checkers fingerprinted different models."""

    BAD = (1e300, 2.5, 0, -3, 2.0 ** 60)

    def _c(self, cert):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cert, f)
        try:
            return subprocess.run([_C_BIN, f.name, "--json"], capture_output=True, text=True, timeout=120)
        finally:
            os.unlink(f.name)

    def test_every_checker_refuses_a_k_that_is_not_a_positive_integer(self):
        for K in self.BAD:
            cert = _cert(K=K, lam=1.0, claimed_bound=1e15)
            with self.subTest(K=K):
                self.assertEqual(certcheck.check_rigorous(cert)["status"], "REFUTED")
                self.assertEqual(certcheck.check_exact(cert)["status"], "REFUTED")
                if not _C_OK:
                    continue
                p = self._c(cert)
                self.assertNotIn("PROVED", p.stdout.split("verdict")[-1][:20] if p.stdout else "")
                self.assertNotEqual(p.returncode, _EXIT["PROVED"], p.stdout + p.stderr)

    @unittest.skipUnless(_C_OK, "ccpocheck binary not built for this platform")
    def test_an_integral_float_k_fingerprints_like_the_integer(self):
        for cert in (_cert(K=6), _cert(K=6.0)):
            p = self._c(cert)
            self.assertEqual(json.loads(p.stdout)["model_hash"], certcheck.check_rigorous(cert)["model_hash"])


class TestProverBoundary(unittest.TestCase):
    """The prover must leave the checker real slack on the z-block even at ``w = 0``: the witness at ``x_hat = 0``
    has ``w = 2 A 0 = 0``, exactly the boundary where a zero-slack ``beta_z`` is unprovable."""

    @classmethod
    def setUpClass(cls):
        from heurics_cert.models import Problem
        from heurics_cert.prover import tangent
        rng = np.random.default_rng(0)
        n = 12
        G = rng.standard_normal((n, n)) * 0.05
        Q = 3.0 * np.eye(n) + 0.5 * (G + G.T)
        Q = 0.5 * (Q + Q.T)
        mu = rng.uniform(0.05, 0.15, n)
        P = Problem(Q, mu, np.full(n, 0.01), np.full(n, 0.5), n, float(np.quantile(mu, 0.3)))
        cls.witness_obj = tangent.extract(P, np.full(n, 1.0), np.zeros(n))
        cls.witness = {k: (v.tolist() if hasattr(v, "tolist") else v)
                       for k, v in dict(P.fields(), **cls.witness_obj.fields()).items()}

    def test_instance_reaches_the_boundary(self):
        self.assertEqual(float(np.max(np.abs(self.witness["w"]))), 0.0)

    def test_beta_z_is_strictly_negative(self):
        self.assertLess(self.witness["beta_z"], 0.0)

    def test_boundary_certificate_is_proved_by_both_tiers(self):
        self.assertEqual(certcheck.check_exact(self.witness)["verdict"], "PROVED")
        res = certcheck.check_rigorous(self.witness)
        self.assertEqual(res["verdict"], "PROVED", res)

    def test_slack_costs_no_more_than_roundoff_scale(self):
        Q = np.asarray(self.witness["Q"])
        scale = float(np.sum(np.abs(np.diag(Q))))
        self.assertLess(-self.witness["beta_z"], 1e-9 * scale)


if __name__ == "__main__":
    unittest.main()
