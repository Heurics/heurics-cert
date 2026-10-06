"""The problem a certificate is about: cardinality-constrained mean-variance with buy-in thresholds.

    min  x'Qx
    s.t. 1'x = 1,   mu'x >= rho,   lo_i y_i <= x_i <= hi_i y_i,   1'y <= K,   y in {0,1}^n
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from heurics_cert.errors import InvalidProblemError
from heurics_cert.checkers.bound import model_hash


@dataclass(frozen=True)
class Problem:
    """A cardinality-constrained mean-variance problem. Arrays are stored as contiguous float64.

    ``Q`` must be exactly symmetric, ``0 <= lo <= hi`` elementwise, ``K >= 1``. ``lo`` and ``hi`` may be scalars.
    ``name`` is a label only; it is not part of the model and not fingerprinted."""
    Q: np.ndarray
    mu: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    K: int
    rho: float
    name: str = field(default="", compare=False)

    def __post_init__(self):
        def f64(a):
            return np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        Q = f64(self.Q)
        n = Q.shape[0] if Q.ndim else 0
        if Q.ndim != 2 or Q.shape != (n, n):
            raise InvalidProblemError(f"Q must be square, got shape {Q.shape}")
        if not np.array_equal(Q, Q.T):
            raise InvalidProblemError("Q must be exactly symmetric")
        mu = f64(self.mu)
        lo, hi = (np.asarray(v, dtype=np.float64) for v in (self.lo, self.hi))
        lo = f64(np.broadcast_to(lo, (n,)) if lo.ndim == 0 else lo)
        hi = f64(np.broadcast_to(hi, (n,)) if hi.ndim == 0 else hi)
        for name, v in (("mu", mu), ("lo", lo), ("hi", hi)):
            if v.shape != (n,):
                raise InvalidProblemError(f"{name} must have shape ({n},), got {v.shape}")
        if np.any(lo < 0) or np.any(lo > hi):
            raise InvalidProblemError("need 0 <= lo <= hi elementwise")
        if int(self.K) < 1:
            raise InvalidProblemError("K must be >= 1")
        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "rho", float(self.rho))

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def fields(self) -> dict:
        """The model fields of the certificate format (``Q, mu, lo, hi, K, rho``)."""
        return {"Q": self.Q, "mu": self.mu, "lo": self.lo, "hi": self.hi, "K": self.K, "rho": self.rho}

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the model fields, exactly as every checker computes and reports it."""
        return model_hash(self.fields())

    @property
    def cardinality_implied(self) -> bool:
        """Whether the relaxed cardinality row ``sum x_i/hi_i <= K`` is implied by the budget (uniform ``hi`` and
        ``K * hi >= 1``). On a singular ``Q`` the relaxation is then the convex hull of the feasible set."""
        return bool(np.allclose(self.hi, self.hi[0])) and self.K * float(self.hi.min()) >= 1.0

    def objective(self, x) -> float:
        x = np.asarray(x, np.float64)
        return float(x @ self.Q @ x)

    def feasibility(self, x, *, rel_tol: float = 1e-9) -> dict:
        """Row residuals of a candidate, each relative to its own scale, and ``feasible``."""
        x = np.asarray(x, np.float64)
        on = x > 0
        rs = max(abs(self.rho), float(np.abs(self.mu).max()) * 1e-12, 1e-300)
        res = {
            "budget": abs(float(x.sum()) - 1.0),
            "return": max(0.0, (self.rho - float(self.mu @ x)) / rs),
            "box": float(np.max(np.maximum(self.lo - x, 0.0)[on], initial=0.0)
                         + np.max(np.maximum(x - self.hi, 0.0), initial=0.0)),
            "negative": float(max(0.0, -x.min())),
            "cardinality": int(on.sum()) - self.K,
        }
        res["feasible"] = (res["budget"] <= rel_tol and res["return"] <= rel_tol and res["box"] <= rel_tol
                           and res["negative"] == 0.0 and res["cardinality"] <= 0)
        return res

    def gap(self, x, bound: float, *, rel_tol: float = 1e-9) -> dict:
        """Relative gap of a candidate portfolio against a (proved) lower bound. The candidate is validated first:
        an infeasible point can sit below the optimum and make a gap look small or negative."""
        feas = self.feasibility(x, rel_tol=rel_tol)
        obj = self.objective(x)
        return {"objective": obj, "bound": float(bound), "feasible": feas["feasible"], "residuals": feas,
                "gap": (obj - bound) / abs(obj) if feas["feasible"] and obj != 0 else float("nan")}

    @classmethod
    def from_npz(cls, path: str, *, matrices_dir: str | None = None) -> "Problem":
        """Load a problem stored as ``.npz`` (``Q`` inline, or by content hash ``qref`` in a ``matrices/``
        directory beside the problem's own directory)."""
        z = np.load(path, allow_pickle=False)
        if "Q" in z.files:
            Q = z["Q"]
        else:
            mdir = matrices_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "matrices")
            Q = np.load(os.path.join(mdir, str(z["qref"]) + ".npz"))["Q"]
        name = str(z["name"]) if "name" in z.files else os.path.basename(path)[:-4]
        return cls(Q=Q, mu=z["mu"], lo=z["lo"], hi=z["hi"], K=int(z["K"]), rho=float(z["rho"]), name=name)
