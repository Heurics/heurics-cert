"""The witness: the O(n) part of a bound certificate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Witness:
    """A diagonal split ``d``, a tangent ``(w, beta_z)``, multipliers ``(nu, pi, lam)`` and the claimed bound.

    It proves ``x'Qx >= claimed_bound`` on the feasible set when ``Q - diag(d)`` is PSD, ``beta_z`` bounds
    ``min_z z'(Q - diag d)z - w'z`` from below, and ``pi, lam >= 0`` -- which is exactly what the checkers verify."""
    d: np.ndarray
    w: np.ndarray
    nu: float
    pi: float
    lam: float
    beta_z: float
    claimed_bound: float

    def fields(self) -> dict:
        return {"d": self.d, "w": self.w, "nu": float(self.nu), "pi": float(self.pi), "lam": float(self.lam),
                "beta_z": float(self.beta_z), "claimed_bound": float(self.claimed_bound)}

    @classmethod
    def from_document(cls, doc: dict) -> "Witness":
        return cls(d=np.asarray(doc["d"], np.float64), w=np.asarray(doc["w"], np.float64), nu=float(doc["nu"]),
                   pi=float(doc["pi"]), lam=float(doc["lam"]), beta_z=float(doc["beta_z"]),
                   claimed_bound=float(doc["claimed_bound"]))
