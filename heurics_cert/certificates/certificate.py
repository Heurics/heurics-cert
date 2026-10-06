"""Bound certificates: CCPO-CERT/1 (diagonal split) and CCPO-CERT/2 (copositive split), on disk as JSON.

Two layouts (docs/CERTIFICATE_FORMAT.md):

* self-contained -- every field in one JSON document;
* split -- the O(n) witness as JSON plus ``Q`` once per covariance in a binary model file (``qbin``), which many
  certificates on one covariance share. At n = 2,901 that is 0.2 MB of JSON instead of ~195 MB.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from heurics_cert.certificates.witness import Witness
from heurics_cert.errors import FormatError
from heurics_cert.models.problem import Problem
from heurics_cert.checkers import qbin as _qbin


@dataclass
class Certificate:
    """A problem, a witness, and for a copositive certificate ``sigma`` and ``G_off``.

    For a copositive certificate the witness is about ``Qt = Q + sigma 11' - G_off``; the certificate's bound is the
    witness's minus ``sigma``. The fingerprint is always that of ``problem``, the model the reader holds."""
    problem: Problem
    witness: Witness
    sigma: float = 0.0
    G_off: np.ndarray | None = None

    @property
    def copositive(self) -> bool:
        return self.sigma != 0.0 or (self.G_off is not None and bool(np.any(self.G_off)))

    @property
    def bound(self) -> float:
        """The claimed lower bound. It means nothing until a checker has proved it."""
        return float(self.witness.claimed_bound) - (float(self.sigma) if self.copositive else 0.0)

    @property
    def fingerprint(self) -> str:
        return self.problem.fingerprint

    def document(self) -> dict:
        """The CCPO-CERT document (arrays as numpy; every checker accepts them)."""
        doc = {**self.problem.fields(), **self.witness.fields()}
        if self.copositive:
            doc["sigma"] = float(self.sigma)
            if self.G_off is not None:
                doc["G_off"] = self.G_off
            doc["claimed_bound"] = self.bound
        doc["model_hash"] = self.problem.fingerprint
        return doc

    def witness_document(self) -> dict:
        """The split layout's JSON part: everything but ``Q``, plus ``n``."""
        doc = {k: v for k, v in self.document().items() if k != "Q"}
        doc["n"] = self.problem.n
        return doc

    def save(self, path: str, *, qbin: str | None = None) -> None:
        """Write the self-contained document, or with ``qbin`` the split layout (the model file is written only if it
        does not exist yet, so certificates on one covariance share it)."""
        doc = self.witness_document() if qbin else self.document()
        if qbin and not os.path.exists(qbin):
            _qbin.write(self.problem.Q, qbin)
        _dump(doc, path)


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _dump(doc: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({k: _plain(v) for k, v in doc.items()}, fh)
    os.replace(tmp, path)


def load(path: str, *, qbin: str | None = None) -> Certificate:
    """Read a certificate (self-contained, or the split layout with its ``qbin`` model file)."""
    with open(path) as fh:
        doc = json.load(fh)
    try:
        Q = _qbin.read(qbin) if qbin else np.asarray(doc["Q"], np.float64)
        problem = Problem(Q=Q, mu=doc["mu"], lo=doc["lo"], hi=doc["hi"], K=int(doc["K"]), rho=float(doc["rho"]))
        sigma = float(doc.get("sigma", 0.0))
        G = doc.get("G_off")
        witness = Witness.from_document(dict(doc, claimed_bound=float(doc["claimed_bound"]) + sigma))
    except KeyError as exc:
        raise FormatError(f"{path}: not a CCPO certificate (missing {exc})") from None
    return Certificate(problem, witness, sigma, None if G is None else np.asarray(G, np.float64))
