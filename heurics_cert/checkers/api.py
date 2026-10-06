"""``verify``: one entry point for every certificate checker, returning a typed ``Verdict``."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from heurics_cert.status import Status
from heurics_cert.checkers import bound

METHODS = ("rigorous", "exact", "c")


@dataclass
class Verdict:
    """The outcome of one checker on one certificate."""
    method: str
    status: Status
    certified_bound: float | None = None
    model_hash: str | None = None
    reason: str = ""
    detail: dict = field(default_factory=dict, repr=False)

    @property
    def proved(self) -> bool:
        return self.status is Status.PROVED


def _document(cert, qbin):
    """A CCPO-CERT document from a path (with ``qbin``: the split layout), a dict, or a ``Certificate``."""
    if isinstance(cert, (str, os.PathLike)):
        path = os.fspath(cert)
        if not qbin:
            return bound.load(path)
        import json
        from heurics_cert.checkers import qbin as _qbin
        with open(path) as fh:
            doc = json.load(fh)
        doc["Q"] = _qbin.read(qbin)
        doc.pop("n", None)
        return doc
    return cert.document() if hasattr(cert, "document") else cert


def _plain(doc):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in doc.items()}


def verify(cert, method: str = "rigorous", *, qbin: str | None = None, expect_model: str | None = None,
           timeout: float = 1800.0) -> Verdict:
    """Check a certificate (a ``Certificate``, a CCPO-CERT document, or a path to one).

    ``method``: ``"rigorous"`` (float64, outward rounding), ``"exact"`` (rationals; n up to ~200) or ``"c"`` (the
    independent C99 checker). ``qbin``: the binary model file of the split layout. ``expect_model``: a fingerprint
    (prefix) the reader holds; any other model yields ``MODEL_MISMATCH``. Copositive (CCPO-CERT/2) documents, those
    carrying ``sigma``, are routed through their reduction."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    doc = _document(cert, qbin)
    if "sigma" in doc:
        from heurics_cert.checkers import copositive
        r = copositive.check(doc, method=method, qbin=qbin)
    elif method == "rigorous":
        r = bound.check_rigorous(doc)
    elif method == "exact":
        r = bound.check_exact(_plain(doc))
    else:
        from heurics_cert.checkers import native
        r = native.run(doc, qbin=qbin, expect_model=expect_model, timeout=timeout)
    raw = str(r.get("verdict", "ERROR"))
    status = Status.parse(raw)
    reason = r.get("reason") or (raw.split(":", 1)[1].strip() if ":" in raw else "")
    mh = r.get("model_hash")
    if expect_model and mh and not str(mh).startswith(expect_model) and status is not Status.UNAVAILABLE:
        status = Status.MODEL_MISMATCH
    return Verdict(method, status, r.get("certified_bound"), mh, reason, r)
