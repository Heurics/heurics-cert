"""Tree certificates (CCPO-TREE/0): the record of a branch-and-bound search that proves an optimum.

The record lists nodes, splits, terminal proofs, tangent points and the incumbent (docs/TREE_CERTIFICATE.md). It is
checked against the model and the root certificate's witness, never trusted.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass

from heurics_cert.errors import FormatError

FORMAT = "CCPO-TREE/0"


@dataclass
class TreeCertificate:
    record: dict

    @property
    def fingerprint(self) -> str:
        return self.record.get("model_sha256", "")

    def check(self, problem, witness, *, method: str = "rigorous", rel_tol: float = 1e-6) -> dict:
        """Verify against the model the reader holds and the root witness. Returns ``dict(verdict, LB, U, gap_pct,
        ...)`` with verdict ``OPTIMAL``, ``BOUND``, ``REFUTED`` or ``NOT PROVED``."""
        from heurics_cert.checkers.tree import check_tree
        return check_tree(problem, witness, self.record, method=method, rel_tol=rel_tol)

    def save(self, path: str) -> None:
        """Write as JSON; gzip-compressed when ``path`` ends in ``.gz``."""
        data = json.dumps(self.record).encode()
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(gzip.compress(data) if path.endswith(".gz") else data)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "TreeCertificate":
        with open(path, "rb") as fh:
            data = fh.read()
        if path.endswith(".gz"):
            data = gzip.decompress(data)
        record = json.loads(data)
        if not isinstance(record, dict) or "nodes" not in record:
            raise FormatError(f"{path}: not a {FORMAT} record")
        return cls(record)
