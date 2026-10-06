"""The outcome of ``prove``."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from heurics_cert.certificates.certificate import Certificate
from heurics_cert.status import Status
from heurics_cert.checkers.api import Verdict


@dataclass
class Result:
    """A certificate, the checkers' verdicts on it, and how it was obtained.

    ``relaxation_value`` is the solver's objective at the certified split (a reference, not a bound);
    ``relaxation_point`` the point the witness was extracted at."""
    certificate: Certificate
    verdicts: dict[str, Verdict]
    split: str
    relaxation_value: float
    relaxation_point: np.ndarray | None = None
    ascent_accepted: int = 0
    margin: float = 0.0
    timings: dict = field(default_factory=dict)

    @property
    def status(self) -> Status:
        """PROVED when every requested checker proved the certificate; REFUTED or MODEL_MISMATCH if any checker
        said so; otherwise NOT_PROVED (including when no checker was run)."""
        states = [v.status for v in self.verdicts.values()]
        if states and all(s is Status.PROVED for s in states):
            return Status.PROVED
        for s in (Status.REFUTED, Status.MODEL_MISMATCH):
            if s in states:
                return s
        return Status.NOT_PROVED

    @property
    def proved(self) -> bool:
        return self.status is Status.PROVED

    @property
    def bound(self) -> float:
        return self.certificate.bound

    @property
    def spread(self) -> float | None:
        """The relaxation point's weight outside its K largest positions. Below ~2 % the certificate is reliably
        tight (gap < 1 %); larger values flag problems where the relaxation diversifies beyond K names."""
        if self.relaxation_point is None:
            return None
        x = np.asarray(self.relaxation_point, np.float64)
        return float(1.0 - np.sort(x)[::-1][:self.certificate.problem.K].sum())

    def gap(self, x, *, rel_tol: float = 1e-9) -> dict:
        """The certified gap of a candidate portfolio (validated first) against this bound."""
        return self.certificate.problem.gap(x, self.bound, rel_tol=rel_tol)
