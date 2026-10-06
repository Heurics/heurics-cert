"""Verdicts, as an enum rather than strings.

Certificate checks end in one of ``PROVED``, ``REFUTED``, ``NOT_PROVED`` or ``MODEL_MISMATCH``; ``UNAVAILABLE`` and
``ERROR`` say a checker could not be run. Tree checks end in ``OPTIMAL``, ``BOUND``, ``REFUTED`` or ``NOT_PROVED``.
``NOT_PROVED`` is never a refutation: it means the claim could not be established at this precision.
"""
from __future__ import annotations

import enum


class Status(str, enum.Enum):
    PROVED = "PROVED"
    REFUTED = "REFUTED"
    NOT_PROVED = "NOT PROVED"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"
    OPTIMAL = "OPTIMAL"
    BOUND = "BOUND"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, text: str) -> "Status":
        """The status at the head of a checker's verdict string (``"REFUTED: reason"`` -> ``REFUTED``)."""
        head = str(text).split(":", 1)[0].strip().replace("_", " ")
        for s in cls:
            if s.value.replace("_", " ") == head:
                return s
        return cls.ERROR

    @property
    def ok(self) -> bool:
        """True for the outcomes that establish the claim (``PROVED``, ``OPTIMAL``)."""
        return self in (Status.PROVED, Status.OPTIMAL)


#: process exit codes, shared by the command line and the C checker
EXIT_CODES = {Status.PROVED: 0, Status.OPTIMAL: 0, Status.REFUTED: 1, Status.MODEL_MISMATCH: 2}


def exit_code(status: Status) -> int:
    return EXIT_CODES.get(status, 3)
