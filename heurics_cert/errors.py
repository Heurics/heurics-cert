"""Exceptions raised by heurics-cert.

A checker never raises on a bad certificate: it returns a verdict (``Status.REFUTED`` with a reason). Exceptions
are for misuse of the API and for things that could not be run at all.
"""


class HeuricsCertError(Exception):
    """Base class for every error this package raises."""


class InvalidProblemError(HeuricsCertError, ValueError):
    """The model data are not a valid cardinality-constrained mean-variance problem."""


class FormatError(HeuricsCertError, ValueError):
    """A file is not in the format it claims to be (certificate, tree record, model file)."""


class SolverError(HeuricsCertError, RuntimeError):
    """A relaxation solver produced no usable point."""


class UnsupportedProblemError(HeuricsCertError, ValueError):
    """The problem is outside what the requested method handles (e.g. the batched path on a non-singular Q)."""
