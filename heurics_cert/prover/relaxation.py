"""The perspective relaxation, and the solvers that propose its tangent point.

    min  x'Ax + sum_i d_i x_i^2 / y_i
    s.t. 1'x = 1,  mu'x >= rho,  lo y <= x <= hi y,  0 <= y <= 1,  1'y <= K          (Q = A + diag(d))

A solver's value is not a bound. Only the witness extracted at its point is, and that witness is valid whatever the
solver's accuracy: an inaccurate solve costs tightness, never soundness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heurics_cert.params import DEFAULT, Params

SOLVERS = ("fista", "clarabel")


@dataclass
class Relaxation:
    """A point of the relaxation. ``value`` is the objective there (a reference, not a bound); ``duals`` holds the
    budget, return and cardinality multipliers in model units, used to seed the witness search."""
    x: np.ndarray
    y: np.ndarray
    value: float
    status: str
    duals: dict


def solve(problem, d, *, params: Params = DEFAULT, pi_hint: float | None = None) -> Relaxation | None:
    """Solve the relaxation for split ``d`` with ``params.solver``; None when the solver produced no point."""
    if params.solver == "fista":
        from heurics_cert.prover import fista
        return fista.solve(problem, d, grid=params.fista_grid, rounds=params.fista_rounds,
                           coarse=params.fista_coarse, final=params.fista_final, pi_hint=pi_hint)
    if params.solver == "clarabel":
        from heurics_cert.prover import clarabel
        return clarabel.solve(problem, d, time_limit=params.clarabel_time_limit)
    raise ValueError(f"unknown solver {params.solver!r}; expected one of {SOLVERS}")
