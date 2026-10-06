"""Parameters of the prover, in one place.

    params = hc.Params(solver="clarabel", check=("rigorous",))
    hc.prove(problem, params=params)

Every field has a default that reproduces the paper. None of them affects soundness: they trade time for tightness,
or choose which independent checkers confirm the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Params:
    #: relaxation solver proposing the tangent point: ``"fista"`` (ours, JAX) or ``"clarabel"`` (interior point)
    solver: str = "fista"
    #: checkers that must prove the certificate: any of ``"rigorous"``, ``"exact"``, ``"c"``; ``()`` skips checking
    check: tuple[str, ...] = ("rigorous", "c")
    #: supergradient steps on the diagonal split (non-singular Q only)
    ascent_steps: int = 8
    #: safety multiples of the rigorous checker's Cholesky error scale, tried in order until the checkers prove
    margin_ladder: tuple[float, ...] = (64.0, 4096.0, 262144.0)
    #: wall-clock limit of one interior-point solve (``solver="clarabel"``)
    clarabel_time_limit: float = 900.0
    #: FISTA iteration budget: per bracketing round and for the final solve at the located multiplier
    fista_coarse: int = 500
    fista_final: int = 4000
    #: points on the return-multiplier grid, and bracketing rounds
    fista_grid: int = 24
    fista_rounds: int = 3
    #: seconds allowed for each checker run
    check_timeout: float = 1800.0
    extra: dict = field(default_factory=dict, compare=False)

    def replace(self, **changes) -> "Params":
        return replace(self, **changes)


DEFAULT = Params()
