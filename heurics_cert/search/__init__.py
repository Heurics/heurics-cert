"""Search: feasible portfolios, and the certified branch-and-bound that proves an optimum.

    from heurics_cert.search import BranchAndBound, polish
    x, f, info = polish(problem, result.relaxation_point)
    bb = BranchAndBound(problem, result.certificate.witness, result.relaxation_point, x, f)
    bb.run(); tree = bb.certificate()          # CCPO-TREE/0, checked by heurics_cert.checkers.tree

These are research instruments: certifying a portfolio needs only ``prove``. Nothing here is trusted.
"""
from heurics_cert.search.branch_and_bound import BranchAndBound
from heurics_cert.search.primal import polish, restricted_qp

__all__ = ["BranchAndBound", "polish", "restricted_qp"]
