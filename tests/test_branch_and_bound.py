"""Tree certificates: the branch-and-bound's optimum matches enumeration, `treecheck` certifies it, and every
tampering of the record is refused (or at least never certified above the true optimum)."""
import copy
import itertools
import unittest

import numpy as np
from scipy.optimize import minimize

from heurics_cert.checkers import tree as treecheck
from heurics_cert.models import Problem
from heurics_cert.prover import fista, tangent
from heurics_cert.search import primal
from heurics_cert.search.branch_and_bound import BranchAndBound


def _instance(seed):
    rng = np.random.default_rng(seed)
    n, K = 10, int(rng.integers(2, 4))
    F = rng.normal(size=(n, 3)) * 0.1
    spec = rng.uniform(0.002, 0.02, n)
    Q = F @ F.T + np.diag(spec)
    mu = rng.uniform(0.0, 0.1, n)
    P = Problem(Q=0.5 * (Q + Q.T), mu=mu, lo=np.full(n, 0.05), hi=np.full(n, 0.6 if K > 2 else 0.7), K=K,
                rho=float(np.quantile(mu, 0.6)))
    return P, spec


def _enumerate(P):
    """The optimum by enumerating every support; each support QP by SLSQP (an independent reference)."""
    best = np.inf
    for k in range(1, P.K + 1):
        for S in itertools.combinations(range(P.n), k):
            S = list(S)
            if P.lo[S].sum() > 1 or P.hi[S].sum() < 1:
                continue
            QS = P.Q[np.ix_(S, S)]
            cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0},
                    {"type": "ineq", "fun": lambda x, S=S: P.mu[S] @ x - P.rho}]
            for x0 in (np.full(k, 1.0 / k), primal.max_return_point(P.mu[S], P.lo[S], P.hi[S])):
                r = minimize(lambda x: x @ QS @ x, x0, jac=lambda x: 2 * QS @ x, method="SLSQP",
                             bounds=list(zip(P.lo[S], P.hi[S])), constraints=cons,
                             options={"ftol": 1e-15, "maxiter": 500})
                if r.success and abs(r.x.sum() - 1) < 1e-9 and P.mu[S] @ r.x >= P.rho - 1e-10:
                    best = min(best, float(r.x @ QS @ r.x))
    return best


def _solve(P, spec, **kw):
    d = spec * (1 - 1e-9)
    rel = fista.solve(P, d)
    W = tangent.extract(P, d, rel.x, rel.duals)
    x0, f0, _ = primal.polish(P, np.asarray(rel.x, np.float64), time_limit=2.0)
    bb = BranchAndBound(P, W, rel.x, x0, f0, time_limit=60.0, **kw)
    res = bb.run()
    return W, bb, res, bb.record()


def _mutated(rec, mutate):
    """A deep copy of the record with `mutate` applied to it."""
    t = copy.deepcopy(rec)
    mutate(t)
    return t


class TreeCertificates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for seed in (3, 11, 19):
            P, spec = _instance(seed)
            W, bb, res, rec = _solve(P, spec)
            cls.cases.append((P, W, res, rec, _enumerate(P)))

    def test_certified_optimum_matches_enumeration(self):
        for P, W, res, rec, opt in self.cases:
            v = treecheck.check_tree(P, W, rec)
            self.assertEqual(v["verdict"], "OPTIMAL", v)
            self.assertLessEqual(v["LB"], opt * (1 + 1e-9), "certified bound above the true optimum")
            self.assertAlmostEqual(v["U"] / opt, 1.0, delta=1e-6)

    def test_search_and_checker_agree(self):
        for P, W, res, rec, opt in self.cases:
            self.assertEqual(res["closed"], treecheck.check_tree(P, W, rec)["verdict"] == "OPTIMAL")

    def test_tampering_is_refused(self):
        P, W, res, rec, opt = self.cases[0]

        def resplit(t):
            k = next(iter(t["split"]))
            t["split"][k] = (t["split"][k] + 1) % P.n

        def drop_terminal(t):
            del t["term"][next(k for k, p in t["term"].items() if p["kind"] == "bound")]

        def negative_lam(t):
            next(p for p in t["term"].values() if p["kind"] == "bound")["lam"] = -1.0

        def other_split(t):
            t["root_d"] = [2 * v for v in t["root_d"]]

        def other_format(t):
            t["format"] = "CCPO-TREE/9"

        for mutate in (resplit, drop_terminal, negative_lam, other_split, other_format):
            self.assertEqual(treecheck.check_tree(P, W, _mutated(rec, mutate))["verdict"], "REFUTED", mutate.__name__)

    def test_other_model_is_refused(self):
        P, W, res, rec, opt = self.cases[0]
        P2 = Problem(Q=P.Q, mu=P.mu * 1.01, lo=P.lo, hi=P.hi, K=P.K, rho=P.rho)
        self.assertEqual(treecheck.check_tree(P2, W, rec)["verdict"], "REFUTED")

    def test_manipulated_values_never_certify_above_optimum(self):
        P, W, res, rec, opt = self.cases[1]

        def inflate_tangent(t):
            t["tangents"][0] = (t["tangents"][0][0], [3 * x for x in t["tangents"][0][1]])

        def shrink_incumbent(t):
            t["incumbent"]["val"] = [0.9 * x for x in t["incumbent"]["val"]]

        for mutate in (inflate_tangent, shrink_incumbent):
            v = treecheck.check_tree(P, W, _mutated(rec, mutate))
            self.assertFalse(v["verdict"] == "OPTIMAL" and v["LB"] > opt * (1 + 1e-9), mutate.__name__)

    def _suboptimal_incumbent(self, P, opt):
        """A feasible portfolio clearly above the optimum: what a forged record would try to pass off as optimal."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            x, f, _ = primal.polish(P, rng.uniform(size=P.n), time_limit=0.3)
            if x is not None and f > 1.05 * opt:
                S = np.flatnonzero(x)
                return dict(idx=S.tolist(), val=x[S].tolist())
        self.skipTest("no suboptimal feasible portfolio found")

    def test_empty_tree_is_refused(self):
        """No nodes, no terminals: the minimum over terminals is +inf and used to certify ANY incumbent optimal."""
        P, W, res, rec, opt = self.cases[0]
        t = dict(rec, nodes=[], split={}, term={}, incumbent=self._suboptimal_incumbent(P, opt))
        self.assertEqual(treecheck.check_tree(P, W, t)["verdict"], "REFUTED")

    def test_proof_at_the_root_sentinel_is_refused(self):
        """`at = -1` used to pass the ancestor walk (which ends at -1) and alias IN[-1], the LAST node's sets. A chain
        whose last node holds K+1 assets turned every bound into +inf and certified a suboptimal portfolio."""
        P, W, res, rec, opt = self.cases[0]
        nodes, split, term, p = [[-1, -1, -1]], {}, {}, 0
        for i in range(P.K + 1):
            split[str(p)] = i
            nodes.append([p, i, 0])
            term[str(len(nodes) - 1)] = dict(kind="bound", at=-1, t=0, nu=0.0, pi=0.0, lam=0.0, lo0=False, val=0.0)
            nodes.append([p, i, 1])
            p = len(nodes) - 1
        term[str(p)] = dict(kind="card")
        t = dict(rec, nodes=nodes, split=split, term=term, incumbent=self._suboptimal_incumbent(P, opt))
        self.assertEqual(treecheck.check_tree(P, W, t)["verdict"], "REFUTED")

    def test_repeated_incumbent_index_is_not_an_incumbent(self):
        """One asset listed twice at w/2 each passes a per-entry `hi` check while the portfolio holds w of it."""
        P, W, res, rec, opt = self.cases[0]
        j = next(i for i in np.argsort(np.diag(P.Q)) if P.mu[i] >= P.rho)
        t = copy.deepcopy(rec)
        t["incumbent"] = dict(idx=[int(j), int(j)], val=[0.5, 0.5])      # asset j at weight 1 > hi
        v = treecheck.check_tree(P, W, t)
        self.assertIsNone(v.get("U"), v)
        self.assertNotEqual(v["verdict"], "OPTIMAL")

    def test_a_tangent_point_with_a_repeated_index_is_refused(self):
        """Splitting one tangent entry across a repeated index keeps the point's sum but decouples `w` from `beta`;
        forged that way, the binding terminal's bound was certified 0.3-1.7 % ABOVE the true optimum."""
        P, W, res, rec, opt = self.cases[0]

        def duplicate_entry(t):
            idx, val = (list(v) for v in t["tangents"][0])
            val[0] -= 0.5
            t["tangents"][0] = (idx + [idx[0]], val + [0.5])

        def out_of_range_entry(t):
            idx, val = (list(v) for v in t["tangents"][0])
            t["tangents"][0] = (idx + [P.n + 1], val + [0.0])

        for mutate in (duplicate_entry, out_of_range_entry):
            self.assertEqual(treecheck.check_tree(P, W, _mutated(rec, mutate))["verdict"], "REFUTED", mutate.__name__)

    def test_malformed_records_are_refused_not_crashed_on(self):
        P, W, res, rec, opt = self.cases[0]

        def stray_terminal(t):
            t["term"][str(10 ** 6)] = {"kind": "card"}

        def asset_out_of_range(t):
            k = next(iter(t["split"]))
            t["split"][k] = P.n + 3
            for nd in t["nodes"]:
                if str(nd[0]) == k:
                    nd[1] = P.n + 3

        def truncated_node(t):
            del t["nodes"][1][2]

        for mutate in (stray_terminal, asset_out_of_range, truncated_node):
            self.assertEqual(treecheck.check_tree(P, W, _mutated(rec, mutate))["verdict"], "REFUTED", mutate.__name__)

        def incumbent_out_of_range(t):
            t["incumbent"]["idx"][0] = 10 ** 6

        self.assertNotEqual(treecheck.check_tree(P, W, _mutated(rec, incumbent_out_of_range))["verdict"], "OPTIMAL")

    def test_unfinished_tree_certifies_a_bound(self):
        P, spec = _instance(3)
        W, bb, res, rec = _solve(P, spec, node_limit=2)
        v = treecheck.check_tree(P, W, rec)
        self.assertIn(v["verdict"], ("BOUND", "OPTIMAL"))
        self.assertLessEqual(v["LB"], self.cases[0][4] * (1 + 1e-9))

    def test_the_record_cannot_loosen_the_acceptance_tolerance(self):
        """The verdict uses the reader's tolerance; the record's `rel_tol` set to 1 used to label any gap OPTIMAL."""
        P, spec = _instance(3)
        W, bb, res, rec = _solve(P, spec, node_limit=2)
        v = treecheck.check_tree(P, W, rec)
        if v["verdict"] == "OPTIMAL":
            self.skipTest("the two-node search already closed")
        loose = treecheck.check_tree(P, W, dict(rec, rel_tol=1.0))
        self.assertEqual(loose["verdict"], "BOUND", loose)
        self.assertEqual(loose["rel_tol"], 1e-6)
        self.assertEqual(treecheck.check_tree(P, W, rec, rel_tol=1.0)["verdict"], "OPTIMAL")   # the reader may


if __name__ == "__main__":
    unittest.main()
