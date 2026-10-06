"""Certified branch-and-bound: proves an optimum and records the proof as a tree certificate (CCPO-TREE/0).

This is a measuring instrument, not part of certifying a portfolio: it establishes the optimum a bound is compared
against, and ``heurics_cert.checkers.tree`` verifies its record without trusting anything computed here.

**Node bound.** A node fixes IN (held) and OUT (excluded); FREE is the rest. For the root split ``Q = A + diag(d)``
(``A`` PSD, proved with the root certificate) and any tangent point ``x_hat``, ``x'Ax >= w'x + beta`` with
``w = 2 A x_hat``, ``beta = -x_hat'A x_hat``, so for every ``nu``, ``pi >= 0``, ``lam >= 0``

    bound = beta - nu + pi rho - lam (K - |IN|) + sum_IN m_i + sum_FREE min(m_i + lam, 0),
    m_i   = min_{lo_i <= s <= hi_i} d_i s^2 + (w_i + nu - pi mu_i) s

bounds every portfolio in the node. Tangent points: the root's, and at small nodes the continuous QP optimum over
IN u FREE (its multipliers seed the search). ``(nu, pi)`` are found on a shrinking grid; ``lam`` exactly.

**Fully decided nodes** (FREE empty) are convex QPs on their support, solved exactly; their optimum is offered as an
incumbent. **Search:** best-first (ties go deeper), reduced-cost fixing recorded as splits with a leaf on the pruned
side, and incumbents from the restricted QP on each node's most attractive support.
"""
from __future__ import annotations

import heapq
import itertools
import time

import numpy as np

from heurics_cert.certificates.tree import FORMAT
from heurics_cert.prover import active_set, tangent
from heurics_cert.search import primal


def _node_dual(w, beta, d, mu, lo, hi, rho, IN, FREE, slots, nu, pi):
    """The node bound at one (nu, pi), lam at its exact maximiser. Returns (value, lam, t_FREE, t_IN), t = m + lam."""
    J = np.concatenate([IN, FREE])
    m = tangent.per_asset_min(d[J], w[J] + nu - pi * mu[J], lo[J], hi[J])
    mi, mf = m[: IN.size], m[IN.size:]
    lam = max(0.0, -float(np.partition(mf, slots)[slots])) if slots < mf.size else 0.0
    val = beta - nu + pi * rho - lam * slots + float(mi.sum()) + float(np.minimum(mf + lam, 0.0).sum())
    return val, lam, mf + lam, mi + lam


def _dual_batch(w, beta, d, mu, lo, hi, rho, IN, FREE, slots, nus, pis):
    """The node bound at many (nu, pi): rows = multiplier pairs, columns = assets of IN u FREE."""
    J = np.concatenate([IN, FREE])
    dJ, loJ, hiJ = d[J], lo[J], hi[J]
    c = w[J][None, :] + nus[:, None] - pis[:, None] * mu[J][None, :]
    pos = dJ > 0
    safe = np.where(pos, dJ, 1.0)
    vert = np.where(pos, np.clip(-c / (2.0 * safe), loJ, hiJ), loJ)
    f = lambda s: dJ * s * s + c * s                           # noqa: E731
    m = np.minimum(np.minimum(f(loJ), f(hiJ)), f(vert))
    mi, mf = m[:, :IN.size], m[:, IN.size:]
    if slots < mf.shape[1]:
        lam = np.maximum(0.0, -np.partition(mf, slots, axis=1)[:, slots])
    else:
        lam = np.zeros(len(nus))
    return beta - nus + pis * rho - lam * slots + mi.sum(1) + np.minimum(mf + lam[:, None], 0.0).sum(1)


def _maximize(w, beta, d, mu, lo, hi, rho, IN, FREE, slots, nu, pi, rn, rp, rounds=14, k=9):
    """Concave 2-D maximisation over (nu, pi >= 0) by a shrinking k x k grid, one batched evaluation per round; the
    grid recentres without shrinking while the best point sits on its edge. Every point is a valid bound."""
    pi = max(pi, 0.0)
    v0 = _dual_batch(w, beta, d, mu, lo, hi, rho, IN, FREE, slots, np.array([nu]), np.array([pi]))[0]
    best = (float(v0), nu, pi)
    g = np.linspace(-1.0, 1.0, k)
    for _ in range(rounds):
        N, Pp = np.meshgrid(best[1] + rn * g, np.maximum(best[2] + rp * g, 0.0))
        N, Pp = N.ravel(), Pp.ravel()
        v = _dual_batch(w, beta, d, mu, lo, hi, rho, IN, FREE, slots, N, Pp)
        i = int(np.argmax(v))
        edge = (abs(N[i] - best[1]) >= 0.99 * rn) or (abs(Pp[i] - best[2]) >= 0.99 * rp and Pp[i] > 0)
        if v[i] > best[0]:
            best = (float(v[i]), float(N[i]), float(Pp[i]))
        if not edge:
            rn, rp = 0.4 * rn, 0.4 * rp
    return best


def node_relaxation(problem, d, OUT, pi_hint, *, grid=8, iters=(300, 1500)):
    """The node's perspective relaxation by the FISTA kernel, excluded assets capped at ``hi = 0`` (one compiled
    program per problem). Held assets stay free: a valid, slightly weaker relaxation. Returns ``(x, nu, pi)`` in model
    units (``nu`` up to sign), or None. It only supplies a tangent point and a multiplier seed."""
    import jax.numpy as jnp
    from heurics_cert.prover import fista as _fista
    fista = _fista.kernels()
    hi = np.asarray(problem.hi, np.float64).copy()
    lo = np.asarray(problem.lo, np.float64).copy()
    hi[OUT] = 0.0
    lo[OUT] = 0.0
    if hi.sum() < 1.0:
        return None
    Q, mu = problem.Q, problem.mu
    dd = np.maximum(np.asarray(d, np.float64), 0.0)
    alpha = float(np.mean(np.diag(Q)))
    mscale = max(float(np.max(np.abs(mu))), 1e-12)
    f = jnp.float32
    args = (jnp.asarray((Q - np.diag(dd)) / alpha, f), jnp.asarray(dd / alpha, f), jnp.asarray(lo, f),
            jnp.asarray(hi, f), float(problem.K), jnp.asarray(mu / mscale, f), float(problem.rho / mscale))
    h = max(float(pi_hint) * mscale / alpha, 1e-6)
    pis = np.concatenate([[0.0], np.linspace(0.3 * h, 3.0 * h, grid - 1)])
    x, _, _, _ = fista(*args, jnp.asarray(pis, f), iters[0])
    x = np.asarray(x, np.float64)
    r = x @ (mu / mscale) - problem.rho / mscale
    ok = np.flatnonzero(r >= 0.0)
    p = float(pis[int(ok[0])] if ok.size else pis[-1])
    xs, _, g, _ = fista(*args, jnp.asarray([p], f), iters[1])
    xs, g = np.asarray(xs, np.float64)[0], np.asarray(g, np.float64)[0]
    if not np.all(np.isfinite(xs)) or xs.sum() <= 0:
        return None
    xs[OUT] = 0.0
    xs = np.maximum(xs, 0.0)
    xs /= xs.sum()
    free = (xs > 1e-9) & (xs < hi - 1e-9)
    nu = float(np.median(g[free])) * alpha if free.any() else 0.0
    return xs, nu, p * alpha / mscale


class BranchAndBound:
    """``BranchAndBound(problem, witness, x_root, x_inc, f_inc).run()``, then ``.certificate()``.

    ``witness`` is the root certificate's witness (its ``d`` is the split the tree uses) and ``x_root`` its tangent
    point; ``x_inc, f_inc`` an optional starting incumbent. The search stops when the certified bound is within
    ``rel_tol`` of the incumbent, or at ``time_limit`` / ``node_limit``."""

    def __init__(self, problem, witness, x_root, x_inc=None, f_inc=np.inf, *, rel_tol=1e-6, time_limit=600.0,
                 node_limit=10 ** 6, node_qp_max=300, node_relax=True, node_relax_min_free=20):
        self.node_relax, self.node_relax_min_free = node_relax, node_relax_min_free
        self.problem, self.witness = problem, witness
        self.d = np.asarray(witness.d, np.float64)
        self.incumbent, self.incumbent_value = x_inc, (float(f_inc) if x_inc is not None else np.inf)
        self.tol, self.tl, self.nl, self.node_qp_max = rel_tol, time_limit, node_limit, node_qp_max
        scale = (float(np.max(np.abs(witness.w)))
                 + 2.0 * float(np.max(np.abs(self.d))) * max(float(np.max(problem.hi)), 1.0))
        self.rn = 2e-2 * scale                          # initial half-widths of the (nu, pi) multiplier grid
        self.rp = self.rn / max(float(np.max(np.abs(problem.mu))), 1e-12)
        self.zero = np.zeros(problem.n)
        self.nodes, self.term, self.split = [], {}, {}
        xr = np.asarray(x_root, np.float64)
        self.tangents = [(np.flatnonzero(xr).tolist(), xr[xr != 0].tolist())]   # tangent 0 = the root's point
        self.stats = dict(nodes=0, incumbents=0, node_tangent_wins=0, unresolved=0, fixed=0, relax_calls=0,
                          relax_wins=0)
        self.inc_every = 10 if problem.K >= 300 else 1
        self.traj = []

    # -- record -------------------------------------------------------------------------------------------------
    def _new(self, parent, var, side):
        self.nodes.append([parent, var, side])
        return len(self.nodes) - 1

    def _branch(self, nid, j):
        self.split[nid] = int(j)
        return self._new(nid, int(j), 1), self._new(nid, int(j), 0)

    def _tan(self, x):
        S = np.flatnonzero(x)
        self.tangents.append((S.tolist(), x[S].tolist()))
        return len(self.tangents) - 1

    @staticmethod
    def _proof(at, nb):
        return dict(kind="bound", at=int(at), t=int(nb["tid"]), nu=float(nb["nu"]), pi=float(nb["pi"]),
                    lam=float(nb["lam"]), lo0=bool(nb.get("lo0", False)), val=float(nb["val"]))

    # -- bounds -------------------------------------------------------------------------------------------------
    def _tangent_at(self, x_hat):
        S = np.flatnonzero(x_hat)
        Ax = self.problem.Q[:, S] @ x_hat[S] - self.d * x_hat
        return 2.0 * Ax, -float(x_hat @ Ax)

    def bound(self, IN, OUT, nu0, pi0):
        """The node bound for held set IN and excluded set OUT, maximised over (nu, pi) from the seed (nu0, pi0).

        Candidates: the root tangent, and (when something is excluded and the node is small enough) the tangent at
        the continuous QP optimum over IN u FREE, seeded by that QP's own multipliers. Returns None when IN alone
        exceeds K, else a dict with the bound `val`, its multipliers, tangent id `tid`, FREE, and `tf` = m_i + lam
        on FREE (the reduced costs that drive fixing and branching)."""
        P = self.problem
        mask = np.ones(P.n, bool)
        mask[IN] = False
        mask[OUT] = False
        FREE = np.flatnonzero(mask)
        slots = P.K - IN.size
        if slots < 0:
            return None
        J = np.concatenate([IN, FREE])
        cands = [(0, self.witness.w, self.witness.beta_z, None, (nu0, pi0))]
        if OUT.size and J.size <= self.node_qp_max:
            xs, nu_q, pi_q, _ = active_set.solve(P.Q[np.ix_(J, J)], P.mu[J], P.hi[J], P.rho,
                                                 np.full(J.size, 1.0 / J.size))
            if np.isfinite(xs).all() and xs.sum() > 0:
                xh = np.zeros(P.n)
                xh[J] = xs
                w, b = self._tangent_at(xh)
                # active_set's stationarity is 2Qx - nu - pi mu = 0: ours with nu -> -nu
                cands.append((None, w, b, xh, (-float(nu_q), max(float(pi_q), 0.0))))
        best = None
        for tid, w, b, xh, (s_nu, s_pi) in cands:
            v, nu, pi = _maximize(w, b, self.d, P.mu, P.lo, P.hi, P.rho, IN, FREE, slots, s_nu, s_pi, self.rn, self.rp)
            if tid is None:
                v2 = _maximize(w, b, self.d, P.mu, P.lo, P.hi, P.rho, IN, FREE, slots, nu0, pi0, self.rn, self.rp)
                if v2[0] > v:
                    v, nu, pi = v2
            if best is None or v > best[0]:
                best = (v, nu, pi, tid, w, b, xh)
        v, nu, pi, tid, w, b, xh = best
        if tid is None:
            tid = self._tan(xh)
            self.stats["node_tangent_wins"] += 1
        _, lam, tf, _ = _node_dual(w, b, self.d, P.mu, P.lo, P.hi, P.rho, IN, FREE, slots, nu, pi)
        return dict(val=v, nu=nu, pi=pi, lam=lam, tid=tid, FREE=FREE, tf=tf, slots=slots)

    def leaf_bound(self, IN, nb):
        """Fully decided node: tangents at the support QP's exact optimum (with lo) and at the lo-relaxed optimum."""
        P = self.problem
        empty = np.array([], np.int64)
        slots = P.K - IN.size
        QI, muI, loI, hiI = P.Q[np.ix_(IN, IN)], P.mu[IN], P.lo[IN], P.hi[IN]
        xs = primal.box_qp_approx(QI, muI, loI, hiI, P.rho)
        if xs is None:
            return None, None                                  # budget or return unreachable on this support
        pts = []                                              # (support point, full lo vector, KKT multipliers, lo0)
        for lo_v, lo_full, lo0 in ((loI, P.lo, False), (np.zeros(IN.size), self.zero, True)):
            x0 = primal.project_box_simplex(xs, lo_v, hiI) if lo0 else xs
            kk = primal.box_qp_exact(QI, muI, lo_v, hiI, P.rho, x0, multipliers=True)
            pts.append((kk[0], lo_full, (kk[1], kk[2]), lo0) if kk is not None else (x0, lo_full, None, lo0))
        out = []
        for xs_p, lo, mult, lo0 in pts:
            xp = np.zeros(P.n)
            xp[IN] = xs_p
            w, b = self._tangent_at(xp)
            runs = [_maximize(w, b, self.d, P.mu, lo, P.hi, P.rho, IN, empty, slots, nb["nu"], nb["pi"],
                              50 * self.rn, 50 * self.rp, rounds=24)]
            if mult is not None:                               # KKT multipliers: exact up to rounding; polish locally
                runs.append(_maximize(w, b, self.d, P.mu, lo, P.hi, P.rho, IN, empty, slots,
                                      mult[0], max(mult[1], 0.0),
                                      1e-4 * self.rn + 1e-6 * abs(mult[0]), 1e-4 * self.rp + 1e-6 * abs(mult[1]),
                                      rounds=10))
            v, nu, pi = max(runs)
            out.append(dict(val=v, nu=nu, pi=pi, lam=0.0, xp=xp, lo0=lo0))
        best = max(out, key=lambda r: r["val"])
        best["tid"] = self._tan(best.pop("xp"))
        x = np.zeros(P.n)
        x[IN] = pts[0][0]
        return best, x

    def relax_bound(self, IN, OUT, FREE, nb):
        """Tangent at the node's own perspective relaxation (excluded assets capped at 0), multipliers seeded from
        the relaxation (both budget signs) and from the node's current ones. Returns an nb-like dict or None."""
        P = self.problem
        self.stats["relax_calls"] += 1
        nr = node_relaxation(P, self.d, OUT, max(float(nb["pi"]), 1e-12))
        if nr is None:
            return None
        xh, nu_r, pi_r = nr
        w, b = self._tangent_at(xh)
        slots = P.K - IN.size
        seeds = ((nu_r, pi_r), (-nu_r, pi_r), (nb["nu"], nb["pi"]))      # both budget-sign conventions
        v, nu, pi = max(_maximize(w, b, self.d, P.mu, P.lo, P.hi, P.rho, IN, FREE, slots, s_nu, s_pi,
                                  self.rn, self.rp, rounds=20)
                        for s_nu, s_pi in seeds)
        _, lam, tf, _ = _node_dual(w, b, self.d, P.mu, P.lo, P.hi, P.rho, IN, FREE, slots, nu, pi)
        return dict(val=v, nu=nu, pi=pi, lam=lam, tid=self._tan(xh), FREE=FREE, tf=tf, slots=slots)

    def _offer(self, x):
        if x is not None and self.problem.feasibility(x)["feasible"]:
            f = self.problem.objective(x)
            if f < self.incumbent_value * (1 - 1e-12):
                self.incumbent, self.incumbent_value = x, f
                self.stats["incumbents"] += 1

    def try_incumbent(self, IN, FREE, tf, slots):
        order = FREE[np.argsort(tf)][: max(slots, 0)]
        S = set(IN.tolist()) | set(order.tolist())
        if S:
            x, _ = primal.restricted_qp(self.problem, S, self.incumbent, time.perf_counter() + 2.0)
            self._offer(x)

    def _resolve_leaf(self, nid, IN, nb, best):
        """A fully decided node becomes a terminal at once: its support QP, solved exactly, is offered as incumbent
        and gives the bound (queued instead, such leaves can sit behind plateaus of equal keys)."""
        lb, x = self.leaf_bound(IN, nb)
        if lb is None:
            self.term[nid] = dict(kind="support")
            return
        self._offer(x)
        own = self._proof(nid, lb)
        self.term[nid] = own if own["val"] > best["val"] else best
        if self.term[nid]["val"] < self.incumbent_value - self.tol * abs(self.incumbent_value):
            self.stats["unresolved"] += 1

    # -- search -------------------------------------------------------------------------------------------------
    def run(self):
        t0 = time.perf_counter()
        cnt = itertools.count()
        e = np.array([], np.int64)
        root = self._new(-1, -1, -1)
        nb = self.bound(e, e, self.witness.nu, self.witness.pi)
        # key: (bound, -depth, arrival): among equal bounds -- plateaus of an inherited proof -- go deeper first
        heap = [(nb["val"], 0, next(cnt), root, e, e, nb, self._proof(root, nb))]
        while heap:
            if time.perf_counter() - t0 > self.tl or self.stats["nodes"] >= self.nl:
                break
            L, _, _, nid, IN, OUT, nb, best = heapq.heappop(heap)
            self.stats["nodes"] += 1
            if self.stats["nodes"] % 25 == 1:
                self.traj.append((round(time.perf_counter() - t0, 2), self.stats["nodes"], float(L),
                                  float(self.incumbent_value), len(heap) + 1))
            thr = self.incumbent_value - self.tol * abs(self.incumbent_value)
            if L >= thr:
                self.term[nid] = best
                continue
            FREE, tf, Ln = nb["FREE"], nb["tf"], nb["val"]
            to_out = FREE[Ln + np.maximum(tf, 0.0) >= thr]
            to_in = FREE[(Ln + np.maximum(-tf, 0.0) >= thr) & ~np.isin(FREE, to_out)]
            if to_out.size or to_in.size:
                cur = nid
                for j in to_out:
                    ci, co = self._branch(cur, j)
                    self.term[ci] = dict(self._proof(ci, nb), val=Ln + max(float(tf[FREE == j][0]), 0.0))
                    cur = co
                for j in to_in:
                    ci, co = self._branch(cur, j)
                    self.term[co] = dict(self._proof(co, nb), val=Ln + max(-float(tf[FREE == j][0]), 0.0))
                    cur = ci
                self.stats["fixed"] += to_out.size + to_in.size
                IN, OUT, nid = np.union1d(IN, to_in), np.union1d(OUT, to_out), cur
                nb = self.bound(IN, OUT, nb["nu"], nb["pi"])
                if nb is None:
                    self.term[nid] = dict(kind="card")
                    continue
                own = self._proof(nid, nb)
                best = own if own["val"] > best["val"] else best
                L = best["val"]
                if L >= thr:
                    self.term[nid] = best
                    continue
            FREE = nb["FREE"]
            if (self.node_relax and FREE.size > self.node_relax_min_free and nb["val"] <= best["val"]):
                # the node's own bound does not beat what it inherited: try its own perspective relaxation
                nr = self.relax_bound(IN, OUT, FREE, nb)
                if nr is not None and nr["val"] > nb["val"]:
                    nb = nr
                    if nr["val"] > best["val"]:
                        self.stats["relax_wins"] += 1
                        best = self._proof(nid, nr)
                        L = best["val"]
                        if L >= thr:
                            self.term[nid] = best
                            continue
            FREE, tf = nb["FREE"], nb["tf"]
            if self.stats["nodes"] % self.inc_every == 0:
                self.try_incumbent(IN, FREE, tf, nb["slots"])
                thr = self.incumbent_value - self.tol * abs(self.incumbent_value)
            if FREE.size == 0:
                self._resolve_leaf(nid, IN, nb, best)
                continue
            j = int(FREE[np.argmin(np.abs(tf))])
            for side, child in zip((1, 0), self._branch(nid, j)):
                IN_c, OUT_c = (np.union1d(IN, [j]), OUT) if side else (IN, np.union1d(OUT, [j]))
                nc = self.bound(IN_c, OUT_c, nb["nu"], nb["pi"])
                if nc is None:
                    self.term[child] = dict(kind="card")
                    continue
                own = self._proof(child, nc)
                b = own if own["val"] > best["val"] else best
                if nc["FREE"].size == 0:
                    self._resolve_leaf(child, IN_c, nc, b)
                    continue
                heapq.heappush(heap, (b["val"], -(IN_c.size + OUT_c.size), next(cnt), child, IN_c, OUT_c, nc, b))
        for _, _, _, nid, _, _, _, best in heap:               # open at the limit: terminal with its proof
            self.term[nid] = best
        thr = self.incumbent_value - self.tol * abs(self.incumbent_value)
        vals = [p["val"] for p in self.term.values() if p["kind"] == "bound"]
        U = self.incumbent_value
        lb = min(vals + [U])
        gap_pct = 100 * (U - lb) / U if np.isfinite(U) else np.inf
        return dict(U=U, LB=lb, gap_pct=gap_pct, closed=lb >= thr, open=len(heap), secs=time.perf_counter() - t0,
                    **self.stats)

    def record(self) -> dict:
        """The CCPO-TREE/0 record: the search, the model fingerprint and the root split."""
        x = np.zeros(self.problem.n) if self.incumbent is None else np.asarray(self.incumbent, np.float64)
        S = np.flatnonzero(x)
        return dict(format=FORMAT, model_sha256=self.problem.fingerprint, root_d=self.d.tolist(),
                    nodes=self.nodes, split={str(k): v for k, v in self.split.items()},
                    term={str(k): v for k, v in self.term.items()}, tangents=self.tangents,
                    incumbent=dict(idx=S.tolist(), val=x[S].tolist()), U=float(self.incumbent_value), rel_tol=self.tol)

    def certificate(self):
        """The record as a ``TreeCertificate``."""
        from heurics_cert.certificates.tree import TreeCertificate
        return TreeCertificate(self.record())
