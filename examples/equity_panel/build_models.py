"""Rebuild the paper's equity models from the returns, and check every one against its published fingerprint.

    python build_models.py --returns returns.npy                     # check (default)
    python build_models.py --returns returns.npy --riskprism DIR     # also the fundamental-model tier
    python build_models.py --returns returns.npy --write out.tsv     # write the table instead of checking

The models are built with ``heurics_cert.models``, which uses no BLAS in its arithmetic, so the same returns give the
same model bytes on any machine. A row that matches proves you hold exactly the model a certificate was proved on.

Tiers, in the table's order:

* factor -- 189 mandates (3 universes x 7 two-year windows x 9 mandate shapes), 10-factor model, 5 % specific floor;
* shrinkage, identity and constant-correlation targets -- the same 189 mandates on Ledoit-Wolf covariances;
* fundamental -- the 27 mandates of the last window on the Risk Prism snapshot ``model-2026-09-12``
  (``--riskprism``: a directory with the release's ``riskprism-artifacts.tar.gz`` unpacked; needs pandas + pyarrow);
* sensitivity -- 27 index-tracking mandates under 3 factor counts x 3 specific floors;
* sandbox -- the six singular test problems of the copositive table;
* sample -- the 189 mandates on the plain sample covariance (singular: 104 weeks, 1,437-2,901 stocks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from heurics_cert.models import (Problem, factor_form, factor_model, ledoit_wolf, return_target,
                                 sample_covariance, statistics)
from heurics_cert.models._exact import outer_sum

HERE = os.path.dirname(os.path.abspath(__file__))

UNIVERSES = ("nyse", "nasdaq", "all")
_EXCHANGE_TO_UNIVERSE = {"Q": "nasdaq", "N": "nyse", "A": "nyse"}
#: mandate shapes: cardinality and per-name cap; index tracking (cap 1.0) at each K in INDEX_TRACKING_K
FUND_ARCHETYPES = {
    "concentrated": dict(K=30, hi=0.05),
    "diversified_active": dict(K=100, hi=0.05),
    "closet_indexer": dict(K=300, hi=0.10),
    "enhanced_index": dict(K=750, hi=0.10),
}
INDEX_TRACKING_K = (10, 20, 30, 40, 50)
Q_RHO = 0.5                                   # return target: this quantile of the reachable return range
WINDOW, STEP = 104, 26                        # two-year windows, one quarter apart
SENS_FACTORS, SENS_FLOORS = (5, 10, 20), (0.025, 0.05, 0.1)
SANDBOX = ("nyse:52:200:10:0", "nasdaq:26:200:10:0", "nyse:52:300:10:0", "nasdaq:26:300:10:0",
           "nyse:52:400:10:0", "nyse:52:600:10:0")
RISKPRISM_FILES = ("riskprism-artifacts.tar.gz", "exposures.parquet", "factor_covariance.parquet",
                   "specific_risk.parquet")


def eps_for(K: int) -> float:
    """Buy-in floor for cardinality K."""
    return min(0.01, 0.5 / K)


class Panel:
    def __init__(self, returns_path):
        self.returns = np.load(returns_path)
        self.symbols = json.load(open(os.path.join(HERE, "symbols.json")))
        self.exchange = json.load(open(os.path.join(HERE, "exchange.json")))

    def universe(self, uni):
        if uni == "all":
            return self.returns, self.symbols
        mask = np.array([_EXCHANGE_TO_UNIVERSE[e] == uni for e in self.exchange])
        return self.returns[:, mask], [s for s, m in zip(self.symbols, mask) if m]

    def windows(self):
        T = self.returns.shape[0]
        return [(st, st + WINDOW) for st in range(0, T - WINDOW + 1, STEP)]


def mandates(n, index_k=INDEX_TRACKING_K, only=None):
    ms = [(nm, p["K"], p["hi"]) for nm, p in FUND_ARCHETYPES.items() if p["K"] < n]
    ms += [("index_tracking", K, 1.0) for K in index_k]
    return [m for m in ms if only is None or m[0] in only]


def problem(Q, mu, K, hicap):
    n = len(mu)
    lo, hi = np.full(n, eps_for(K)), np.full(n, hicap)
    return Problem(Q=Q, mu=mu, lo=lo, hi=hi, K=K, rho=return_target(mu, lo, hi, K, Q_RHO))


def factor_cells(panel, kfac=10, floor=0.05, windows=None, only=None, index_k=INDEX_TRACKING_K):
    tag = f"factor{kfac}" + ("" if floor == 0.05 else f"f{floor:g}")
    for uni in UNIVERSES:
        R, _ = panel.universe(uni)
        for st, en in panel.windows():
            if windows is not None and st not in windows:
                continue
            mu, Q, _ = factor_model(np.asarray(R[st:en], np.float64), kfac, floor)
            for nm, K, hicap in mandates(len(mu), index_k, only):
                yield f"{tag}-{uni}-w{st}-{nm}-K{K}", problem(Q, mu, K, hicap)


def shrinkage_cells(panel, target):
    tag = {"identity": "lwid", "cc": "lwcc"}[target]
    for uni in UNIVERSES:
        R, _ = panel.universe(uni)
        for st, en in panel.windows():
            Rw = np.asarray(R[st:en], np.float64)
            mu, _ = statistics(Rw)
            Q, _, _ = ledoit_wolf(Rw, target)
            for nm, K, hicap in mandates(len(mu)):
                yield f"{tag}-{uni}-w{st}-{nm}-K{K}", problem(Q, mu, K, hicap)


def riskprism_arrays(directory):
    """The Risk Prism snapshot as arrays: tickers, exposures X, factor covariance F, specific variance."""
    import pandas as pd
    ex = pd.read_parquet(os.path.join(directory, "exposures.parquet"))
    F = pd.read_parquet(os.path.join(directory, "factor_covariance.parquet"))
    sp = pd.read_parquet(os.path.join(directory, "specific_risk.parquet")).loc[ex.index]
    return ([str(t) for t in ex.index], ex.values.astype(np.float64), F.values.astype(np.float64),
            sp["specific_vol"].values.astype(np.float64) ** 2)


def fundamental_cells(panel, directory, start=156):
    tickers, X_all, F, spec_var = riskprism_arrays(directory)
    row = {t: i for i, t in enumerate(tickers)}
    for uni in UNIVERSES:
        R, symbols = panel.universe(uni)
        keep = [j for j, s in enumerate(symbols) if s in row]
        idx = np.array([row[symbols[j]] for j in keep])
        mu, _ = statistics(np.asarray(R[start:start + WINDOW][:, keep], np.float64))
        d = spec_var[idx] / 52.0
        Q = factor_form(X_all[idx], F / 52.0, d)
        for nm, K, hicap in mandates(len(mu)):
            yield f"prism-{uni}-w{start}-{nm}-K{K}", problem(Q, mu, K, hicap)


def sensitivity_cells(panel):
    for k in SENS_FACTORS:
        for f in SENS_FLOORS:
            yield from factor_cells(panel, k, f, windows={0, 78, 156}, only={"index_tracking"}, index_k=(10, 30, 50))


def sandbox_problem(panel, uni, start, n_sub, K, seed):
    R, _ = panel.universe(uni)
    R = R[start:start + WINDOW]
    idx = np.sort(np.random.default_rng(seed).choice(R.shape[1], n_sub, replace=False))
    mu, Xc = statistics(np.asarray(R[:, idx], np.float64))
    return problem(outer_sum(Xc) / (Xc.shape[0] - 1.0), mu, K, 1.0)


def sample_cells(panel):
    for uni in UNIVERSES:
        R, _ = panel.universe(uni)
        for st, en in panel.windows():
            mu, Q = sample_covariance(np.asarray(R[st:en], np.float64))
            for nm, K, hicap in mandates(len(mu)):
                yield f"bulk_market-{uni}-w{st}-{nm}-K{K}", problem(Q, mu, K, hicap)


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, "<f8").tobytes()).hexdigest()


def _file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rows(panel, returns_path, riskprism=None):
    """The fingerprint table, row by row: ``(kind, name, n, sha256)``."""
    yield ("raw file", "returns.npy", "", _file(returns_path))
    for f in ("symbols.json", "exchange.json", "weeks.json"):
        yield ("raw file", f, "", _file(os.path.join(HERE, f)))
    seen = set()
    for cell, P in factor_cells(panel):
        key = "-".join(cell.split("-")[1:3])
        if key not in seen:
            seen.add(key)
            yield ("factor covariance (k=10, floor 5%)", key, P.n, _h(P.Q))
        yield ("factor cell model fingerprint", cell, P.n, P.fingerprint)
    tiers = [("shrinkage, identity target", shrinkage_cells(panel, "identity")),
             ("shrinkage, constant correlation", shrinkage_cells(panel, "cc"))]
    if riskprism:
        tiers.append(("fundamental model (Risk Prism model-2026-09-12)", fundamental_cells(panel, riskprism)))
    for model, gen in tiers:
        seen = set()
        for cell, P in gen:
            key = "-".join(cell.split("-")[:3])
            if key not in seen:
                seen.add(key)
                yield (f"{model} covariance", key, P.n, _h(P.Q))
            yield (f"{model} cell model fingerprint", cell, P.n, P.fingerprint)
    if riskprism:
        for f in RISKPRISM_FILES:
            if os.path.exists(os.path.join(riskprism, f)):
                yield ("Risk Prism download", f, "", _file(os.path.join(riskprism, f)))
    seen = set()
    for cell, P in sensitivity_cells(panel):
        key = "-".join(cell.split("-")[:3])
        if key not in seen:
            seen.add(key)
            yield ("sensitivity covariance", key, P.n, _h(P.Q))
        yield ("sensitivity cell model fingerprint", cell, P.n, P.fingerprint)
    for c in SANDBOX:
        u, s, n, K, sd = c.split(":")
        P = sandbox_problem(panel, u, int(s), int(n), int(K), int(sd))
        yield ("sandbox cell model fingerprint", f"sb-{u}-w{s}-n{n}-K{K}-s{sd}", P.n, P.fingerprint)
    seen = set()
    for cell, P in sample_cells(panel):
        key = "-".join(cell.split("-")[1:3])
        if key not in seen:
            seen.add(key)
            yield ("sample covariance", key, P.n, _h(P.Q))
        yield ("sample cell model fingerprint", cell, P.n, P.fingerprint)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--returns", default=os.path.join(HERE, "returns.npy"))
    ap.add_argument("--riskprism", help="directory of the unpacked Risk Prism release model-2026-09-12")
    ap.add_argument("--write", help="write the table here instead of checking it")
    a = ap.parse_args()
    panel = Panel(a.returns)
    expected = {}
    for line in open(os.path.join(HERE, "model_fingerprints.tsv")).read().splitlines()[1:]:
        kind, name, _, digest = line.split("\t")
        expected[(kind, name)] = digest
    out = open(a.write, "w") if a.write else None
    if out:
        out.write("kind\tname\tn\tsha256\n")
    match = differ = 0
    for r in rows(panel, a.returns, a.riskprism):
        if out:
            out.write("\t".join(str(v) for v in r) + "\n")
            continue
        ok = expected.get((r[0], r[1])) == r[3]
        match += ok
        differ += not ok
        if not ok:
            print(f"DIFFERS  {r[0]}: {r[1]}")
    if not out:
        print(f"{match} fingerprints match, {differ} differ")
        raise SystemExit(0 if differ == 0 else 1)


if __name__ == "__main__":
    main()
