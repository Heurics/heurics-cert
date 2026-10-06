"""Rebuild the equity panel the paper used (``returns.npy``, 260 weeks x 2,901 stocks) from Yahoo Finance.

The returns themselves cannot be redistributed (Yahoo Finance's terms of use), so this repository ships the
recipe and what pins it down instead:

* ``symbols.json``, ``exchange.json`` -- the 2,901 tickers and their listing exchange (public facts);
* ``weeks.json`` -- the 260 ISO weeks (2021-09 to 2026-09) the returns refer to;
* ``fingerprints.tsv`` -- SHA-256 of the full returns matrix and of every ticker's column.

Pipeline (the published build's, minus the universe selection, which ``symbols.json`` now fixes): for each ticker,
Yahoo's weekly chart (adjusted close) over the window; bars bucketed by ISO (year, week), because Yahoo's bar
timestamps jitter by exchange and daylight saving; the in-progress week dropped; prices aligned to the week grid;
simple returns ``R[t] = P[t+1] / P[t] - 1``. Each column is then compared with its fingerprint.

Yahoo rewrites *adjusted* closes when a company later pays a dividend or splits, so a pull made after September 2026
can differ from ours in the history it adjusts. A column whose fingerprint does not match is reported as differing
(dividend payers are the usual reason); nothing computed from the prices ships to say by how much, since that would
redistribute derived data. The matrix fingerprint matches only if every column does. The end-to-end check is to
rerun the method on the rebuilt panel and compare with the paper's numbers.

    python build_returns.py --out returns.npy [--tickers AAPL,MSFT] [--pause 0.4]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URL = ("https://query2.finance.yahoo.com/v8/finance/chart/{t}?period1={a}&period2={b}&interval=1wk"
       "&events=div%2Csplits&includeAdjustedClose=true")


def col_hash(v) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, "<f8").tobytes()).hexdigest()


def fetch(ticker: str, a: int, b: int) -> dict:
    req = urllib.request.Request(URL.format(t=ticker, a=a, b=b), headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]


def weekly_prices(res: dict, grid: list[tuple[int, int]]) -> np.ndarray:
    """Adjusted close per ISO week of `grid` (NaN where missing); the last bar of a week wins."""
    ts = res.get("timestamp") or []
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    by_week = {}
    for t, p in zip(ts, adj):
        if p is None:
            continue
        iso = dt.datetime.fromtimestamp(t, dt.timezone.utc).isocalendar()
        by_week[(iso[0], iso[1])] = float(p)
    return np.array([by_week.get(w, np.nan) for w in grid])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="returns.npy")
    ap.add_argument("--tickers", default="", help="comma list: rebuild only these columns (a spot check)")
    ap.add_argument("--pause", type=float, default=0.4, help="seconds between requests")
    a = ap.parse_args()
    symbols = json.load(open(os.path.join(HERE, "symbols.json")))
    weeks = [tuple(w) for w in json.load(open(os.path.join(HERE, "weeks.json")))]
    fp = dict(l.rstrip("\n").split("\t")[::-1][:2][::-1] for l in open(os.path.join(HERE, "fingerprints.tsv"))
              if not l.startswith("kind"))
    # price grid: the week before the first return week, then every return week (R[t] = P[t+1] / P[t] - 1)
    first = dt.date.fromisocalendar(weeks[0][0], weeks[0][1], 1) - dt.timedelta(weeks=1)
    grid = [tuple(first.isocalendar())[:2]] + weeks
    a0 = int(dt.datetime.combine(first - dt.timedelta(days=7), dt.time(), dt.timezone.utc).timestamp())
    last = dt.date.fromisocalendar(weeks[-1][0], weeks[-1][1], 7) + dt.timedelta(days=8)
    b0 = int(dt.datetime.combine(last, dt.time(), dt.timezone.utc).timestamp())
    want = [s for s in a.tickers.split(",") if s] or symbols
    R = np.full((len(weeks), len(symbols)), np.nan)
    ok = bad = 0
    for s in want:
        j = symbols.index(s)
        try:
            P = weekly_prices(fetch(s, a0, b0), grid)
        except Exception as exc:                                    # noqa: BLE001
            print(f"{s}: fetch failed ({exc})")
            bad += 1
            continue
        R[:, j] = P[1:] / P[:-1] - 1.0
        match = fp.get(f"returns column {s}") == col_hash(R[:, j])
        ok += match
        if match:
            print(f"{s:<8} identical (exact fingerprint)", flush=True)
        else:
            print(f"{s:<8} differs from our pull (expected for dividend payers: Yahoo re-adjusts their history)"
                  + (f"  [missing weeks {int(np.isnan(R[:, j]).sum())}]" if np.isnan(R[:, j]).any() else ""),
                  flush=True)
        time.sleep(a.pause)
    np.save(a.out, R)
    if len(want) == len(symbols):
        print("matrix fingerprint", "MATCH" if fp.get("returns matrix") == col_hash(R) else "differs")
    print(f"tickers: {ok} identical, {len(want) - ok - bad} differ, {bad} failed to download")


if __name__ == "__main__":
    main()
