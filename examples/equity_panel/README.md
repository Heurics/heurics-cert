# The equity panel: a recipe, not the data

The paper's 189 equity mandates are built from weekly adjusted closes of 2,901 US stocks (NYSE and NASDAQ common stocks
with a complete five-year history to September 2026) taken from Yahoo Finance. Yahoo's terms forbid redistributing that
data or works derived from it, so this folder ships what lets you rebuild it and confirm you hold exactly what the paper
used, and nothing computed from the prices.

| file | what it is |
|---|---|
| `build_returns.py` | the recipe: downloads each ticker's weekly chart, aligns it to the week grid, computes simple returns, and checks every column against `fingerprints.tsv` |
| `symbols.json`, `exchange.json`, `weeks.json` | the 2,901 tickers, their listing exchanges, the 260 ISO weeks (public facts) |
| `fingerprints.tsv` | SHA-256 of the returns matrix and of each ticker's column |
| `build_models.py` | builds every model in the paper from the returns with `heurics_cert.models` and checks it against `model_fingerprints.tsv` |
| `model_fingerprints.tsv` | SHA-256 of 1,208 models and inputs: the sample-covariance tier, the factor tier, both Ledoit-Wolf shrinkage tiers, the fundamental-model tier, the sensitivity panel, the singular test problems, and the raw and Risk Prism input files |

## Rebuild and check

```
python build_returns.py --out returns.npy                 # rate-limited: about 20 minutes for all 2,901 tickers
python build_models.py --returns returns.npy              # rebuild every model, compare every fingerprint
python build_models.py --returns returns.npy --riskprism DIR   # also the fundamental-model tier (pandas, pyarrow)
```

For the fundamental-model tier, download the public Risk Prism release `model-2026-09-12`
(`riskprism-artifacts.tar.gz`, github.com/wanxinwanxin/risk-prism) and unpack it into `DIR`; its file hashes are in the
table too.

## What to expect

**Models are bit-reproducible.** `heurics_cert.models` uses no BLAS in its arithmetic, so the same returns give the
same model bytes, and the same fingerprint, on any machine, thread count or numpy version.

**Returns may not be.** Yahoo rewrites a stock's adjusted history each time it pays a dividend or splits. Checked two
weeks after our pull, stocks without dividends rebuilt identically; dividend payers differed by at most 8e-7 in a
weekly return (rounding), and the recipe reports them as differing. A rebuild that differs in any column gives models
that agree with ours to about five significant digits but not bit for bit, so their fingerprints will not match and
our certificates (bound to exact fingerprints) will not check against them. Rerunning the method on such a rebuild
reproduces the paper's numbers at the precision they are reported.

The paper's certificates for these problems are derived from the data and are not published; they are available to
reviewers.
