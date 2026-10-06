# heurics-cert

Checkable optimality certificates for cardinality-constrained portfolio optimization.

Hand someone a portfolio of K stocks and a claim that it is close to the minimum-variance choice. heurics-cert turns
that claim into a **certificate**: a lower bound on the optimum together with a short witness from which anyone can
recompute the bound with rigorous arithmetic, without trusting the program that produced it. Two independent checkers
(Python and C99, sharing no code) and an exact rational tier confirm every certificate.

```python
import heurics_cert as hc

problem = hc.Problem(Q, mu, lo, hi, K, rho)      # min x'Qx  s.t. 1'x = 1, mu'x >= rho, buy-in/caps, at most K names
result = hc.prove(problem)                       # relaxation -> witness -> independent checkers

result.status                                    # Status.PROVED
result.bound                                     # a certified lower bound on the optimum
result.gap(x)                                    # the certified gap of any candidate portfolio (validated first)
result.certificate.save("cert.json")

hc.verify("cert.json", method="c").status        # anyone, later, with only the file and a checker
```

## Install

```
git clone https://github.com/heurics/heurics-cert && cd heurics-cert
pip install .                         # load and check certificates (numpy only)
pip install ".[prove]"                # produce them: the FISTA prover, batched GPU and copositive provers (JAX)
pip install ".[reference]"            # the interior-point relaxation solver (Clarabel), an independent cross-check
heurics-cert build-checker            # compile the C99 checker (gcc, clang or MSVC) and self-test it
```

`heurics-cert info` reports the version and whether a runnable C checker was found.

Or run every check in a clean Linux container, with nothing installed locally but Docker:

```
docker build -t heurics-cert .
docker run --rm heurics-cert          # tests, adversarial suite, Python/C cross-check, quickstart
```

## Concepts

**The problem** is cardinality-constrained mean-variance with buy-in thresholds:

    min x'Qx   s.t.  1'x = 1,  mu'x >= rho,  lo_i y_i <= x_i <= hi_i y_i,  1'y <= K,  y in {0,1}^n

**A certificate** (CCPO-CERT/1) carries the model, a diagonal split `d`, a tangent `(w, beta_z)` and three
multipliers. The checker proves `Q - diag(d)` positive semidefinite, proves `beta_z` bounds the inner minimum, and
re-derives the bound with outward rounding. Validity never depends on the prover: an inaccurate relaxation solve
costs tightness, never soundness. A copositive certificate (CCPO-CERT/2) adds `sigma` and `G_off >= 0`, which escape
the diagonal ceiling on singular covariances. Specification: [docs/CERTIFICATE_FORMAT.md](docs/CERTIFICATE_FORMAT.md).

**A fingerprint** (SHA-256 of the model's IEEE bytes) travels with every verdict. A reader who holds the model
recomputes it, so a certificate about a different (e.g. more constrained) problem is caught: `expect_model=` yields
`MODEL_MISMATCH`.

**A tree certificate** (CCPO-TREE/0) records a branch-and-bound search that proves an optimum; its checker re-derives
every node bound with a-priori error bounds. It is a research instrument for measuring how close bounds are to the
truth, not part of certifying a portfolio. Specification: [docs/TREE_CERTIFICATE.md](docs/TREE_CERTIFICATE.md).

**Verdicts** are `PROVED`, `REFUTED` (a certified negation), `NOT PROVED` (undecided at this precision; never a
refutation) and `MODEL_MISMATCH`; tree checks end in `OPTIMAL`, `BOUND`, `REFUTED` or `NOT PROVED`. Checkers answer
every input, malformed ones included, with a verdict rather than an exception.

## The library

| package | contents |
|---|---|
| `hc.models` | `Problem`; bit-reproducible risk models: `factor_model`, `ledoit_wolf`, `factor_form`, `sample_covariance`, `return_target` |
| `hc.certificates` | `Certificate`, `TreeCertificate`, `Witness`, `load` |
| `hc.prover` | `prove`, `prove_batch` (many problems on one singular covariance, GPU), `prove_copositive`, `extract` |
| `hc.checkers` | `verify`, `check_tree`; modules `bound` (rigorous + exact), `copositive`, `tree`, `native` (C99), `qbin` |
| `hc.search` | `BranchAndBound` (proves optima, emits tree certificates), `polish` (feasible portfolios) |
| `hc.extras` | certificates for mean-CVaR (`cvar`) and lot-constrained index tracking (`index_tracking`) |

`hc.Params` collects the prover's settings (relaxation solver, which checkers must agree, effort); none affects
soundness. `hc.Result.spread`, the relaxation's weight outside its K largest positions, predicts tightness: below
about 2 % the certificate is reliably within 1 % of the optimum.

The checkers import nothing from the prover side (a test enforces it), and `checkers/bound.py` runs as a standalone
script on stdlib plus numpy, its exact tier on stdlib alone:

```
python heurics_cert/checkers/bound.py cert.json --exact
```

Risk models are built without BLAS, so the same returns give the same model bytes, and the same fingerprint, on any
machine.

## Command line

```
heurics-cert prove problem.npz -o cert.json [--qbin model.qbin] [--solver fista|clarabel]
heurics-cert verify cert.json [--method rigorous|exact|c] [--expect-model PREFIX]
heurics-cert check-tree tree.json.gz --certificate cert.json
heurics-cert info
```

Exit codes: 0 proved or optimal, 1 refuted, 2 model mismatch, 3 not proved or unavailable.

## Examples and data

* `examples/quickstart.py` -- certify a small problem end to end: prove, save, re-check from the file, certified gap.
* `examples/certificates/` -- 16 certificates to run the checkers on with no prover step.
* `examples/equity_panel/` -- the paper's equity dataset as a recipe: rebuild the returns from the public source, then
  `build_models.py` rebuilds every model in the paper and checks it against its published SHA-256 fingerprint (the
  data itself cannot be redistributed).

## Development

```
pip install -e ".[test]"
pytest
python tools/adversarial.py      # mutations every checker must reject
python tools/cross_check.py      # the Python and C checkers agree on every certificate and forgery
```

## Licence

Apache License 2.0; see [LICENSE](LICENSE).

